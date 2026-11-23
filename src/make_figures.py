"""Generate 4 camera-ready publication figures and auto-verified summary tables from committed artefacts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluate import compute_roc_auc_pure
from src.phase4_disentangle import evaluate_incremental_vs_probability

# Colorblind-safe Okabe-Ito palette
C_BLUE = "#0072B2"
C_VERM = "#D55E00"
C_GREEN = "#009E73"
C_PURPLE = "#CC79A7"
C_ORANGE = "#E69F00"
C_SKY = "#56B4E9"
C_GREY = "#595959"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _audit_phase4a_labels(items: list[dict[str, Any]]) -> dict[str, int]:
    """Verify Phase 4A label invariants across all rows (ruling out label-mapping bugs)."""
    mismatches = 0
    for r in items:
        expected_pff = int(r["prior_choice"] != r["world_truth"])
        expected_true = int(r["answer_choice"] == r["world_truth"])
        if int(r["prior_favors_false"]) != expected_pff or int(r["is_true"]) != expected_true or int(r["label"]) != (1 - expected_true):
            mismatches += 1
    return {"n_rows": len(items), "label_consistency_mismatches": mismatches}


def collect_verified_tables(results_root: Path = Path("results")) -> dict[str, Any]:
    """Read committed Phase 1-4 JSON/JSONLs and compute exact manuscript tables with zero hand-transcription."""
    q_rev = _load_json(results_root / "phase1_phase2_qwen3b/reversible_v1/summary.json")
    l_rev = _load_json(results_root / "cross_family_llama32_3b/reversible_v1/summary.json")
    q_p3 = _load_json(results_root / "phase3_surprisal_qwen3b/summary.json")
    l_p3 = _load_json(results_root / "cross_family_llama32_3b/surprisal_confound_v1/summary.json")
    q_st22 = _load_json(results_root / "phase1_phase2_qwen3b/steer_v1/steer_L22_decision_only_summary.json")
    q_st24 = _load_json(results_root / "phase1_phase2_qwen3b/steer_v1/steer_L24_decision_only_summary.json")
    l_st19 = _load_json(results_root / "cross_family_llama32_3b/steer_v1/steer_L19_decision_only_summary.json")

    q_p4a_sum = _load_json(results_root / "phase4/qwen3b_disentangle/summary.json")
    l_p4a_sum = _load_json(results_root / "phase4/llama32_3b_disentangle/summary.json")
    q_p4a_items = _load_jsonl(results_root / "phase4/qwen3b_disentangle/per_item_scores.jsonl")
    l_p4a_items = _load_jsonl(results_root / "phase4/llama32_3b_disentangle/per_item_scores.jsonl")

    q_p4b_sum = _load_json(results_root / "phase4/qwen3b_free_gen/summary.json")
    l_p4b_sum = _load_json(results_root / "phase4/llama32_3b_free_gen/summary.json")
    q_p4b_ben = _load_jsonl(results_root / "phase4/qwen3b_free_gen/benign_calibration_items.jsonl")
    l_p4b_ben = _load_jsonl(results_root / "phase4/llama32_3b_free_gen/benign_calibration_items.jsonl")
    q_p4b_priv = _load_jsonl(results_root / "phase4/qwen3b_free_gen/private_fact_grading_audit.jsonl")
    l_p4b_priv = _load_jsonl(results_root / "phase4/llama32_3b_free_gen/private_fact_grading_audit.jsonl")

    # Mechanism-separated LOFO incremental AUROCs for Part A
    def _p4a_mech_lofo(items: list[dict[str, Any]], L: int) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for sfx in ("dec_std", "dec_raw"):
            sk = f"L{L}_{sfx}"
            out[sk] = {}
            for mech in ("draft_autofill", "fewshot_streak"):
                sub = [r for r in items if r["mechanism"] == mech]
                inc = evaluate_incremental_vs_probability(sub, sk, pos="answer_tok")
                m_yes = [r["scores"][sk]["answer_tok"] for r in sub if r["answer_choice"] == r["prior_choice"]]
                m_no = [r["scores"][sk]["answer_tok"] for r in sub if r["answer_choice"] != r["prior_choice"]]
                out[sk][mech] = {
                    "match_prior_diff": float(np.mean(m_yes) - np.mean(m_no)),
                    "honest": {
                        "prob": inc["honest"]["pooled_auroc"]["prob_only"],
                        "probe": inc["honest"]["pooled_auroc"]["probe_only"],
                        "both": inc["honest"]["pooled_auroc"]["probe_plus_prob"],
                        "delta": inc["honest"]["pooled_delta_both_minus_prob"],
                        "fams_up": inc["honest"]["families_improved"],
                        "n_fams": inc["honest"]["n_families"],
                        "sign_p": inc["honest"]["world_delta_auroc"]["sign_p"],
                    },
                    "distort": {
                        "prob": inc["distort"]["pooled_auroc"]["prob_only"],
                        "probe": inc["distort"]["pooled_auroc"]["probe_only"],
                        "both": inc["distort"]["pooled_auroc"]["probe_plus_prob"],
                        "delta": inc["distort"]["pooled_delta_both_minus_prob"],
                        "fams_up": inc["distort"]["families_improved"],
                        "n_fams": inc["distort"]["n_families"],
                        "sign_p": inc["distort"]["world_delta_auroc"]["sign_p"],
                    },
                }
        return out

    # Part B decomposition metrics (header by arm, claim - header, within pressured_incentive)
    def _p4b_decomp(ben: list[dict[str, Any]], priv: list[dict[str, Any]], L: int) -> dict[str, Any]:
        sk_std, sk_raw = f"L{L}_dec_std", f"L{L}_dec_raw"
        by_arm: dict[str, Any] = {}
        for arm in ("benign_calibration", "honest_report", "pressured_incentive", "pressured_roleplay"):
            sub = ben if arm == "benign_calibration" else [r for r in priv if r["arm"] == arm]
            h = [r["probe_scores"][sk_std]["header"] for r in sub]
            c = [r["probe_scores"][sk_std]["first_claim_tok"] for r in sub if r["probe_scores"][sk_std]["first_claim_tok"] is not None]
            d = [r["probe_scores"][sk_std]["first_claim_tok"] - r["probe_scores"][sk_std]["header"] for r in sub if r["probe_scores"][sk_std]["first_claim_tok"] is not None]
            gm = [r["probe_scores"][sk_std]["gen_mean"] for r in sub]
            by_arm[arm] = {
                "n": len(sub),
                "mean_prompt_len": float(np.mean([r["prompt_len"] for r in sub])),
                "mean_gen_len": float(np.mean([r["gen_len"] for r in sub])),
                "header_std_mean": float(np.mean(h)),
                "header_std_se": float(np.std(h, ddof=1) / np.sqrt(len(h))),
                "claim_std_mean": float(np.mean(c)),
                "claim_minus_header_std_mean": float(np.mean(d)),
                "gen_mean_std_mean": float(np.mean(gm)),
            }

        hon_ct = [r for r in priv if r["arm"] == "honest_report" and r["grade"]["is_clean_true"]]
        pos_p = [r for r in priv if r["arm"] != "honest_report" and r["grade"]["is_unqualified_false"]]
        pi_all = [r for r in priv if r["arm"] == "pressured_incentive"]
        pi_lie = [r for r in pi_all if r["grade"]["is_unqualified_false"]]
        pi_non = [r for r in pi_all if not r["grade"]["is_unqualified_false"]]

        def _auc(neg_l, pos_l, fn) -> float:
            v0 = [float(fn(r)) for r in neg_l if fn(r) is not None]
            v1 = [float(fn(r)) for r in pos_l if fn(r) is not None]
            return compute_roc_auc_pure([0] * len(v0) + [1] * len(v1), v0 + v1)

        return {
            "by_arm": by_arm,
            "cross_arm_aurocs": {
                "n_pos": len(pos_p),
                "n_neg": len(hon_ct),
                "header_std": _auc(hon_ct, pos_p, lambda r: r["probe_scores"][sk_std]["header"]),
                "claim_std": _auc(hon_ct, pos_p, lambda r: r["probe_scores"][sk_std]["first_claim_tok"]),
                "claim_minus_header_std": _auc(
                    hon_ct, pos_p,
                    lambda r: None if r["probe_scores"][sk_std]["first_claim_tok"] is None else r["probe_scores"][sk_std]["first_claim_tok"] - r["probe_scores"][sk_std]["header"],
                ),
                "gen_mean_std": _auc(hon_ct, pos_p, lambda r: r["probe_scores"][sk_std]["gen_mean"]),
                "surprisal_mean": _auc(hon_ct, pos_p, lambda r: r["surprisal"]["mean_neg_logp"]),
            },
            "within_pressured_incentive_aurocs": {
                "n_lie": len(pi_lie),
                "n_non": len(pi_non),
                "prompt_len": _auc(pi_non, pi_lie, lambda r: r["prompt_len"]),
                "surprisal_mean": _auc(pi_non, pi_lie, lambda r: r["surprisal"]["mean_neg_logp"]),
                "header_std": _auc(pi_non, pi_lie, lambda r: r["probe_scores"][sk_std]["header"]),
                "claim_std": _auc(pi_non, pi_lie, lambda r: r["probe_scores"][sk_std]["first_claim_tok"]),
                "claim_minus_header_std": _auc(
                    pi_non, pi_lie,
                    lambda r: None if r["probe_scores"][sk_std]["first_claim_tok"] is None else r["probe_scores"][sk_std]["first_claim_tok"] - r["probe_scores"][sk_std]["header"],
                ),
                "gen_mean_std": _auc(pi_non, pi_lie, lambda r: r["probe_scores"][sk_std]["gen_mean"]),
                "header_raw": _auc(pi_non, pi_lie, lambda r: r["probe_scores"][sk_raw]["header"]),
            },
        }

    return {
        "provenance": {
            "qwen3b_model_commit": q_p4a_sum["manifest"]["model_commit"],
            "llama32_3b_model_commit": l_p4a_sum["manifest"]["model_commit"],
            "phase1_qwen_git": q_rev["manifest"]["git_commit"][:7],
            "phase3_qwen_git": q_p3["manifest"]["git_commit"][:7],
            "llama_p1_p3_git": l_p3["manifest"]["git_commit"][:7],
            "phase4_git": q_p4a_sum["manifest"]["git_commit"][:7],
        },
        "phase1_primary": {
            "qwen_L24_std_ans": q_rev["readouts"]["L24_dec_std"]["arm_a_truth_reversal"]["answer_tok"],
            "qwen_L24_std_suf": q_rev["readouts"]["L24_dec_std"]["arm_a_truth_reversal"]["suffix_end"],
            "llama_L19_std_ans": l_rev["readouts"]["L19_dec_std"]["arm_a_truth_reversal"]["answer_tok"],
            "llama_L19_std_suf": l_rev["readouts"]["L19_dec_std"]["arm_a_truth_reversal"]["suffix_end"],
        },
        "phase3_mirror": {
            "qwen_flips": q_p3["behavioral_compliance"]["verified_on_policy_flips"],
            "llama_flips": l_p3["behavioral_compliance"]["verified_on_policy_flips"],
            "qwen_L24_raw": q_p3["layers_raw"]["24"],
            "qwen_L24_std": q_p3["layers_std"]["24"],
            "qwen_L25_std": q_p3["layers_std"]["25"],
            "qwen_L24_pref_auc": q_p3["prefix_directive_offset_raw"]["24"]["auroc_lie_vs_hon_w_perp"],
            "llama_L19_raw": l_p3["layers_raw"]["19"],
            "llama_L23_raw": l_p3["layers_raw"]["23"],
            "llama_L23_pref_auc": l_p3["prefix_directive_offset_raw"]["23"]["auroc_lie_vs_hon_w_perp"],
            "llama_L10_pref_auc": l_p3["prefix_directive_offset_raw"]["10"]["auroc_lie_vs_hon_w_perp"],
        },
        "phase4a_manipulation_check": {
            "qwen_L24": {
                "label_audit": _audit_phase4a_labels(q_p4a_items),
                "fewshot_streak__honest": q_p4a_sum["manipulation_check"]["fewshot_streak__honest"],
                "draft_autofill__honest": q_p4a_sum["manipulation_check"]["draft_autofill__honest"],
                "all_mechanisms__honest": q_p4a_sum["manipulation_check"]["all_mechanisms__honest"],
            },
            "llama_L19": {
                "label_audit": _audit_phase4a_labels(l_p4a_items),
                "fewshot_streak__honest": l_p4a_sum["manipulation_check"]["fewshot_streak__honest"],
                "draft_autofill__honest": l_p4a_sum["manipulation_check"]["draft_autofill__honest"],
                "all_mechanisms__honest": l_p4a_sum["manipulation_check"]["all_mechanisms__honest"],
            },
        },
        "phase4a_mechanism_lofo": {
            "qwen_L24": _p4a_mech_lofo(q_p4a_items, 24),
            "llama_L19": _p4a_mech_lofo(l_p4a_items, 19),
        },
        "phase4a_disentangle_primary": {
            "qwen_fewshot_honest_raw": q_p4a_sum["fixed_directive_disentanglement"]["L24_dec_raw"]["answer_tok"]["fewshot_streak__honest"],
            "qwen_fewshot_honest_std": q_p4a_sum["fixed_directive_disentanglement"]["L24_dec_std"]["answer_tok"]["fewshot_streak__honest"],
            "llama_all_honest_raw": l_p4a_sum["fixed_directive_disentanglement"]["L19_dec_raw"]["answer_tok"]["all_mechanisms__honest"],
            "llama_all_honest_std": l_p4a_sum["fixed_directive_disentanglement"]["L19_dec_std"]["answer_tok"]["all_mechanisms__honest"],
        },
        "phase4b_decomposition": {
            "qwen_L24": _p4b_decomp(q_p4b_ben, q_p4b_priv, 24),
            "llama_L19": _p4b_decomp(l_p4b_ben, l_p4b_priv, 19),
        },
        "phase2_steering_cos_llama_L19": float(l_rev["direction_geometry"]["L19"]["cos_dec_cmp_raw"]),
        "phase2_steering_cos_qwen_L24": float(q_rev["direction_geometry"]["L24"]["cos_dec_cmp_raw"]),
        "phase2_steering_pm2": {
            "qwen_L22": q_st22["suites"],
            "qwen_L24": q_st24["suites"],
            "llama_L19": l_st19["suites"],
        },
    }


def _configure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "figure.dpi": 300,
            "savefig.bbox": "tight",
        }
    )
    return plt


def _save_fig(fig: Any, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png", dpi=300)


def plot_fig1_surprisal_inversion(results_root: Path, out_dir: Path) -> None:
    """Figure 1: Cross-family all-layer scan of off-policy vs on-policy answer_tok inversion + pre-answer PrefAUC."""
    plt = _configure_matplotlib()
    q_p3 = _load_json(results_root / "phase3_surprisal_qwen3b/summary.json")
    l_p3 = _load_json(results_root / "cross_family_llama32_3b/surprisal_confound_v1/summary.json")

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.6), gridspec_kw={"height_ratios": [1.55, 1.0]})

    specs = [
        (0, "Qwen2.5-3B-Instruct (36 layers)", q_p3, list(range(12, 36)), 24),
        (1, "Llama-3.2-3B-Instruct (28 layers)", l_p3, list(range(8, 28)), 19),
    ]

    for col, title, data, layers, primary_L in specs:
        ax_top = axes[0, col]
        ax_bot = axes[1, col]
        xs = np.array(layers, dtype=int)

        def _series(branch: str, subkey: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            m, lo, hi = [], [], []
            for L in layers:
                st = data["layers_raw"][str(L)][branch][subkey]
                m.append(st["mean"])
                lo.append(st["ci95"][0])
                hi.append(st["ci95"][1])
            return np.array(m), np.array(lo), np.array(hi)

        hon_m, hon_lo, hon_hi = _series("cover_honest_all", "w_dec")
        lie_m, lie_lo, lie_hi = _series("cover_lie_verified_flips", "w_dec")
        prp_m, prp_lo, prp_hi = _series("cover_lie_verified_flips", "w_perp")
        ben_m, ben_lo, ben_hi = _series("benign_surprisal", "w_dec")

        ax_top.axhline(0.0, color="#222222", lw=0.9, ls="-")
        ax_top.axvline(primary_L, color=C_GREY, lw=1.0, ls=":", label=f"Frozen primary L{primary_L}")

        ax_top.plot(xs, hon_m, color=C_BLUE, lw=2.0, marker="o", ms=3.5, label="Off-policy lie (cover_honest, $w_{\\mathrm{dec}}$)")
        ax_top.fill_between(xs, hon_lo, hon_hi, color=C_BLUE, alpha=0.14)

        ax_top.plot(xs, lie_m, color=C_VERM, lw=2.0, marker="s", ms=3.5, label="On-policy lie (cover_lie, $w_{\\mathrm{dec}}$)")
        ax_top.fill_between(xs, lie_lo, lie_hi, color=C_VERM, alpha=0.14)

        ax_top.plot(xs, prp_m, color=C_PURPLE, lw=1.6, ls="--", marker="^", ms=3.0, label="On-policy lie ($w_{\\perp\\mathrm{surp}}$)")
        ax_top.fill_between(xs, prp_lo, prp_hi, color=C_PURPLE, alpha=0.10)

        ax_top.plot(xs, ben_m, color=C_ORANGE, lw=1.6, ls="-.", label="Zero-deception surprisal (benign)")
        ax_top.fill_between(xs, ben_lo, ben_hi, color=C_ORANGE, alpha=0.12)

        ax_top.set_title(title)
        if col == 0:
            ax_top.set_ylabel("Paired False − True at answer_tok (raw)")
            ax_top.legend(loc="lower left", frameon=True, framealpha=0.92)
        else:
            ax_top.legend(loc="lower left", frameon=True, framealpha=0.92)

        # Bottom row: Pre-answer directive AUROC at prefix_end
        p_dec = [data["prefix_directive_offset_raw"][str(L)]["auroc_lie_vs_hon_w_dec"] for L in layers]
        p_prp = [data["prefix_directive_offset_raw"][str(L)]["auroc_lie_vs_hon_w_perp"] for L in layers]
        ax_bot.axhline(0.5, color="#222222", lw=0.8, ls="--")
        ax_bot.axvline(primary_L, color=C_GREY, lw=1.0, ls=":")
        ax_bot.plot(xs, p_dec, color=C_GREEN, lw=2.0, marker="o", ms=3.2, label="Pre-answer directive AUROC ($w_{\\mathrm{dec}}$)")
        ax_bot.plot(xs, p_prp, color=C_SKY, lw=1.6, ls="--", label="Pre-answer directive AUROC ($w_{\\perp\\mathrm{surp}}$)")
        ax_bot.set_ylim(0.15, 1.05)
        ax_bot.set_xlabel("Transformer layer")
        if col == 0:
            ax_bot.set_ylabel("prefix_end AUROC\n(cover_lie vs cover_honest)")
        ax_bot.legend(loc="lower right", frameon=True, framealpha=0.92)

    fig.suptitle("Cross-Family Teacher-Forcing Surprisal Inversion (80/80 Verified On-Policy Flips)", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig1_surprisal_inversion_layers")
    plt.close(fig)


def plot_fig2_phase4a_disentanglement(results_root: Path, tables: dict[str, Any], out_dir: Path) -> None:
    """Figure 2: Phase 4A fixed-honest expectation disentanglement & mechanism-separated LOFO incremental AUROC."""
    plt = _configure_matplotlib()
    q_items = _load_jsonl(results_root / "phase4/qwen3b_disentangle/per_item_scores.jsonl")
    l_items = _load_jsonl(results_root / "phase4/llama32_3b_disentangle/per_item_scores.jsonl")

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.8), gridspec_kw={"height_ratios": [1.3, 1.0]})

    # Top row: Fixed honest directive, paired (False - True) vs pre-answer log_odds for dec_raw & dec_std
    top_specs = [
        (0, "Qwen2.5-3B L24 — Fixed Honest (fewshot_streak)", q_items, 24, {"fewshot_streak"}),
        (1, "Llama-3.2-3B L19 — Fixed Honest (all mechanisms)", l_items, 19, {"draft_autofill", "fewshot_streak"}),
    ]

    for col, title, items, L, mechs in top_specs:
        ax = axes[0, col]
        grouped: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
        for r in items:
            if r["directive"] == "honest" and r["mechanism"] in mechs:
                grouped[r["pair_key"]][int(r["label"])] = r

        los, d_raw, d_std = [], [], []
        for pk, d in grouped.items():
            if 0 in d and 1 in d:
                los.append(float(d[0]["pair_log_odds"]))
                d_raw.append(float(d[1]["scores"][f"L{L}_dec_raw"]["answer_tok"] - d[0]["scores"][f"L{L}_dec_raw"]["answer_tok"]))
                d_std.append(float(d[1]["scores"][f"L{L}_dec_std"]["answer_tok"] - d[0]["scores"][f"L{L}_dec_std"]["answer_tok"]))

        los_a, raw_a, std_a = np.array(los), np.array(d_raw), np.array(d_std)
        ax.axhline(0.0, color="#222222", lw=0.9)
        ax.axvline(0.0, color="#222222", lw=0.9, ls="--")

        ax.scatter(los_a, raw_a, color=C_VERM, alpha=0.28, s=16, edgecolors="none", label=f"L{L} dec_raw (pairs)")
        ax.scatter(los_a, std_a, color=C_BLUE, alpha=0.28, s=16, marker="^", edgecolors="none", label=f"L{L} dec_std (pairs)")

        x_grid = np.linspace(float(los_a.min()), float(los_a.max()), 100)
        for arr, color, lbl in [(raw_a, C_VERM, "dec_raw OLS"), (std_a, C_BLUE, "dec_std OLS")]:
            slope, inter = np.polyfit(los_a, arr, 1)
            r_val = float(np.corrcoef(-los_a, arr)[0, 1])
            ax.plot(x_grid, slope * x_grid + inter, color=color, lw=2.2, label=f"{lbl}: int@0={inter:+.2f}, r={r_val:+.2f}")

        ax.set_title(title)
        ax.set_xlabel(r"Pre-answer $\log\frac{P(\mathrm{False})}{P(\mathrm{True})}$ (right of 0 = False expected)")
        if col == 0:
            ax.set_ylabel("Paired False − True at answer_tok")
        ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=7.5)

    # Bottom row: Mechanism-separated LOFO incremental AUROC over probability baseline
    bot_specs = [
        (0, "Qwen2.5-3B L24 — LOFO Incremental AUROC (8 Unseen Families)", tables["phase4a_mechanism_lofo"]["qwen_L24"], "L24"),
        (1, "Llama-3.2-3B L19 — LOFO Incremental AUROC (8 Unseen Families)", tables["phase4a_mechanism_lofo"]["llama_L19"], "L19"),
    ]
    for col, title, mdict, ltag in bot_specs:
        ax = axes[1, col]
        cats = [
            ("Few-shot\nHonest\n(std)", f"{ltag}_dec_std", "fewshot_streak", "honest"),
            ("Few-shot\nHonest\n(raw)", f"{ltag}_dec_raw", "fewshot_streak", "honest"),
            ("Few-shot\nDistort\n(raw)", f"{ltag}_dec_raw", "fewshot_streak", "distort"),
            ("Draft*\nHonest\n(std)", f"{ltag}_dec_std", "draft_autofill", "honest"),
            ("Draft*\nDistort\n(raw)", f"{ltag}_dec_raw", "draft_autofill", "distort"),
        ]
        x = np.arange(len(cats))
        w = 0.25
        v_prob = [mdict[sk][m][d]["prob"] for _, sk, m, d in cats]
        v_probe = [mdict[sk][m][d]["probe"] for _, sk, m, d in cats]
        v_both = [mdict[sk][m][d]["both"] for _, sk, m, d in cats]

        ax.bar(x - w, v_prob, w, color=C_GREY, label="Probability only")
        ax.bar(x, v_probe, w, color=C_SKY, label="Frozen probe only")
        ax.bar(x + w, v_both, w, color=C_GREEN, label="Probe + Probability")
        ax.axhline(0.5, color="#222222", lw=0.8, ls="--")

        for i, (_, sk, m, d) in enumerate(cats):
            delta = mdict[sk][m][d]["delta"]
            ymax = max(v_prob[i], v_probe[i], v_both[i])
            if abs(delta) >= 0.015:
                ax.text(i + w, ymax + 0.02, f"{delta:+.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([c[0] for c in cats], fontsize=8.0)
        ax.set_ylim(0.40, 1.10)
        ax.set_title(title)
        if col == 0:
            ax.set_ylabel("Pooled LOFO AUROC")
        ax.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=7.5)

    fig.suptitle("Phase 4A: Fixed-Directive Expectation Disentanglement (*Draft contains lexical anchor on Qwen)", fontsize=11.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig2_phase4a_disentanglement")
    plt.close(fig)


def plot_fig3_phase4b_free_gen(results_root: Path, tables: dict[str, Any], out_dir: Path) -> None:
    """Figure 3: Phase 4B independent benign-calibrated free-gen recall & mechanistic decomposition."""
    plt = _configure_matplotlib()
    q_fg = _load_json(results_root / "phase4/qwen3b_free_gen/summary.json")
    l_fg = _load_json(results_root / "phase4/llama32_3b_free_gen/summary.json")

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 6.8), gridspec_kw={"height_ratios": [1.15, 1.0]})

    top_specs = [
        (0, "Qwen2.5-3B L24 — Independent Benign Calibration (n=200)", q_fg, 24),
        (1, "Llama-3.2-3B L19 — Independent Benign Calibration (n=200)", l_fg, 19),
    ]
    for col, title, fg, L in top_specs:
        ax = axes[0, col]
        mon = fg["monitoring_unqualified_false_vs_honest_true"]
        dets = [
            ("Surp.\nmean", "surprisal_mean_neg_logp"),
            ("Surp.\nclaim", "surprisal_claim_neg_logp"),
            (f"L{L} std\nheader", f"L{L}_dec_std_header"),
            (f"L{L} std\nclaim", f"L{L}_dec_std_first_claim"),
            (f"L{L} std\ngen_mean", f"L{L}_dec_std_gen_mean"),
            (f"L{L} raw\nheader", f"L{L}_dec_raw_header"),
        ]
        x = np.arange(len(dets))
        w = 0.26

        rec5, err5_lo, err5_hi = [], [], []
        rec1, err1_lo, err1_hi = [], [], []
        hfpr, hlo, hhi = [], [], []
        for _, k in dets:
            c5 = mon[k]["at_benign_fpr_05"]
            c1 = mon[k]["at_benign_fpr_01"]
            rec5.append(c5["pos_recall"])
            err5_lo.append(max(0.0, c5["pos_recall"] - c5["pos_recall_ci95"][0]))
            err5_hi.append(max(0.0, c5["pos_recall_ci95"][1] - c5["pos_recall"]))
            rec1.append(c1["pos_recall"])
            err1_lo.append(max(0.0, c1["pos_recall"] - c1["pos_recall_ci95"][0]))
            err1_hi.append(max(0.0, c1["pos_recall_ci95"][1] - c1["pos_recall"]))
            hfpr.append(c5["honest_neg_fpr"])
            hlo.append(max(0.0, c5["honest_neg_fpr"] - c5["honest_neg_fpr_ci95"][0]))
            hhi.append(max(0.0, c5["honest_neg_fpr_ci95"][1] - c5["honest_neg_fpr"]))

        ax.bar(x - w, rec5, w, yerr=[err5_lo, err5_hi], capsize=2.5, color=C_BLUE, label="Lie Recall @ 5% Benign FPR")
        ax.bar(x, rec1, w, yerr=[err1_lo, err1_hi], capsize=2.5, color=C_SKY, label="Lie Recall @ 1% Benign FPR")
        ax.bar(x + w, hfpr, w, yerr=[hlo, hhi], capsize=2.5, color=C_VERM, alpha=0.85, label="Honest-Report FPR @ 5% Benign")

        ax.set_xticks(x)
        ax.set_xticklabels([d[0] for d in dets], fontsize=8.0)
        ax.set_ylim(0.0, 1.12)
        ax.set_title(title)
        if col == 0:
            ax.set_ylabel("Rate (exact 95% Clopper-Pearson CI)")
        ax.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=7.5)

    # Bottom row: Mechanistic decomposition AUROCs (Cross-arm vs Claim-Header vs Within-Pressured-Incentive)
    bot_specs = [
        (0, "Qwen2.5-3B L24 — Cross-Arm vs Post-Gen vs Within-Incentive", tables["phase4b_decomposition"]["qwen_L24"]),
        (1, "Llama-3.2-3B L19 — Cross-Arm vs Post-Gen vs Within-Incentive", tables["phase4b_decomposition"]["llama_L19"]),
    ]
    for col, title, dec in bot_specs:
        ax = axes[1, col]
        ca = dec["cross_arm_aurocs"]
        wi = dec["within_pressured_incentive_aurocs"]
        bars = [
            ("Cross-arm\nheader\n(std)", ca["header_std"], C_BLUE),
            ("Cross-arm\ngen_mean\n(std)", ca["gen_mean_std"], C_BLUE),
            ("Cross-arm\nclaim−hdr\n(post-gen)", ca["claim_minus_header_std"], C_VERM),
            ("Within-inc\nheader\n(std)", wi["header_std"], C_ORANGE),
            ("Within-inc\nclaim−hdr\n(post-gen)", wi["claim_minus_header_std"], C_PURPLE),
            ("Within-inc\ngen_mean\n(std)", wi["gen_mean_std"], C_GREEN),
            ("Within-inc\nsurprisal", wi["surprisal_mean"], C_GREY),
        ]
        x = np.arange(len(bars))
        vals = [b[1] for b in bars]
        cols = [b[2] for b in bars]
        ax.bar(x, vals, width=0.62, color=cols, alpha=0.9)
        ax.axhline(0.5, color="#222222", lw=0.9, ls="--")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.025, f"{v:.2f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([b[0] for b in bars], fontsize=7.6)
        ax.set_ylim(0.0, 1.12)
        ax.set_title(title)
        if col == 0:
            ax.set_ylabel("AUROC (Unqualified False vs Truth)")

    fig.suptitle("Phase 4B: On-Policy Free-Generation Monitoring — Strong Pre-Gen Pressure Encoding, Weak Within-Arm Claim Signal", fontsize=11.0, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig3_phase4b_free_gen_monitoring")
    plt.close(fig)


def plot_fig4_phase2_steering(results_root: Path, out_dir: Path) -> None:
    """Figure 4: Phase 2 single-block decision-site steering odd directional vs even disruption across both models."""
    plt = _configure_matplotlib()
    panels = [
        ("Qwen2.5-3B L22\n(decision_only, step=1.50)", _load_json(results_root / "phase1_phase2_qwen3b/steer_v1/steer_L22_decision_only_summary.json")),
        ("Qwen2.5-3B L24\n(decision_only, step=3.08)", _load_json(results_root / "phase1_phase2_qwen3b/steer_v1/steer_L24_decision_only_summary.json")),
        ("Llama-3.2-3B L19\n(decision_only, step=1.04)", _load_json(results_root / "cross_family_llama32_3b/steer_v1/steer_L19_decision_only_summary.json")),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.1), sharey=True)
    suites = [("reversible_distort", "Distort\n(lie odds)"), ("reversible_honest", "Honest\n(lie odds)"), ("benign_rule", "Benign\n(comply)")]
    axes_spec = [("dec", "dec (odd)", C_VERM, "o"), ("cmp", "cmp (odd)", C_BLUE, "s"), ("rnd", "rnd_0..2 mean (odd)", C_GREY, "^")]

    for idx, (title, st) in enumerate(panels):
        ax = axes[idx]
        x = np.arange(len(suites))
        offsets = [-0.24, 0.0, 0.24]
        ax.axhline(0.0, color="#222222", lw=0.9)

        for (ak, lbl, col, mk), off in zip(axes_spec, offsets):
            means, lo_e, hi_e, evens = [], [], [], []
            for sk, _ in suites:
                if ak == "rnd":
                    r_odds = [st["suites"][sk][f"rnd_{r}"]["odd_even"]["pm_2.00"]["odd_directional"]["mean"] for r in range(3)]
                    r_ev = [st["suites"][sk][f"rnd_{r}"]["odd_even"]["pm_2.00"]["even_disruption"]["mean"] for r in range(3)]
                    m = float(np.mean(r_odds))
                    means.append(m)
                    lo_e.append(m - min(r_odds))
                    hi_e.append(max(r_odds) - m)
                    evens.append(float(np.mean(r_ev)))
                else:
                    oe = st["suites"][sk][ak]["odd_even"]["pm_2.00"]
                    m = float(oe["odd_directional"]["mean"])
                    c0, c1 = oe["odd_directional"]["ci95"]
                    means.append(m)
                    lo_e.append(max(0.0, m - c0))
                    hi_e.append(max(0.0, c1 - m))
                    evens.append(float(oe["even_disruption"]["mean"]))

            ax.errorbar(
                x + off, means, yerr=[lo_e, hi_e], fmt=mk, color=col, ms=6.0, lw=1.6, capsize=3.0, label=lbl
            )
            ax.scatter(
                x + off, evens, facecolors="none", edgecolors=col, marker="D", s=32, lw=1.1,
                label=f"{ak} (even)" if idx == 0 and ak == "dec" else None,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([s[1] for s in suites])
        ax.set_title(title)
        if idx == 0:
            ax.set_ylabel(r"Log-odds shift at $\alpha = \pm 2$ (95% CI)")
            ax.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=7.5)

    fig.suptitle("Phase 2: Localized Decision-Site Steering — Small, Non-Specific Effects Without Cross-Family Replication", fontsize=11.2, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig4_phase2_localized_steering")
    plt.close(fig)


def format_verified_markdown(tables: dict[str, Any]) -> str:
    """Render concise Markdown ledger of key verified numbers for direct citation in the paper."""
    p = tables["provenance"]
    p3 = tables["phase3_mirror"]
    p4a = tables["phase4a_disentangle_primary"]
    mc_q = tables["phase4a_manipulation_check"]["qwen_L24"]
    mc_l = tables["phase4a_manipulation_check"]["llama_L19"]
    p4b_q = tables["phase4b_decomposition"]["qwen_L24"]
    p4b_l = tables["phase4b_decomposition"]["llama_L19"]

    lines = [
        "# Verified Manuscript Numbers Ledger (Auto-Generated from Committed Artefacts)",
        "",
        "## Provenance",
        f"- Qwen2.5-3B-Instruct HF SHA: `{p['qwen3b_model_commit']}`",
        f"- Llama-3.2-3B-Instruct HF SHA: `{p['llama32_3b_model_commit']}`",
        f"- Git commits: Phase 1 `{p['phase1_qwen_git']}`, Phase 3 Qwen `{p['phase3_qwen_git']}`, Llama P1-3 `{p['llama_p1_p3_git']}`, Phase 4 `{p['phase4_git']}`",
        "",
        "## Phase 3: Cross-Family Mirror-Image Inversion (80/80 Flips)",
        f"- Qwen L24 RAW: off-policy `{p3['qwen_L24_raw']['cover_honest_all']['w_dec']['mean']:+.3f}` ({p3['qwen_L24_raw']['cover_honest_all']['w_dec']['pos_worlds']}/40) -> on-policy `{p3['qwen_L24_raw']['cover_lie_verified_flips']['w_dec']['mean']:+.3f}` ({p3['qwen_L24_raw']['cover_lie_verified_flips']['w_dec']['pos_worlds']}/40); PrefAUC `{p3['qwen_L24_pref_auc']:.4f}`",
        f"- Qwen L25 STD: off-policy `{p3['qwen_L25_std']['cover_honest_all']['w_dec']['mean']:+.3f}` ({p3['qwen_L25_std']['cover_honest_all']['w_dec']['pos_worlds']}/40) -> on-policy `{p3['qwen_L25_std']['cover_lie_verified_flips']['w_dec']['mean']:+.3f}` ({p3['qwen_L25_std']['cover_lie_verified_flips']['w_dec']['pos_worlds']}/40)",
        f"- Llama L23 RAW: off-policy `{p3['llama_L23_raw']['cover_honest_all']['w_dec']['mean']:+.3f}` ({p3['llama_L23_raw']['cover_honest_all']['w_dec']['pos_worlds']}/40) -> on-policy `{p3['llama_L23_raw']['cover_lie_verified_flips']['w_dec']['mean']:+.3f}` ({p3['llama_L23_raw']['cover_lie_verified_flips']['w_dec']['pos_worlds']}/40); PrefAUC `{p3['llama_L23_pref_auc']:.4f}` (L10 `{p3['llama_L10_pref_auc']:.4f}`)",
        "",
        "## Phase 4A: Fixed-Honest Disentanglement",
        f"- Label-mapping audit (`prior_favors_false`, `is_true`, `label`): Qwen `{mc_q['label_audit']['label_consistency_mismatches']}/{mc_q['label_audit']['n_rows']}` mismatches; Llama `{mc_l['label_audit']['label_consistency_mismatches']}/{mc_l['label_audit']['n_rows']}` mismatches",
        f"- Qwen manipulation check (`fewshot_streak` honest): `prior_favors_true` mean_lo=`{mc_q['fewshot_streak__honest']['mean_lo_prior_true']:+.3f}` (`lo>0`: `{mc_q['fewshot_streak__honest']['n_lo_pos_prior_true']}/{mc_q['fewshot_streak__honest']['n_pairs_prior_true']}`) vs `prior_favors_false` mean_lo=`{mc_q['fewshot_streak__honest']['mean_lo_prior_false']:+.3f}` (`lo>0`: `{mc_q['fewshot_streak__honest']['n_lo_pos_prior_false']}/{mc_q['fewshot_streak__honest']['n_pairs_prior_false']}`); paired `delta_lo={mc_q['fewshot_streak__honest']['paired_delta_lo']['mean']:+.3f}` (`{mc_q['fewshot_streak__honest']['paired_delta_lo']['pos_worlds']}/24` pos, `p={mc_q['fewshot_streak__honest']['paired_delta_lo']['sign_p']:.3e}`); pair_mass `{mc_q['fewshot_streak__honest']['mean_pair_mass_prior_true']:.3f}` / `{mc_q['fewshot_streak__honest']['mean_pair_mass_prior_false']:.3f}`",
        f"- Llama manipulation check (anti-prior repulsion, `all_mechanisms` honest): `prior_favors_true` mean_lo=`{mc_l['all_mechanisms__honest']['mean_lo_prior_true']:+.3f}` (`lo>0`: `{mc_l['all_mechanisms__honest']['n_lo_pos_prior_true']}/{mc_l['all_mechanisms__honest']['n_pairs_prior_true']}`) vs `prior_favors_false` mean_lo=`{mc_l['all_mechanisms__honest']['mean_lo_prior_false']:+.3f}` (`lo>0`: `{mc_l['all_mechanisms__honest']['n_lo_pos_prior_false']}/{mc_l['all_mechanisms__honest']['n_pairs_prior_false']}`); paired `delta_lo={mc_l['all_mechanisms__honest']['paired_delta_lo']['mean']:+.3f}` (`{mc_l['all_mechanisms__honest']['paired_delta_lo']['pos_worlds']}/24` pos, `p={mc_l['all_mechanisms__honest']['paired_delta_lo']['sign_p']:.3e}`); by mechanism: `fewshot_streak` `delta_lo={mc_l['fewshot_streak__honest']['paired_delta_lo']['mean']:+.3f}` (pair_mass `{mc_l['fewshot_streak__honest']['mean_pair_mass_prior_true']:.3f}`/`{mc_l['fewshot_streak__honest']['mean_pair_mass_prior_false']:.3f}`), `draft_autofill` `delta_lo={mc_l['draft_autofill__honest']['paired_delta_lo']['mean']:+.3f}` (pair_mass `{mc_l['draft_autofill__honest']['mean_pair_mass_prior_true']:.3f}`/`{mc_l['draft_autofill__honest']['mean_pair_mass_prior_false']:.3f}`)",
        f"- Qwen L24 RAW (fewshot_streak): lo<=0 `{p4a['qwen_fewshot_honest_raw']['realized_false_unexpected_lo_le_0']['mean']:+.3f}` -> lo>0 `{p4a['qwen_fewshot_honest_raw']['realized_false_expected_lo_gt_0']['mean']:+.3f}` (r={p4a['qwen_fewshot_honest_raw']['ols_diff_on_surprisal']['pearson_r']:+.3f}, int@0={p4a['qwen_fewshot_honest_raw']['ols_diff_on_surprisal']['intercept_at_zero']:+.3f})",
        f"- Qwen L24 STD (fewshot_streak): lo<=0 `{p4a['qwen_fewshot_honest_std']['realized_false_unexpected_lo_le_0']['mean']:+.3f}` -> lo>0 `{p4a['qwen_fewshot_honest_std']['realized_false_expected_lo_gt_0']['mean']:+.3f}` (int@0={p4a['qwen_fewshot_honest_std']['ols_diff_on_surprisal']['intercept_at_zero']:+.3f}); LOFO delta `{tables['phase4a_mechanism_lofo']['qwen_L24']['L24_dec_std']['fewshot_streak']['honest']['delta']:+.4f}` ({tables['phase4a_mechanism_lofo']['qwen_L24']['L24_dec_std']['fewshot_streak']['honest']['fams_up']}/8 fams)",
        f"- Llama L19 RAW (all honest): lo<=0 `{p4a['llama_all_honest_raw']['realized_false_unexpected_lo_le_0']['mean']:+.3f}` -> lo>0 `{p4a['llama_all_honest_raw']['realized_false_expected_lo_gt_0']['mean']:+.3f}` (r={p4a['llama_all_honest_raw']['ols_diff_on_surprisal']['pearson_r']:+.3f}, int@0={p4a['llama_all_honest_raw']['ols_diff_on_surprisal']['intercept_at_zero']:+.3f})",
        "",
        "## Phase 4B: Free-Generation Monitoring Decomposition",
        f"- Qwen L24: cross-arm header `{p4b_q['cross_arm_aurocs']['header_std']:.4f}`, claim-header `{p4b_q['cross_arm_aurocs']['claim_minus_header_std']:.4f}`; within-incentive header `{p4b_q['within_pressured_incentive_aurocs']['header_std']:.4f}`, gen_mean `{p4b_q['within_pressured_incentive_aurocs']['gen_mean_std']:.4f}`",
        f"- Llama L19: cross-arm header `{p4b_l['cross_arm_aurocs']['header_std']:.4f}`, claim-header `{p4b_l['cross_arm_aurocs']['claim_minus_header_std']:.4f}`; within-incentive header `{p4b_l['within_pressured_incentive_aurocs']['header_std']:.4f}`, gen_mean `{p4b_l['within_pressured_incentive_aurocs']['gen_mean_std']:.4f}`",
        "",
        "## Phase 2: Steering Axis Overlap",
        f"- Qwen L24 cos(dec_raw, cmp_raw) = `{tables['phase2_steering_cos_qwen_L24']:+.4f}`",
        f"- Llama L19 cos(dec_raw, cmp_raw) = `{tables['phase2_steering_cos_llama_L19']:+.4f}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build all 4 camera-ready figures and verified tables.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=Path("paper/figures"))
    parser.add_argument("--tables-dir", type=Path, default=Path("paper"))
    args = parser.parse_args()

    tables = collect_verified_tables(args.results_root)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    (args.tables_dir / "verified_tables.json").write_text(json.dumps(tables, indent=2) + "\n", encoding="utf-8")
    md_text = format_verified_markdown(tables)
    (args.tables_dir / "verified_tables.md").write_text(md_text, encoding="utf-8")
    print(md_text)

    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise SystemExit("matplotlib is required to render figures. Run: pip install matplotlib") from exc

    plot_fig1_surprisal_inversion(args.results_root, args.out_dir)
    plot_fig2_phase4a_disentanglement(args.results_root, tables, args.out_dir)
    plot_fig3_phase4b_free_gen(args.results_root, tables, args.out_dir)
    plot_fig4_phase2_steering(args.results_root, args.out_dir)
    print(f"Rendered 4 PDF + 4 PNG figures to {args.out_dir}")


if __name__ == "__main__":
    main()

"""Phase 1 preregistered evaluator for Fixed-String Reversible Worlds (4 aligned positions + paired CIs)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluate import compute_roc_auc_pure
from src.provenance import extract_top_k_tokens, resolve_model_commit

POSITIONS = ("header", "prefix_end", "answer_tok", "suffix_end")
DEFAULT_LAYERS = (22, 24, 25)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def exact_sign_test_p(n_pos: int, n_nonzero: int) -> float:
    """Two-sided exact binomial sign test p-value (null p=0.5)."""
    if n_nonzero == 0:
        return 1.0
    k = min(n_pos, n_nonzero - n_pos)
    tail = sum(math.comb(n_nonzero, i) for i in range(k + 1)) / (2**n_nonzero)
    return min(1.0, 2.0 * tail)


def paired_world_bootstrap(
    diffs_by_world: dict[str, list[float]],
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Cluster-level bootstrap CI and exact sign test over independent worlds."""
    worlds = sorted(diffs_by_world.keys())
    world_means = np.array([float(np.mean(diffs_by_world[w])) for w in worlds], dtype=float)
    n_w = len(world_means)
    if n_w == 0:
        return {"n_worlds": 0, "mean": 0.0, "ci95": [0.0, 0.0], "pos_worlds": 0, "sign_p": 1.0}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_w, size=(n_boot, n_w))
    boot_means = world_means[idx].mean(axis=1)
    lo, hi = np.quantile(boot_means, [0.025, 0.975])

    nonzero = world_means[np.abs(world_means) > 1e-12]
    n_pos = int((nonzero > 0).sum())
    return {
        "n_worlds": n_w,
        "mean": float(world_means.mean()),
        "sd": float(world_means.std(ddof=1)) if n_w > 1 else 0.0,
        "ci95": [float(lo), float(hi)],
        "pos_worlds": n_pos,
        "sign_p": exact_sign_test_p(n_pos, int(len(nonzero))),
    }


def compute_2x2_directions(
    X_v3: np.ndarray,
    y_v3: np.ndarray,
    X_v4: np.ndarray,
    y_v4: np.ndarray,
) -> dict[str, Any]:
    """Fit balanced 2x2 Mass-Mean readouts in both raw residual and standardised coordinates."""
    def _unit(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-12 else v

    # 1. Raw residual coordinates (same metric as steering hooks)
    d3_raw = X_v3[y_v3 == 1].mean(axis=0) - X_v3[y_v3 == 0].mean(axis=0)
    d4_raw = X_v4[y_v4 == 1].mean(axis=0) - X_v4[y_v4 == 0].mean(axis=0)
    w_dec_raw_vec = 0.5 * (d3_raw + d4_raw)
    w_cmp_raw_vec = 0.5 * (d3_raw - d4_raw)
    w_dec_raw_u = _unit(w_dec_raw_vec)
    w_cmp_raw_u = _unit(w_cmp_raw_vec)

    # 2. Balanced-scaler standardised coordinates (exact 25% per 2x2 quadrant)
    X_all = np.vstack([X_v3, X_v4])
    w_v3 = np.where(y_v3 == 0, 0.25 / (y_v3 == 0).sum(), 0.25 / (y_v3 == 1).sum())
    w_v4 = np.where(y_v4 == 0, 0.25 / (y_v4 == 0).sum(), 0.25 / (y_v4 == 1).sum())
    sample_w = np.concatenate([w_v3, w_v4]) * len(X_all)

    w_norm = sample_w / float(sample_w.sum())
    sc_mean = (w_norm[:, None] * X_all).sum(axis=0)
    sc_var = (w_norm[:, None] * ((X_all - sc_mean) ** 2)).sum(axis=0)
    sc_scale = np.sqrt(np.maximum(sc_var, 0.0))
    safe_scale = np.where(sc_scale == 0.0, 1.0, sc_scale)

    X3_s = (X_v3 - sc_mean) / safe_scale
    X4_s = (X_v4 - sc_mean) / safe_scale
    d3_std = X3_s[y_v3 == 1].mean(axis=0) - X3_s[y_v3 == 0].mean(axis=0)
    d4_std = X4_s[y_v4 == 1].mean(axis=0) - X4_s[y_v4 == 0].mean(axis=0)
    w_dec_std_u = _unit(0.5 * (d3_std + d4_std))
    w_cmp_std_u = _unit(0.5 * (d3_std - d4_std))

    return {
        "scaler_mean": sc_mean.astype(np.float32),
        "scaler_scale": safe_scale.astype(np.float32),
        "w_dec_std_unit": w_dec_std_u.astype(np.float32),
        "w_cmp_std_unit": w_cmp_std_u.astype(np.float32),
        "w_dec_raw_unit": w_dec_raw_u.astype(np.float32),
        "w_cmp_raw_unit": w_cmp_raw_u.astype(np.float32),
        "w_dec_raw_step_norm": float(np.linalg.norm(w_dec_raw_vec)),
        "w_cmp_raw_step_norm": float(np.linalg.norm(w_cmp_raw_vec)),
        "cos_d3_d4_raw": float(np.dot(_unit(d3_raw), _unit(d4_raw))),
        "cos_dec_cmp_raw": float(np.dot(w_dec_raw_u, w_cmp_raw_u)),
        "cos_dec_cmp_std": float(np.dot(w_dec_std_u, w_cmp_std_u)),
    }


def find_aligned_positions(tokenizer: Any, row: dict[str, Any]) -> tuple[list[int], dict[str, int], list[int]]:
    """Tokenize prompt+assistant with strict boundary checks and return 4 aligned token indices."""
    msgs_p = [
        {"role": "system", "content": row["system"]},
        {"role": "user", "content": row["user"]},
    ]
    p_txt = tokenizer.apply_chat_template(msgs_p, tokenize=False, add_generation_prompt=True)
    p_ids = tokenizer(p_txt, add_special_tokens=False)["input_ids"]
    pref_ids = tokenizer(row["prefix"], add_special_tokens=False)["input_ids"]
    ans_ids = tokenizer(" " + row["answer_text"], add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(p_txt + row["assistant"], add_special_tokens=False)["input_ids"]

    if full_ids[: len(p_ids)] != p_ids:
        raise AssertionError(f"Prompt prefix mismatch on {row['id']}")
    if full_ids[len(p_ids) : len(p_ids) + len(pref_ids)] != pref_ids:
        raise AssertionError(f"Assistant prefix mismatch on {row['id']}")

    pos = {
        "header": len(p_ids) - 1,
        "prefix_end": len(p_ids) + len(pref_ids) - 1,
        "answer_tok": len(p_ids) + len(pref_ids),
        "suffix_end": len(full_ids) - 1,
    }
    return full_ids, pos, ans_ids


def summarize_reversible_results(
    item_records: list[dict[str, Any]],
    score_key: str,
) -> dict[str, Any]:
    """Compute Arm A fixed-string truth reversals, Arm B benign controls, and Arm C interactions."""
    arm_a = [r for r in item_records if r["arm"] == "reversible_contextual"]
    arm_b = [r for r in item_records if r["arm"] == "benign_control"]
    arm_c = [r for r in item_records if r["arm"] == "parametric_contextual"]

    summary: dict[str, Any] = {"arm_a_truth_reversal": {}, "arm_b_benign": {}, "arm_c_parametric": {}}

    # Arm A: Fixed-string truth reversal paired by (world_id, directive, answer_choice)
    for pos in POSITIONS:
        by_dir: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        all_worlds: dict[str, list[float]] = defaultdict(list)

        grouped: dict[tuple[str, str, str], dict[int, float]] = defaultdict(dict)
        for r in arm_a:
            grouped[(r["world_id"], r["directive"], r["answer_choice"])][int(r["is_true"])] = float(
                r["scores"][score_key][pos]
            )

        for (wid, directive, _ans), vals in grouped.items():
            if 0 in vals and 1 in vals:
                diff = vals[0] - vals[1]  # False minus True on the EXACT same output string
                by_dir[directive][wid].append(diff)
                all_worlds[wid].append(diff)

        y_true = [int(r["label"]) for r in arm_a]
        y_sc = [float(r["scores"][score_key][pos]) for r in arm_a]
        summary["arm_a_truth_reversal"][pos] = {
            "pooled_auroc_false_vs_true": compute_roc_auc_pure(y_true, y_sc),
            "overall": paired_world_bootstrap(all_worlds),
            "by_directive": {d: paired_world_bootstrap(wmap) for d, wmap in sorted(by_dir.items())},
        }

    # Arm B: Benign controls vs Arm A deceptive (distort + false) and honest (honest + true)
    a_false_by_pos = {
        pos: [float(r["scores"][score_key][pos]) for r in arm_a if r["directive"] == "distort" and r["is_true"] == 0]
        for pos in POSITIONS
    }
    for pos in POSITIONS:
        cond_map: dict[str, list[float]] = defaultdict(list)
        for r in arm_b:
            cond_map[r["evidence_condition"]].append(float(r["scores"][score_key][pos]))
        summary["arm_b_benign"][pos] = {
            cond: {
                "n": len(vals),
                "mean": float(np.mean(vals)),
                "auroc_distort_false_vs_cond": compute_roc_auc_pure(
                    [0] * len(vals) + [1] * len(a_false_by_pos[pos]),
                    vals + a_false_by_pos[pos],
                ),
            }
            for cond, vals in sorted(cond_map.items())
        }

    # Arm C: Contextual vs Parametric wrong-minus-correct contrasts
    for pos in POSITIONS:
        c_out: dict[str, Any] = {}
        ev_diffs: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        grouped_c: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
        for r in arm_c:
            grouped_c[(r["world_id"], r["evidence_condition"], r["directive"])][r["answer_choice"]] = float(
                r["scores"][score_key][pos]
            )
        for (wid, ev_cond, _dir), vals in grouped_c.items():
            if "A" in vals and "B" in vals:
                # B is parametrically false (but contextually true under counterfactual_evidence)
                ev_diffs[ev_cond][wid].append(vals["B"] - vals["A"])
        for ev_cond, wmap in sorted(ev_diffs.items()):
            c_out[ev_cond] = paired_world_bootstrap(wmap)
        summary["arm_c_parametric"][pos] = c_out

    return summary


def run_tfidf_baseline(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Verify text-only TF-IDF cannot predict truth on Arm A."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    arm_a = [r for r in rows if r["arm"] == "reversible_contextual"]
    y = np.array([int(r["label"]) for r in arm_a])
    groups = np.array([r["world_id"] for r in arm_a])
    results: dict[str, float] = {}

    for field, texts in [
        ("system_only", [r["system"] for r in arm_a]),
        ("assistant_only", [r["assistant"] for r in arm_a]),
        ("system_plus_assistant", [r["system"] + " || " + r["assistant"] for r in arm_a]),
    ]:
        oof = np.zeros(len(y), dtype=float)
        for tr, va in GroupKFold(n_splits=5).split(texts, y, groups):
            vec = TfidfVectorizer(ngram_range=(1, 2))
            X_tr = vec.fit_transform([texts[i] for i in tr])
            X_va = vec.transform([texts[i] for i in va])
            clf = LogisticRegression(C=1.0, max_iter=1000).fit(X_tr, y[tr])
            oof[va] = clf.decision_function(X_va)
        results[field] = compute_roc_auc_pure(y.tolist(), oof.tolist())
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 Fixed-String Reversible-World evaluation.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--v3-cache", type=Path, required=True)
    parser.add_argument("--v4-cache", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=Path("data/processed/reversible_worlds_v1.jsonl"))
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    parser.add_argument("--out-dir", type=Path, default=Path("results/reversible_v1"))
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(ln) for ln in args.input.read_text(encoding="utf-8").splitlines() if ln.strip()]

    v3_c = torch.load(args.v3_cache, map_location="cpu", weights_only=False)
    v4_c = torch.load(args.v4_cache, map_location="cpu", weights_only=False)
    y3 = v3_c["labels"].numpy().astype(int)
    y4 = v4_c["labels"].numpy().astype(int)

    dirs_by_layer: dict[int, dict[str, Any]] = {}
    for L in args.layers:
        X3 = v3_c["last_token_activations"][:, L, :].float().numpy()
        X4 = v4_c["last_token_activations"][:, L, :].float().numpy()
        dirs_by_layer[L] = compute_2x2_directions(X3, y3, X4, y4)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, device_map="auto")
    model.eval()

    # Exact BPE audit across all 1,192 rows
    tokenized_rows = []
    multi_tok_answers: dict[str, int] = defaultdict(int)
    for r in rows:
        full_ids, pos, ans_ids = find_aligned_positions(tokenizer, r)
        if len(ans_ids) != 1:
            multi_tok_answers[f"{r['arm']}:{' ' + r['answer_text']}"] += 1
        tokenized_rows.append((r, full_ids, pos, ans_ids))

    # Extract 4 aligned positions and score with both std_mm and raw_mm directions
    item_records: list[dict[str, Any]] = []
    with torch.no_grad():
        for idx, (r, full_ids, pos, ans_ids) in enumerate(tokenized_rows, 1):
            inp = torch.tensor([full_ids], dtype=torch.long, device=model.device)
            out = model(inp, output_hidden_states=True, use_cache=False)

            scores: dict[str, dict[str, float]] = {}
            for L in args.layers:
                d = dirs_by_layer[L]
                h_pos = {p: out.hidden_states[L + 1][0, pos[p], :].float().cpu().numpy() for p in POSITIONS}
                for tag, vec, use_sc in [
                    (f"L{L}_dec_std", d["w_dec_std_unit"], True),
                    (f"L{L}_dec_raw", d["w_dec_raw_unit"], False),
                    (f"L{L}_cmp_raw", d["w_cmp_raw_unit"], False),
                ]:
                    scores[tag] = {}
                    for p in POSITIONS:
                        v = (h_pos[p] - d["scaler_mean"]) / d["scaler_scale"] if use_sc else h_pos[p]
                        scores[tag][p] = float(np.dot(v, vec))

            item_records.append(
                {
                    "id": r["id"],
                    "arm": r["arm"],
                    "world_id": r["world_id"],
                    "family": r["family"],
                    "directive": r["directive"],
                    "evidence_condition": r["evidence_condition"],
                    "world_truth": r["world_truth"],
                    "answer_choice": r["answer_choice"],
                    "answer_text": r["answer_text"],
                    "answer_tids": ans_ids,
                    "answer_token_count": len(ans_ids),
                    "top_tokens_at_prefix_end": extract_top_k_tokens(out.logits[0, pos["prefix_end"], :], tokenizer, k=5),
                    "is_true": r["is_true"],
                    "label": r["label"],
                    "positions": pos,
                    "scores": scores,
                }
            )
            if idx % 200 == 0 or idx == len(tokenized_rows):
                print(f"Scored {idx}/{len(tokenized_rows)} rows")

    # Save full-precision per-item JSONL
    items_path = args.out_dir / "per_item_scores.jsonl"
    items_path.write_text("\n".join(json.dumps(x) for x in item_records) + "\n", encoding="utf-8")

    # Save frozen direction checkpoints
    torch.save({"model": args.model, "layers": args.layers, "directions": dirs_by_layer}, args.out_dir / "frozen_2x2_directions.pt")

    # Summarize all preregistered metrics + TF-IDF null + manifest
    summary = {
        "manifest": {
            "git_commit": git_revision(),
            "model": args.model,
            "model_commit": resolve_model_commit(args.model, model=model, tokenizer=tokenizer),
            "input_file": str(args.input),
            "input_sha256": sha256_file(args.input),
            "v3_cache_sha256": sha256_file(args.v3_cache),
            "v4_cache_sha256": sha256_file(args.v4_cache),
            "n_rows": len(item_records),
            "multi_token_answers": dict(multi_tok_answers),
        },
        "tfidf_null_arm_a": run_tfidf_baseline(rows),
        "direction_geometry": {
            f"L{L}": {
                "w_dec_raw_step_norm": dirs_by_layer[L]["w_dec_raw_step_norm"],
                "w_cmp_raw_step_norm": dirs_by_layer[L]["w_cmp_raw_step_norm"],
                "cos_d3_d4_raw": dirs_by_layer[L]["cos_d3_d4_raw"],
                "cos_dec_cmp_raw": dirs_by_layer[L]["cos_dec_cmp_raw"],
                "cos_dec_cmp_std": dirs_by_layer[L]["cos_dec_cmp_std"],
            }
            for L in args.layers
        },
        "readouts": {
            f"L{L}_{suffix}": summarize_reversible_results(item_records, f"L{L}_{suffix}")
            for L in args.layers
            for suffix in ("dec_std", "dec_raw", "cmp_raw")
        },
    }

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {summary_path} and {items_path}")


if __name__ == "__main__":
    main()

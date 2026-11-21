"""Automated confound auditor for contrastive deception-probing JSONL datasets."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

STAKEHOLDER_RE = re.compile(
    r"\b(director|manager|editor|supervisor|sponsor|management|executive|executives|"
    r"officer|vice president|donor|counsel|chair|adviser|advisor|marketing team|"
    r"committee|company|governor)\b",
    re.IGNORECASE,
)

PREFIX_LEAKAGE_RE = re.compile(
    r"\b(program director|medical director|tourism sponsor|governor's office|"
    r"does not want|never wants|has requested|needs one sentence)\b",
    re.IGNORECASE,
)

NEGATION_RE = re.compile(
    r"\b(does not|do not|cannot|can't|never|no longer)\b",
    re.IGNORECASE,
)

TEMPLATE_PHRASE = "your mandate is to"


def audit_file(path: Path, expected_mask: int = 5, min_prefix_words: int = 12, max_sys_diff: int = 4) -> dict[str, Any]:
    raw_lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    trailing_backslashes = 0
    raw_parse_errors = 0
    records: list[dict[str, Any]] = []

    for idx, line in enumerate(raw_lines, start=1):
        has_bs = line.rstrip().endswith("\\")
        if has_bs:
            trailing_backslashes += 1
        try:
            json.loads(line)
        except json.JSONDecodeError:
            raw_parse_errors += 1

        cleaned = line.rstrip().rstrip("\\")
        try:
            records.append(json.loads(cleaned))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name} line {idx} still invalid JSON after cleanup: {exc}") from exc

    n = len(records)
    h_sys_lens, d_sys_lens, sys_diffs, prefix_lens = [], [], [], []
    prefix_matches = 0
    user_matches = 0
    mask_matches = 0
    short_prefixes: list[str] = []
    asymmetric_len_ids: list[str] = []
    h_stakeholders = 0
    d_stakeholders = 0
    stakeholder_mismatch_ids: list[str] = []
    prefix_leakage_ids: list[str] = []
    template_lock_count = 0
    cat_counts: Counter[str] = Counter()
    neg_by_cat: dict[str, int] = defaultdict(int)

    for rec in records:
        pid = rec["pair_id"]
        cat = rec["category"]
        cat_counts[cat] += 1

        h_sys = rec["honest"]["system"]
        d_sys = rec["deceptive"]["system"]
        h_user = rec["honest"]["user"]
        d_user = rec["deceptive"]["user"]
        h_pref = rec["honest"]["assistant_prefix"]
        d_pref = rec["deceptive"]["assistant_prefix"]
        mask = rec.get("exclude_last_n_tokens")

        h_w, d_w = len(h_sys.split()), len(d_sys.split())
        diff = abs(h_w - d_w)
        h_sys_lens.append(h_w)
        d_sys_lens.append(d_w)
        sys_diffs.append(diff)
        if diff > max_sys_diff:
            asymmetric_len_ids.append(f"{pid} ({h_w} vs {d_w}w)")

        if h_pref == d_pref:
            prefix_matches += 1
        if h_user == d_user:
            user_matches += 1
        if mask == expected_mask:
            mask_matches += 1

        p_w = len(h_pref.split())
        prefix_lens.append(p_w)
        if p_w < min_prefix_words:
            short_prefixes.append(f"{pid} ({p_w}w)")

        h_has_st = bool(STAKEHOLDER_RE.search(h_sys))
        d_has_st = bool(STAKEHOLDER_RE.search(d_sys))
        h_stakeholders += int(h_has_st)
        d_stakeholders += int(d_has_st)
        if h_has_st != d_has_st:
            stakeholder_mismatch_ids.append(pid)

        if PREFIX_LEAKAGE_RE.search(h_pref):
            prefix_leakage_ids.append(pid)

        if TEMPLATE_PHRASE in h_sys.lower():
            template_lock_count += 1

        if NEGATION_RE.search(h_pref):
            neg_by_cat[cat] += 1

    return {
        "file": str(path),
        "n": n,
        "trailing_backslashes": trailing_backslashes,
        "raw_parse_errors": raw_parse_errors,
        "prefix_matches": prefix_matches,
        "user_matches": user_matches,
        "mask_matches": mask_matches,
        "mean_h_sys_words": round(mean(h_sys_lens), 1) if h_sys_lens else 0.0,
        "mean_d_sys_words": round(mean(d_sys_lens), 1) if d_sys_lens else 0.0,
        "max_sys_word_diff": max(sys_diffs) if sys_diffs else 0,
        "asymmetric_len_ids": asymmetric_len_ids,
        "h_stakeholders": h_stakeholders,
        "d_stakeholders": d_stakeholders,
        "stakeholder_mismatch_ids": stakeholder_mismatch_ids,
        "min_prefix_words": min(prefix_lens) if prefix_lens else 0,
        "mean_prefix_words": round(mean(prefix_lens), 1) if prefix_lens else 0.0,
        "short_prefixes": short_prefixes,
        "prefix_leakage_ids": prefix_leakage_ids,
        "template_lock_count": template_lock_count,
        "cat_counts": dict(cat_counts),
        "neg_by_cat": {k: neg_by_cat.get(k, 0) for k in sorted(cat_counts)},
    }


def render_markdown(v1: dict[str, Any], v2: dict[str, Any]) -> str:
    def fmt_neg(d: dict[str, int]) -> str:
        return ", ".join(f"{k}: {v}" for k, v in d.items())

    lines = [
        "# Contrastive Dataset Confound Audit Report (`v1` vs `v2`)",
        "",
        "Generated automatically by `src/validate_dataset.py`.",
        "",
        "## Quantitative comparison",
        "",
        "| Metric | `v1` (Neutral Baseline) | `v2` (Symmetric Control) | Why it matters |",
        "|---|---:|---:|---|",
        f"| Records parsed | {v1['n']} | {v2['n']} | Pilot batch size |",
        f"| Raw JSONL parse errors (trailing `\\`) | {v1['raw_parse_errors']}/{v1['n']} | {v2['raw_parse_errors']}/{v2['n']} | Parser crashes unless sanitised |",
        f"| Identical `assistant_prefix` | {v1['prefix_matches']}/{v1['n']} | {v2['prefix_matches']}/{v2['n']} | Prevents lexical shortcuts on output tokens |",
        f"| Mean `honest` vs `deceptive` system words | {v1['mean_h_sys_words']} vs {v1['mean_d_sys_words']} | {v2['mean_h_sys_words']} vs {v2['mean_d_sys_words']} | Eliminates prompt-length shortcuts |",
        f"| Max system-prompt word diff | {v1['max_sys_word_diff']} words | {v2['max_sys_word_diff']} words | Target <= 4 words |",
        f"| Stakeholder presence (`honest` vs `deceptive`) | {v1['h_stakeholders']}/{v1['n']} vs {v1['d_stakeholders']}/{v1['n']} | {v2['h_stakeholders']}/{v2['n']} vs {v2['d_stakeholders']}/{v2['n']} | Stops probe firing on authority/boss keywords |",
        f"| Mean (min) prefix length | {v1['mean_prefix_words']}w (min {v1['min_prefix_words']}w) | {v2['mean_prefix_words']}w (min {v2['min_prefix_words']}w) | Prevents token starvation under 5-token mask |",
        f"| Consistent `exclude_last_n_tokens == 5` | {v1['mask_matches']}/{v1['n']} | {v2['mask_matches']}/{v2['n']} | Keeps activation extraction window uniform |",
        f"| Negation spread across categories | {fmt_neg(v1['neg_by_cat'])} | {fmt_neg(v2['neg_by_cat'])} | Prevents category-syntax entanglement |",
        f"| Prefix meta-narration / stakeholder leakage | {len(v1['prefix_leakage_ids'])}/{v1['n']} | {len(v2['prefix_leakage_ids'])}/{v2['n']} | Keeps assistant openings realistic/in-character |",
        f"| Boilerplate lock-in (`\"Your mandate is to\"`) | {v1['template_lock_count']}/{v1['n']} | {v2['template_lock_count']}/{v2['n']} | Signals need for style rotation when scaling |",
        "",
        "## Specific flags detected",
        "",
        f"- **`v1` length-asymmetric pairs (>4 word diff):** {', '.join(v1['asymmetric_len_ids']) or 'None'}",
        f"- **`v1` short prefixes (<12 words, starvation risk):** {', '.join(v1['short_prefixes']) or 'None'}",
        f"- **`v1` stakeholder-asymmetric pairs:** {', '.join(v1['stakeholder_mismatch_ids']) or 'None'}",
        f"- **`v2` meta-narration / stakeholder leakage in `assistant_prefix`:** {', '.join(v2['prefix_leakage_ids']) or 'None'}",
        "",
        "## Actionable takeaways for scaling (`v3` / production split)",
        "",
        "1. Preserve `v2`'s matched stakeholder and word-count symmetry (max diff <= 4 words).",
        "2. Strip trailing backslashes automatically during ingestion into `data/processed/`.",
        "3. Enforce in-character `assistant_prefix` phrasing so hidden stakeholder pressure stays strictly in the system prompt.",
        "4. Rotate 4-5 distinct directive templates when scaling past 400 pairs to prevent overfitting to `\"Your mandate is to\"`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit contrastive JSONL datasets for probe confounds.")
    parser.add_argument("--v1", type=Path, default=Path("data/raw/gpt_v1_raw.jsonl"))
    parser.add_argument("--v2", type=Path, default=Path("data/raw/gpt_v2_raw.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/audit_report.md"))
    args = parser.parse_args()

    v1_res = audit_file(args.v1)
    v2_res = audit_file(args.v2)
    report = render_markdown(v1_res, v2_res)

    args.out.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved audit report to {args.out}")


if __name__ == "__main__":
    main()
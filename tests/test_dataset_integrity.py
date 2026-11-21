"""Unit tests verifying dataset hygiene, symmetry invariants, hard negatives, and OOD eval."""

from __future__ import annotations

import json
import re
import unittest
from collections import Counter
from pathlib import Path

from src.validate_dataset import audit_file

PROCESSED_DIR = Path("data/processed")
EXPECTED_HN_SUBTYPES = {
    "honest_confession",
    "high_stakes_truth",
    "benign_roleplay",
    "jokes_sarcasm",
    "calibrated_ignorance",
    "privacy_safety_refusal",
}
EXPECTED_OOD_SUBTYPES = {
    "overt_lie": 15,
    "concealment": 15,
    "pressure_dishonesty": 15,
    "sandbagging": 15,
}
OOD_LEAK_RE = re.compile(
    r"\b(sandbag|deceptive|deceive|mislead|lying|lie to|hidden instruction)\b",
    re.IGNORECASE,
)


class TestDatasetIntegrity(unittest.TestCase):
    def test_v1_clean_parses_without_errors(self) -> None:
        res = audit_file(PROCESSED_DIR / "pilot_v1_clean.jsonl")
        self.assertEqual(res["raw_parse_errors"], 0)
        self.assertEqual(res["n"], 10)
        self.assertGreater(res["max_sys_word_diff"], 10)
        self.assertGreater(len(res["stakeholder_mismatch_ids"]), 0)

    def test_v2_clean_passes_all_symmetry_and_leakage_checks(self) -> None:
        res = audit_file(PROCESSED_DIR / "pilot_v2_clean.jsonl")
        self.assertEqual(res["raw_parse_errors"], 0)
        self.assertEqual(res["prefix_matches"], res["n"])
        self.assertEqual(res["user_matches"], res["n"])
        self.assertEqual(res["mask_matches"], res["n"])
        self.assertLessEqual(res["max_sys_word_diff"], 4)
        self.assertEqual(res["stakeholder_mismatch_ids"], [])
        self.assertEqual(res["short_prefixes"], [])
        self.assertEqual(res["prefix_leakage_ids"], [])
        self.assertEqual(res["neg_by_cat"], {"concealment": 1, "overt_lie": 2, "pressure_dishonesty": 1})

    def test_hard_negatives_cover_all_six_subtypes(self) -> None:
        path = PROCESSED_DIR / "hard_negatives.jsonl"
        rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        self.assertGreaterEqual(len(rows), 18)
        subtypes = {r["subtype"] for r in rows}
        self.assertEqual(subtypes, EXPECTED_HN_SUBTYPES)
        self.assertTrue(all(r["label"] == 0 for r in rows))

    def test_ood_deceptive_eval_suite_integrity(self) -> None:
        path = PROCESSED_DIR / "ood_deceptive_eval.jsonl"
        rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        self.assertEqual(len(rows), 60)
        self.assertEqual(len({r["id"] for r in rows}), 60)
        self.assertEqual(dict(Counter(r["subtype"] for r in rows)), EXPECTED_OOD_SUBTYPES)
        self.assertTrue(all(r["label"] == 1 for r in rows))
        self.assertTrue(all(r["exclude_last_n_tokens"] == 0 for r in rows))
        self.assertTrue(all(not OOD_LEAK_RE.search(r["assistant"]) for r in rows))
        self.assertEqual({r["generator"] for r in rows}, {"gpt-6.2-astra", "claude-opus-4.6"})


if __name__ == "__main__":
    unittest.main()

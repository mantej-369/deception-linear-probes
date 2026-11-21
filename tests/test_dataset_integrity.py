"""Unit tests verifying dataset hygiene, symmetry invariants, and hard-negative coverage."""

from __future__ import annotations

import json
import unittest
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


class TestDatasetIntegrity(unittest.TestCase):
    def test_v1_clean_parses_without_errors(self) -> None:
        res = audit_file(PROCESSED_DIR / "pilot_v1_clean.jsonl")
        self.assertEqual(res["raw_parse_errors"], 0)
        self.assertEqual(res["n"], 10)
        # Confirm v1 intentionally preserves baseline confounds for ablation
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
        # Confirm balanced negation distribution across all 3 categories
        self.assertEqual(res["neg_by_cat"], {"concealment": 1, "overt_lie": 2, "pressure_dishonesty": 1})

    def test_hard_negatives_cover_all_six_subtypes(self) -> None:
        path = PROCESSED_DIR / "hard_negatives.jsonl"
        rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        self.assertGreaterEqual(len(rows), 18)
        subtypes = {r["subtype"] for r in rows}
        self.assertEqual(subtypes, EXPECTED_HN_SUBTYPES)
        self.assertTrue(all(r["label"] == 0 for r in rows))


if __name__ == "__main__":
    unittest.main()
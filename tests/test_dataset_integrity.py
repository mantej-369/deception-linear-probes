"""Unit tests verifying dataset hygiene, symmetry invariants, hard negatives, OOD eval, and scaled v3."""

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

    def test_v3_symmetric_120_passes_all_conduits(self) -> None:
        path = PROCESSED_DIR / "v3_symmetric_120.jsonl"
        res = audit_file(path)
        self.assertEqual(res["n"], 120)
        self.assertEqual(res["raw_parse_errors"], 0)
        self.assertEqual(res["prefix_matches"], 120)
        self.assertEqual(res["user_matches"], 120)
        self.assertEqual(res["mask_matches"], 120)
        self.assertLessEqual(res["max_sys_word_diff"], 4)
        self.assertEqual(res["stakeholder_mismatch_ids"], [])
        self.assertEqual(res["short_prefixes"], [])
        self.assertEqual(res["prefix_leakage_ids"], [])
        self.assertEqual(res["template_lock_count"], 0)
        self.assertEqual(res["cat_counts"], {"overt_lie": 40, "concealment": 40, "pressure_dishonesty": 40})
        self.assertEqual(res["neg_by_cat"], {"concealment": 16, "overt_lie": 16, "pressure_dishonesty": 16})

        rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        self.assertEqual(len({r["pair_id"] for r in rows}), 120)
        self.assertEqual(dict(Counter(r["generator"] for r in rows)), {"gpt-6.2-astra": 60, "claude-opus-4.6": 60})
        self.assertEqual(
            dict(Counter(r["directive_style"] for r in rows)),
            {"A_policy": 24, "B_priority": 24, "C_direct_verb": 24, "D_editorial": 24, "E_stance": 24},
        )

    def test_v1_confounded_120_is_matched_ablation(self) -> None:
        v1_path = PROCESSED_DIR / "v1_confounded_120.jsonl"
        v3_path = PROCESSED_DIR / "v3_symmetric_120.jsonl"
        res = audit_file(v1_path)
        self.assertEqual(res["n"], 120)
        self.assertEqual(res["raw_parse_errors"], 0)
        self.assertEqual(res["prefix_matches"], 120)
        self.assertEqual(res["user_matches"], 120)
        self.assertEqual(res["h_stakeholders"], 0)
        self.assertEqual(res["d_stakeholders"], 120)
        self.assertGreaterEqual(res["mean_d_sys_words"] - res["mean_h_sys_words"], 12.0)

        v1_rows = [json.loads(ln) for ln in v1_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        v3_rows = [json.loads(ln) for ln in v3_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for r1, r3 in zip(v1_rows, v3_rows):
            self.assertEqual(r1["pair_id"], r3["pair_id"])
            self.assertEqual(r1["deceptive"], r3["deceptive"])
            self.assertEqual(r1["honest"]["assistant_prefix"], r3["honest"]["assistant_prefix"])


if __name__ == "__main__":
    unittest.main()

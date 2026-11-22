"""Unit tests enforcing exact symmetry, zero nuisance AUROC, and zero legacy overlap on reversible_worlds_v1."""

from __future__ import annotations

import unittest
from collections import Counter

from src.reversible_worlds import (
    DIRECTIVE_STYLES,
    audit_legacy_overlap,
    audit_scalar_nuisances,
    build_all_rows,
)


class TestReversibleWorlds(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = build_all_rows()

    def test_exact_arm_counts_and_unique_ids(self) -> None:
        self.assertEqual(len(self.rows), 1192)
        self.assertEqual(len({r["id"] for r in self.rows}), len(self.rows))
        arm_counts = Counter(r["arm"] for r in self.rows)
        self.assertEqual(
            dict(arm_counts),
            {"reversible_contextual": 480, "benign_control": 280, "parametric_contextual": 432},
        )

    def test_directive_templates_have_exact_equal_word_counts(self) -> None:
        for style in DIRECTIVE_STYLES:
            lens = {k: len(v.split()) for k, v in style.items()}
            self.assertEqual(lens, {"honest": 14, "distort": 14, "neutral": 14})

    def test_arm_a_exact_fixed_string_symmetry_and_zero_nuisance_auroc(self) -> None:
        nuis = audit_scalar_nuisances(self.rows)
        self.assertEqual(nuis["arm_a_scalar_auroc_sys_words"], 0.5)
        self.assertEqual(nuis["arm_a_scalar_auroc_user_words"], 0.5)
        self.assertEqual(nuis["arm_a_scalar_auroc_asst_words"], 0.5)

        arm_a = [r for r in self.rows if r["arm"] == "reversible_contextual"]
        sys_true = Counter(r["system"] for r in arm_a if r["is_true"] == 1)
        sys_false = Counter(r["system"] for r in arm_a if r["is_true"] == 0)
        self.assertEqual(sys_true, sys_false)

        asst_true = Counter(r["assistant"] for r in arm_a if r["is_true"] == 1)
        asst_false = Counter(r["assistant"] for r in arm_a if r["is_true"] == 0)
        self.assertEqual(asst_true, asst_false)

    def test_zero_user_ngram_overlap_with_legacy_dev_splits(self) -> None:
        ov = audit_legacy_overlap(self.rows)
        self.assertGreaterEqual(ov["files_checked"], 6)
        self.assertEqual(ov["user_5gram_overlaps"], 0)


if __name__ == "__main__":
    unittest.main()

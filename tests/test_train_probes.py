"""Unit tests for pair-grouped CV splitting, selectivity math, and strict FPR thresholding."""

from __future__ import annotations

import unittest

from src.train_probes import (
    build_pair_folds,
    compute_selectivity,
    compute_threshold_at_fpr,
    extract_pair_id,
    positive_rate_at_threshold,
)


class TestTrainProbes(unittest.TestCase):
    def test_extract_pair_id(self) -> None:
        self.assertEqual(extract_pair_id("overt_001_honest"), "overt_001")
        self.assertEqual(extract_pair_id("pressure_003_deceptive"), "pressure_003")
        self.assertEqual(extract_pair_id("hn_confess_001"), "hn_confess_001")

    def test_pair_folds_have_zero_pair_leakage(self) -> None:
        ids = [f"pair_{i:02d}_{side}" for i in range(10) for side in ("honest", "deceptive")]
        folds = build_pair_folds(ids, n_splits=5, seed=42)
        self.assertEqual(len(folds), 5)

        seen_val = set()
        for train_idx, val_idx in folds:
            train_pairs = {extract_pair_id(ids[i]) for i in train_idx}
            val_pairs = {extract_pair_id(ids[i]) for i in val_idx}
            self.assertEqual(train_pairs.intersection(val_pairs), set())
            seen_val.update(val_idx)
        self.assertEqual(len(seen_val), len(ids))

    def test_threshold_guarantees_target_fpr_budget(self) -> None:
        benign_scores = [i / 100.0 for i in range(100)]
        tau = compute_threshold_at_fpr(benign_scores, target_fpr=0.01)
        empirical_fpr = positive_rate_at_threshold(benign_scores, tau)
        self.assertLessEqual(empirical_fpr, 0.01)

    def test_selectivity_calculation(self) -> None:
        self.assertAlmostEqual(compute_selectivity(0.92, 0.51), 0.41)


if __name__ == "__main__":
    unittest.main()
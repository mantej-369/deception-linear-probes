"""Unit tests for Step 4 calibrated evaluation, pure ROC-AUC, and OOD recall metrics."""

from __future__ import annotations

import unittest

from src.evaluate import compute_roc_auc_pure, evaluate_probe_scores


class TestEvaluate(unittest.TestCase):
    def test_pure_roc_auc_perfect_reversed_and_ties(self) -> None:
        self.assertEqual(compute_roc_auc_pure([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(compute_roc_auc_pure([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]), 0.0)
        self.assertEqual(compute_roc_auc_pure([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]), 0.5)
        # Mixed with one tie across classes: (0.1->0, 0.5->0, 0.5->1, 0.9->1) -> AUC = 0.875
        self.assertEqual(compute_roc_auc_pure([0, 0, 1, 1], [0.1, 0.5, 0.5, 0.9]), 0.875)

    def test_pure_roc_auc_requires_both_classes(self) -> None:
        with self.assertRaises(ValueError):
            compute_roc_auc_pure([0, 0, 0], [0.1, 0.2, 0.3])

    def test_full_calibrated_eval_with_ood_recall_and_auroc(self) -> None:
        benign = [i / 200.0 for i in range(100)]  # max = 0.495, 1% FPR tau = 0.490
        hard_negs = [0.10, 0.20, 0.85, 0.30]
        hn_cats = ["honest_confession", "high_stakes_truth", "jokes_sarcasm", "honest_confession"]
        ood_deceptive = [0.95, 0.90, 0.40, 0.80]
        ood_cats = ["overt_lie", "sandbagging", "concealment", "overt_lie"]

        report = evaluate_probe_scores(
            benign_scores=benign,
            hard_neg_scores=hard_negs,
            hard_neg_categories=hn_cats,
            ood_deceptive_scores=ood_deceptive,
            ood_deceptive_categories=ood_cats,
            target_fpr=0.01,
        )

        # Benign calibration guarantees <= 1% FPR
        self.assertLessEqual(report["benign_eval"]["empirical_fpr"], 0.01)
        tau = report["benign_eval"]["threshold"]
        self.assertAlmostEqual(tau, 0.49)

        # Hard negatives: only jokes_sarcasm (0.85) exceeds 0.49
        self.assertEqual(report["hard_negative_eval"]["overall_false_positive_rate"], 0.25)
        self.assertEqual(report["hard_negative_eval"]["by_subtype"]["jokes_sarcasm"]["fpr"], 1.0)
        self.assertEqual(report["hard_negative_eval"]["by_subtype"]["honest_confession"]["fpr"], 0.0)

        # OOD deceptive: 3 of 4 exceed 0.49 -> 0.75 recall
        ood = report["ood_deceptive_eval"]
        self.assertEqual(ood["recall_at_threshold"], 0.75)
        self.assertEqual(ood["by_category"]["overt_lie"]["recall"], 1.0)
        self.assertEqual(ood["by_category"]["concealment"]["recall"], 0.0)
        self.assertGreater(ood["auroc_vs_benign"], 0.90)
        self.assertGreater(ood["auroc_vs_hard_negatives"], 0.80)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for paired world bootstrap, exact sign test, 2x2 direction geometry, and reversible summary."""

from __future__ import annotations

import unittest
import numpy as np

from src.eval_reversible import (
    POSITIONS,
    compute_2x2_directions,
    exact_sign_test_p,
    paired_world_bootstrap,
    summarize_reversible_results,
)


class TestEvalReversible(unittest.TestCase):
    def test_exact_sign_test_matches_audit_example(self) -> None:
        self.assertAlmostEqual(exact_sign_test_p(4, 4), 0.125, places=6)
        self.assertAlmostEqual(exact_sign_test_p(0, 4), 0.125, places=6)
        self.assertAlmostEqual(exact_sign_test_p(2, 4), 1.0, places=6)

    def test_paired_world_bootstrap_ci_and_counts(self) -> None:
        diffs = {f"w{i}": [1.0 + 0.1 * i, 1.2 + 0.1 * i] for i in range(10)}
        res = paired_world_bootstrap(diffs, n_boot=500, seed=42)
        self.assertEqual(res["n_worlds"], 10)
        self.assertEqual(res["pos_worlds"], 10)
        self.assertLess(res["sign_p"], 0.002)
        self.assertLess(res["ci95"][0], res["mean"])
        self.assertGreater(res["ci95"][1], res["mean"])

    def test_compute_2x2_directions_geometry_and_shapes(self) -> None:
        rng = np.random.default_rng(7)
        X3 = rng.normal(size=(20, 8))
        y3 = np.array([0, 1] * 10)
        X3[y3 == 1, 0] += 2.0
        X3[y3 == 1, 1] += 2.0
        X4 = rng.normal(size=(10, 8))
        y4 = np.array([0, 1] * 5)
        X4[y4 == 1, 0] += 2.0
        X4[y4 == 1, 1] -= 2.0

        d = compute_2x2_directions(X3, y3, X4, y4)
        self.assertAlmostEqual(float(np.linalg.norm(d["w_dec_raw_unit"])), 1.0, places=5)
        self.assertAlmostEqual(float(np.linalg.norm(d["w_dec_std_unit"])), 1.0, places=5)
        self.assertGreater(abs(d["w_dec_raw_unit"][0]), 0.6)
        self.assertGreater(abs(d["w_cmp_raw_unit"][1]), 0.6)

    def test_summarize_reversible_results_detects_fixed_string_flip(self) -> None:
        records = []
        for wid in ("rw_001", "rw_002", "rw_003", "rw_004"):
            for is_true in (1, 0):
                sc = {p: (2.0 if (is_true == 0 and p in ("answer_tok", "suffix_end")) else 0.0) for p in POSITIONS}
                records.append(
                    {
                        "arm": "reversible_contextual",
                        "world_id": wid,
                        "directive": "distort",
                        "evidence_condition": "contextual",
                        "answer_choice": "A",
                        "is_true": is_true,
                        "label": 1 - is_true,
                        "scores": {"L22_dec_raw": sc},
                    }
                )
            records.append(
                {
                    "arm": "benign_control",
                    "world_id": wid,
                    "directive": "benign",
                    "evidence_condition": "benign_quote",
                    "answer_choice": "B",
                    "is_true": 1,
                    "label": 0,
                    "scores": {"L22_dec_raw": {p: 0.1 for p in POSITIONS}},
                }
            )
            for ans in ("A", "B"):
                records.append(
                    {
                        "arm": "parametric_contextual",
                        "world_id": wid,
                        "directive": "honest",
                        "evidence_condition": "with_evidence",
                        "answer_choice": ans,
                        "is_true": int(ans == "A"),
                        "label": int(ans == "B"),
                        "scores": {"L22_dec_raw": {p: (1.5 if ans == "B" else 0.0) for p in POSITIONS}},
                    }
                )

        out = summarize_reversible_results(records, "L22_dec_raw")
        self.assertAlmostEqual(out["arm_a_truth_reversal"]["header"]["overall"]["mean"], 0.0)
        self.assertAlmostEqual(out["arm_a_truth_reversal"]["suffix_end"]["overall"]["mean"], 2.0)
        self.assertEqual(out["arm_b_benign"]["suffix_end"]["benign_quote"]["auroc_distort_false_vs_cond"], 1.0)
        self.assertAlmostEqual(out["arm_c_parametric"]["suffix_end"]["with_evidence"]["mean"], 1.5)


if __name__ == "__main__":
    unittest.main()

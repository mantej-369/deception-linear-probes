"""Unit tests for odd/even decomposition, orthogonal random controls, and steering suite construction."""

from __future__ import annotations

import unittest
from collections import Counter
import numpy as np

from src.reversible_worlds import build_all_rows
from src.steer_eval import (
    build_benign_rule_items,
    build_reversible_steering_items,
    decompose_odd_even,
    make_orthogonal_random_controls,
    summarize_steering_records,
)


class TestSteerEval(unittest.TestCase):
    def test_odd_even_decomposition_separates_directional_from_u_shape(self) -> None:
        # Pure linear directional response: f(-a)=-3, f(0)=0, f(+a)=+3
        odd, even = decompose_odd_even(-3.0, 0.0, 3.0)
        self.assertAlmostEqual(odd, 3.0)
        self.assertAlmostEqual(even, 0.0)

        # Pure symmetric U-shaped disruption: f(-a)=+4, f(0)=0, f(+a)=+4
        odd_u, even_u = decompose_odd_even(4.0, 0.0, 4.0)
        self.assertAlmostEqual(odd_u, 0.0)
        self.assertAlmostEqual(even_u, 4.0)

    def test_orthogonal_random_controls_are_unit_and_orthogonal(self) -> None:
        rng = np.random.default_rng(1)
        b1 = rng.standard_normal(32)
        b2 = rng.standard_normal(32)
        ctrls = make_orthogonal_random_controls([b1, b2], n_controls=3, seed=42)
        self.assertEqual(len(ctrls), 3)
        for c in ctrls:
            self.assertAlmostEqual(float(np.linalg.norm(c)), 1.0, places=5)
            self.assertAlmostEqual(float(np.dot(c, b1)), 0.0, places=5)
            self.assertAlmostEqual(float(np.dot(c, b2)), 0.0, places=5)
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertAlmostEqual(float(np.dot(ctrls[i], ctrls[j])), 0.0, places=5)

    def test_steering_suites_are_fully_counterbalanced(self) -> None:
        rows = build_all_rows()
        rev = build_reversible_steering_items(rows)
        br = build_benign_rule_items(n_worlds=20)
        self.assertEqual(len(rev), 160)
        self.assertEqual(len(br), 40)

        # Every candidate appears equally often as pos_cand and neg_cand in both suites
        self.assertEqual(Counter(x["pos_cand"] for x in rev), Counter(x["neg_cand"] for x in rev))
        self.assertEqual(Counter(x["pos_cand"] for x in br), Counter(x["neg_cand"] for x in br))

    def test_summarize_steering_records_computes_odd_even(self) -> None:
        recs = []
        for wid in ("w1", "w2", "w3", "w4"):
            for alpha, lo in [(-1.0, -2.0), (0.0, 0.0), (1.0, 2.0)]:
                recs.append(
                    {
                        "id": f"i_{wid}",
                        "suite": "reversible_distort",
                        "world_id": wid,
                        "axis": "dec",
                        "alpha": alpha,
                        "log_odds": lo,
                        "cond_prob": 0.5,
                        "p_pos": 0.4,
                        "p_neg": 0.4,
                        "pair_mass": 0.8,
                        "entropy": 1.0,
                        "kl_vs_base": abs(alpha) * 0.1,
                    }
                )
        s = summarize_steering_records(recs, [-1.0, 0.0, 1.0])
        oe = s["reversible_distort"]["dec"]["odd_even"]["pm_1.00"]
        self.assertAlmostEqual(oe["odd_directional"]["mean"], 2.0)
        self.assertAlmostEqual(oe["even_disruption"]["mean"], 0.0)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for Phase 3 balanced on-policy lying, benign surprisal, and orthogonalisation fix."""

from __future__ import annotations

import unittest
from collections import Counter
import numpy as np

from src.surprisal_confound import (
    analyze_layer_orthogonalisation,
    audit_surprisal_dataset,
    build_surprisal_confound_rows,
    project_out_direction,
)


class TestSurprisalConfound(unittest.TestCase):
    def test_exact_counts_symmetry_and_zero_nuisance_auroc(self) -> None:
        rows = build_surprisal_confound_rows()
        self.assertEqual(len(rows), 480)
        self.assertEqual(len({r["id"] for r in rows}), 480)
        audit = audit_surprisal_dataset(rows)
        for arm in ("cover_honest", "cover_lie", "benign_surprisal"):
            self.assertEqual(audit[arm]["n"], 160)
            self.assertEqual(audit[arm]["auroc_sys_words"], 0.5)
            self.assertEqual(audit[arm]["auroc_user_words"], 0.5)
            self.assertEqual(audit[arm]["auroc_asst_words"], 0.5)

            sub = [r for r in rows if r["arm"] == arm]
            self.assertEqual(
                Counter(r["system"] for r in sub if r["label"] == 0),
                Counter(r["system"] for r in sub if r["label"] == 1),
            )
            self.assertEqual(
                Counter(r["assistant"] for r in sub if r["label"] == 0),
                Counter(r["assistant"] for r in sub if r["label"] == 1),
            )
            self.assertEqual(Counter(r["mention_order"] for r in sub), {"confirmed_first": 80, "alternative_first": 80})

    def test_project_out_direction_removes_nuisance_and_normalises(self) -> None:
        e_sig = np.array([1.0, 0.0, 0.0, 0.0])
        e_surp = np.array([0.0, 1.0, 0.0, 0.0])
        w_mixed = 0.5 * e_sig + 2.0 * e_surp
        w_clean = project_out_direction(w_mixed, e_surp)
        self.assertAlmostEqual(float(np.linalg.norm(w_clean)), 1.0, places=5)
        self.assertAlmostEqual(float(np.dot(w_clean, e_surp)), 0.0, places=5)
        self.assertAlmostEqual(float(np.dot(w_clean, e_sig)), 1.0, places=5)

    def test_orthogonalisation_rescues_inverted_on_policy_lies_synthetically(self) -> None:
        # Synthetic world where w_dec = true_lie_axis + 2*surprisal_axis; on-policy lies have negative surprisal
        rows = build_surprisal_confound_rows()
        meta_arm = {arm: [r for r in rows if r["arm"] == arm] for arm in ("cover_honest", "cover_lie", "benign_surprisal")}
        rng = np.random.default_rng(0)
        dim = 8
        H_arm = {}
        pair_lo = {}

        for arm, mlist in meta_arm.items():
            X = rng.normal(scale=0.05, size=(len(mlist), dim))
            for i, r in enumerate(mlist):
                lbl = int(r["label"])
                if arm == "benign_surprisal":
                    X[i, 1] += 2.0 * lbl  # pure surprisal along dim 1
                    pair_lo[r["pair_key"]] = -5.0
                elif arm == "cover_honest":
                    X[i, 0] += 1.0 * lbl  # truth violation along dim 0
                    X[i, 1] += 2.0 * lbl  # off-policy lie is surprising
                    pair_lo[r["pair_key"]] = -5.0
                else:  # cover_lie: model prefers lie on-policy, so TRUTH (lbl=0) is surprising!
                    X[i, 0] += 1.0 * lbl
                    X[i, 1] += 2.0 * (1 - lbl)
                    pair_lo[r["pair_key"]] = +5.0
            H_arm[arm] = X

        w_dec = np.zeros(dim)
        w_dec[0] = 1.0
        w_dec[1] = 2.0
        res = analyze_layer_orthogonalisation(H_arm, meta_arm, pair_lo, w_dec)
        # Before fix: w_dec fires strongly on benign surprisal & flips NEGATIVE on on-policy lies
        self.assertGreater(res["benign_surprisal"]["w_dec"]["mean"], 1.5)
        self.assertLess(res["cover_lie_verified_flips"]["w_dec"]["mean"], -1.0)
        # After fix: benign surprisal collapses to ~0 and on-policy lies flip POSITIVE!
        self.assertLess(abs(res["benign_surprisal"]["w_perp_cv"]["mean"]), 0.15)
        self.assertGreater(res["cover_lie_verified_flips"]["w_perp"]["mean"], 0.8)


if __name__ == "__main__":
    unittest.main()

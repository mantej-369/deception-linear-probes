"""Unit tests for Phase 4 Part A fixed-directive expectation disentanglement & LOFO probability baseline."""

from __future__ import annotations

import unittest
from collections import Counter

import numpy as np

from src.phase4_disentangle import (
    PHASE4_DIRECTIVES,
    PHASE4_DISENTANGLE_WORLDS,
    audit_phase4_nuisances,
    audit_phase4_overlap,
    build_phase4_disentangle_rows,
    summarize_phase4_disentangle,
)


class TestPhase4Disentangle(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = build_phase4_disentangle_rows()

    def test_exact_counts_within_world_crossing_and_symmetry(self) -> None:
        self.assertEqual(len(PHASE4_DISENTANGLE_WORLDS), 24)
        self.assertEqual(len({w[1] for w in PHASE4_DISENTANGLE_WORLDS}), 8)
        self.assertEqual(len(self.rows), 1536)
        self.assertEqual(len({r["id"] for r in self.rows}), 1536)
        self.assertEqual({k: len(v.split()) for k, v in PHASE4_DIRECTIVES.items()}, {"honest": 14, "distort": 14})

        # Every world has both mention orders equally represented
        for wid, *_ in PHASE4_DISENTANGLE_WORLDS:
            wrows = [r for r in self.rows if r["world_id"] == wid]
            self.assertEqual(Counter(r["mention_order"] for r in wrows), {"fact_first": 32, "fact_second": 32})

        # Exact fixed-string symmetry inside BOTH prior_favors_false strata
        for pf in (0, 1):
            sub = [r for r in self.rows if r["prior_favors_false"] == pf]
            self.assertEqual(
                Counter(r["assistant"] for r in sub if r["is_true"] == 1),
                Counter(r["assistant"] for r in sub if r["is_true"] == 0),
            )

    def test_zero_scalar_nuisance_and_zero_overlap_with_all_prior_splits(self) -> None:
        nuis = audit_phase4_nuisances(self.rows)
        for k, v in nuis.items():
            if k != "n_total":
                self.assertEqual(v, 0.5, msg=f"Non-chance scalar nuisance {k}={v}")

        ov = audit_phase4_overlap(self.rows)
        self.assertGreaterEqual(ov["files_checked"], 7)
        self.assertEqual(ov["user_5gram_overlaps"], 0)

    def test_disentanglement_summary_separates_surprisal_from_truth_and_detects_incremental_auroc(self) -> None:
        rng = np.random.default_rng(7)
        records = []
        # Synthetic world: probe = +1.2 * truth_violation + 0.5 * (-log_odds_of_ans)
        # Under honest: prior_false shifts pair_lo from -4 to +3 (crossing zero!)
        for r in self.rows:
            if r["mechanism"] != "draft_autofill":
                continue
            lo = (3.0 if r["prior_favors_false"] == 1 else -4.0) + float(rng.normal(0, 0.2))
            if r["directive"] == "distort":
                lo += 2.0
            ans_rel_lo = lo if r["label"] == 1 else -lo
            log_p_ans = -float(np.log1p(np.exp(-np.clip(ans_rel_lo, -20.0, 20.0))))
            # Probe has BOTH genuine truth signal (+1.2) AND surprisal component (+0.4 * -ans_rel_lo)
            score_ans = 1.2 * r["label"] + 0.4 * (-ans_rel_lo) + float(rng.normal(0, 0.15))
            records.append(
                {
                    **r,
                    "ans_tid": 1,
                    "pos_tid": 1,
                    "neg_tid": 2,
                    "positions": {"header": 1, "prefix_end": 2, "answer_tok": 3, "suffix_end": 4},
                    "log_p_ans": log_p_ans,
                    "log_p_alt": log_p_ans - ans_rel_lo,
                    "ans_rel_log_odds": ans_rel_lo,
                    "pair_log_odds": lo,
                    "pair_mass": 0.9,
                    "top_tokens": [],
                    "scores": {"L24_dec_std": {"header": 0.0, "prefix_end": 0.0, "answer_tok": score_ans, "suffix_end": 0.1 * score_ans}},
                }
            )

        summary, per_world = summarize_phase4_disentangle(records, ["L24_dec_std"], "L24_dec_std")
        # 1. Manipulation check detects the +7.0 logit shift
        mc = summary["manipulation_check"]["draft_autofill__honest"]
        self.assertGreater(mc["paired_delta_lo"]["mean"], 6.0)
        self.assertEqual(mc["frac_lo_pos_prior_false"], 1.0)

        # 2. Disentanglement detects negative interaction (surprisal shifts diff down) AND positive intercept@lo=0 (truth!)
        dis = summary["fixed_directive_disentanglement"]["L24_dec_std"]["answer_tok"]["draft_autofill__honest"]
        self.assertLess(dis["interaction_prior_false_minus_true"]["mean"], -4.0)
        self.assertGreater(dis["ols_diff_on_surprisal"]["intercept_at_zero"], 0.8)

        # 3. Incremental value: probe+prob beats prob_only across all 8 unseen families!
        inc = summary["incremental_vs_probability"]["L24_dec_std"]["answer_tok"]["honest"]
        self.assertEqual(inc["families_improved"], 8)
        self.assertGreater(inc["pooled_delta_both_minus_prob"], 0.1)
        self.assertGreater(len(per_world), 0)


if __name__ == "__main__":
    unittest.main()

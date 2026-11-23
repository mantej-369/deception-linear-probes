"""Regression tests for Issue 1 (Phase 4A label invariants & manipulation check)."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.make_figures import collect_verified_tables, format_verified_markdown


class TestPhase4AAudit(unittest.TestCase):
    def test_issue1_label_invariants_and_llama_antiprior_check(self) -> None:
        tables = collect_verified_tables(Path("results"))
        mc_q = tables["phase4a_manipulation_check"]["qwen_L24"]
        mc_l = tables["phase4a_manipulation_check"]["llama_L19"]

        self.assertEqual(mc_q["label_audit"]["label_consistency_mismatches"], 0)
        self.assertEqual(mc_l["label_audit"]["label_consistency_mismatches"], 0)
        self.assertEqual(mc_q["label_audit"]["n_rows"], 1536)
        self.assertEqual(mc_l["label_audit"]["n_rows"], 1536)

        self.assertGreater(mc_q["fewshot_streak__honest"]["paired_delta_lo"]["mean"], 10.0)
        self.assertAlmostEqual(mc_l["all_mechanisms__honest"]["paired_delta_lo"]["mean"], -2.8557, places=3)
        self.assertEqual(mc_l["all_mechanisms__honest"]["n_lo_pos_prior_true"], 112)
        self.assertEqual(mc_l["all_mechanisms__honest"]["n_lo_pos_prior_false"], 26)

        md = format_verified_markdown(tables)
        self.assertIn("0/1536", md)
        self.assertIn("anti-prior repulsion", md)

    def test_issue2_fig2_surprisal_axis_and_ols_sign_match(self) -> None:
        import json
        from collections import defaultdict
        import numpy as np

        src = Path("src/make_figures.py").read_text(encoding="utf-8")
        self.assertIn("slope, inter = np.polyfit(surp_a, arr, 1)", src)
        self.assertIn("r_val = float(np.corrcoef(surp_a, arr)[0, 1])", src)
        self.assertNotIn("np.corrcoef(-los_a, arr)", src)

        tables = collect_verified_tables(Path("results"))
        specs = [
            ("results/phase4/qwen3b_disentangle/per_item_scores.jsonl", 24, {"fewshot_streak"}, "qwen_fewshot_honest_raw", "qwen_fewshot_honest_std"),
            ("results/phase4/llama32_3b_disentangle/per_item_scores.jsonl", 19, {"draft_autofill", "fewshot_streak"}, "llama_all_honest_raw", "llama_all_honest_std"),
        ]
        for path_str, L, mechs, k_raw, k_std in specs:
            grouped = defaultdict(dict)
            for ln in Path(path_str).read_text(encoding="utf-8").splitlines():
                if not ln.strip():
                    continue
                r = json.loads(ln)
                if r["directive"] == "honest" and r["mechanism"] in mechs:
                    grouped[r["pair_key"]][int(r["label"])] = r
            surp_a = np.array([-float(d[0]["pair_log_odds"]) for d in grouped.values()])
            for sfx, tkey in [("dec_raw", k_raw), ("dec_std", k_std)]:
                arr = np.array([float(d[1]["scores"][f"L{L}_{sfx}"]["answer_tok"] - d[0]["scores"][f"L{L}_{sfx}"]["answer_tok"]) for d in grouped.values()])
                slope, inter = np.polyfit(surp_a, arr, 1)
                r_val = float(np.corrcoef(surp_a, arr)[0, 1])
                ref = tables["phase4a_disentangle_primary"][tkey]["ols_diff_on_surprisal"]
                self.assertGreater(slope, 0.0)
                self.assertGreater(r_val, 0.0)
                self.assertAlmostEqual(slope, ref["slope"], places=5)
                self.assertAlmostEqual(inter, ref["intercept_at_zero"], places=5)
                self.assertAlmostEqual(r_val, ref["pearson_r"], places=5)


if __name__ == "__main__":
    unittest.main()

"""Unit tests verifying auto-generated manuscript tables match committed Phase 1-4 artefacts."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.make_figures import collect_verified_tables, format_verified_markdown


class TestMakeFigures(unittest.TestCase):
    def test_collect_verified_tables_matches_committed_artefacts(self) -> None:
        tables = collect_verified_tables(Path("results"))
        self.assertEqual(tables["provenance"]["qwen3b_model_commit"], "aa8e72537993ba99e69dfaafa59ed015b17504d1")
        self.assertEqual(tables["provenance"]["llama32_3b_model_commit"], "006f5dcd1393c3add266de40994ba96225e9689d")
        self.assertEqual(tables["phase3_mirror"]["qwen_flips"]["total"], 80)
        self.assertEqual(tables["phase3_mirror"]["llama_flips"]["total"], 80)
        self.assertLess(tables["phase2_steering_cos_llama_L19"], -0.45)

        md = format_verified_markdown(tables)
        self.assertIn("aa8e72537993ba99e69dfaafa59ed015b17504d1", md)
        self.assertIn("006f5dcd1393c3add266de40994ba96225e9689d", md)


if __name__ == "__main__":
    unittest.main()

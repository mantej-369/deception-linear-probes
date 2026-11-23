"""Regression tests verifying README.md links, module references, and alignment with verified_tables.json."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from src.make_figures import collect_verified_tables


class TestReadmeIntegrity(unittest.TestCase):
    def test_readme_links_and_verified_numbers(self) -> None:
        readme = Path("README.md").read_text(encoding="utf-8")

        links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", readme)
        self.assertGreaterEqual(len(links), 15)
        for target in links:
            if target.startswith(("http://", "https://", "#")):
                continue
            self.assertTrue(Path(target).exists(), f"Broken README link: {target}")

        self.assertNotIn("src/steer.py", readme)
        self.assertNotIn("- [ ] Phase 1b:", readme)
        self.assertNotIn("- [ ] Phase 2:", readme)
        self.assertNotIn("- [ ] Phase 3-5:", readme)

        for mod in re.findall(r"`([a-z0-9_]+\.py)`", readme):
            self.assertTrue((Path("src") / mod).exists(), f"Missing src module in README: {mod}")

        tables = collect_verified_tables(Path("results"))
        self.assertIn(tables["provenance"]["qwen3b_model_commit"], readme)
        self.assertIn(tables["provenance"]["llama32_3b_model_commit"], readme)
        self.assertIn(f"{tables['phase3_mirror']['qwen_L24_raw']['cover_lie_verified_flips']['w_dec']['mean']:+.3f}", readme)
        self.assertIn(f"{tables['phase3_mirror']['llama_L19_raw']['cover_lie_verified_flips']['w_dec']['mean']:+.3f}", readme)
        self.assertIn(f"{tables['phase4b_decomposition']['qwen_L24']['within_pressured_incentive_aurocs']['prompt_len']:.4f}", readme)
        self.assertIn("provisional regex grading", readme.lower())


if __name__ == "__main__":
    unittest.main()

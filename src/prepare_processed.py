"""Sanitise raw pilot JSONL files into data/processed/ while logging prefix repairs."""

from __future__ import annotations

import json
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Replace meta-narration / stakeholder-leaking prefixes in v2 while preserving negation balance
V2_IN_CHARACTER_PREFIXES = {
    "overt_001": (
        "Based on this quarter's optical laboratory trials, our preliminary measurement "
        "data does not yet settle the question of whether"
    ),
    "overt_002": (
        "This quarterly portfolio note reviews our current corporate bond holdings "
        "alongside historical yield movements to clarify whether"
    ),
    "overt_003": (
        "Visitors reading this city guide map cannot determine the hemisphere from "
        "street names alone, so this caption notes whether"
    ),
    "overt_004": (
        "When preparing for seasonal respiratory illness, patients frequently ask our "
        "clinic care team to clarify whether antibiotics"
    ),
    "pressure_002": (
        "Single-month regional labour surveys never capture every seasonal shift on "
        "their own, so this quarterly briefing assesses whether"
    ),
}


def load_raw_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cleaned = line.strip().rstrip("\\")
        if cleaned:
            rows.append(json.loads(cleaned))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    content = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    path.write_text(content, encoding="utf-8")


def main() -> None:
    # 1. Clean v1 syntax only (confounds intentionally kept for v1-vs-v2 ablation)
    v1_rows = load_raw_jsonl(RAW_DIR / "gpt_v1_raw.jsonl")
    write_jsonl(OUT_DIR / "pilot_v1_clean.jsonl", v1_rows)

    # 2. Clean v2 syntax + repair meta-narration prefixes to stay strictly in-character
    v2_rows = load_raw_jsonl(RAW_DIR / "gpt_v2_raw.jsonl")
    repaired = 0
    for row in v2_rows:
        pid = row["pair_id"]
        if pid in V2_IN_CHARACTER_PREFIXES:
            new_pref = V2_IN_CHARACTER_PREFIXES[pid]
            row["honest"]["assistant_prefix"] = new_pref
            row["deceptive"]["assistant_prefix"] = new_pref
            repaired += 1
    write_jsonl(OUT_DIR / "pilot_v2_clean.jsonl", v2_rows)

    print(f"Wrote {len(v1_rows)} records to data/processed/pilot_v1_clean.jsonl")
    print(f"Wrote {len(v2_rows)} records to data/processed/pilot_v2_clean.jsonl ({repaired} prefixes repaired)")


if __name__ == "__main__":
    main()
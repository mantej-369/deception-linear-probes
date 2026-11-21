# Contrastive Dataset Confound Audit Report (`v1` vs `v2`)

Generated automatically by `src/validate_dataset.py`.

## Quantitative comparison

| Metric | `v1` (Neutral Baseline) | `v2` (Symmetric Control) | Why it matters |
|---|---:|---:|---|
| Records parsed | 10 | 10 | Pilot batch size |
| Raw JSONL parse errors (trailing `\`) | 9/10 | 9/10 | Parser crashes unless sanitised |
| Identical `assistant_prefix` | 10/10 | 10/10 | Prevents lexical shortcuts on output tokens |
| Mean `honest` vs `deceptive` system words | 23.5 vs 45.8 | 36.9 vs 37.2 | Eliminates prompt-length shortcuts |
| Max system-prompt word diff | 30 words | 1 words | Target <= 4 words |
| Stakeholder presence (`honest` vs `deceptive`) | 1/10 vs 10/10 | 10/10 vs 10/10 | Stops probe firing on authority/boss keywords |
| Mean (min) prefix length | 7w (min 6w) | 16.8w (min 15w) | Prevents token starvation under 5-token mask |
| Consistent `exclude_last_n_tokens == 5` | 9/10 | 10/10 | Keeps activation extraction window uniform |
| Negation spread across categories | concealment: 0, overt_lie: 4, pressure_dishonesty: 0 | concealment: 1, overt_lie: 2, pressure_dishonesty: 1 | Prevents category-syntax entanglement |
| Prefix meta-narration / stakeholder leakage | 0/10 | 5/10 | Keeps assistant openings realistic/in-character |
| Boilerplate lock-in (`"Your mandate is to"`) | 0/10 | 10/10 | Signals need for style rotation when scaling |

## Specific flags detected

- **`v1` length-asymmetric pairs (>4 word diff):** overt_001 (19 vs 38w), overt_002 (21 vs 47w), overt_003 (22 vs 47w), overt_004 (21 vs 46w), conceal_001 (26 vs 39w), conceal_002 (23 vs 49w), conceal_003 (24 vs 44w), pressure_001 (25 vs 55w), pressure_002 (27 vs 49w), pressure_003 (27 vs 44w)
- **`v1` short prefixes (<12 words, starvation risk):** overt_001 (7w), overt_002 (7w), overt_003 (6w), overt_004 (6w), conceal_001 (7w), conceal_002 (8w), conceal_003 (8w), pressure_001 (7w), pressure_002 (8w), pressure_003 (6w)
- **`v1` stakeholder-asymmetric pairs:** overt_001, overt_002, overt_003, overt_004, conceal_001, conceal_002, conceal_003, pressure_002, pressure_003
- **`v2` meta-narration / stakeholder leakage in `assistant_prefix`:** overt_001, overt_002, overt_003, overt_004, pressure_002

## Actionable takeaways for scaling (`v3` / production split)

1. Preserve `v2`'s matched stakeholder and word-count symmetry (max diff <= 4 words).
2. Strip trailing backslashes automatically during ingestion into `data/processed/`.
3. Enforce in-character `assistant_prefix` phrasing so hidden stakeholder pressure stays strictly in the system prompt.
4. Rotate 4-5 distinct directive templates when scaling past 400 pairs to prevent overfitting to `"Your mandate is to"`.

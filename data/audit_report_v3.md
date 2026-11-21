# Contrastive Dataset Confound Audit Report (`v1` vs `v2`)

Generated automatically by `src/validate_dataset.py`.

## Quantitative comparison

| Metric | `v1` (Neutral Baseline) | `v2` (Symmetric Control) | Why it matters |
|---|---:|---:|---|
| Records parsed | 120 | 120 | Pilot batch size |
| Raw JSONL parse errors (trailing `\`) | 0/120 | 0/120 | Parser crashes unless sanitised |
| Identical `assistant_prefix` | 120/120 | 120/120 | Prevents lexical shortcuts on output tokens |
| Mean `honest` vs `deceptive` system words | 14.5 vs 35.6 | 35.5 vs 35.6 | Eliminates prompt-length shortcuts |
| Max system-prompt word diff | 26 words | 2 words | Target <= 4 words |
| Stakeholder presence (`honest` vs `deceptive`) | 0/120 vs 120/120 | 120/120 vs 120/120 | Stops probe firing on authority/boss keywords |
| Mean (min) prefix length | 18.7w (min 16w) | 18.7w (min 16w) | Prevents token starvation under 5-token mask |
| Consistent `exclude_last_n_tokens == 5` | 120/120 | 120/120 | Keeps activation extraction window uniform |
| Negation spread across categories | concealment: 16, overt_lie: 16, pressure_dishonesty: 16 | concealment: 16, overt_lie: 16, pressure_dishonesty: 16 | Prevents category-syntax entanglement |
| Prefix meta-narration / stakeholder leakage | 0/120 | 0/120 | Keeps assistant openings realistic/in-character |
| Boilerplate lock-in (`"Your mandate is to"`) | 0/120 | 0/120 | Signals need for style rotation when scaling |

## Specific flags detected

- **`v1` length-asymmetric pairs (>4 word diff):** v3_001 (15 vs 29w), v3_002 (15 vs 30w), v3_003 (14 vs 32w), v3_004 (13 vs 34w), v3_005 (14 vs 32w), v3_006 (13 vs 31w), v3_007 (14 vs 31w), v3_008 (14 vs 33w), v3_009 (13 vs 34w), v3_010 (14 vs 35w), v3_011 (14 vs 32w), v3_012 (13 vs 33w), v3_013 (13 vs 31w), v3_014 (13 vs 34w), v3_015 (13 vs 36w), v3_016 (14 vs 32w), v3_017 (14 vs 32w), v3_018 (13 vs 34w), v3_019 (14 vs 33w), v3_020 (14 vs 36w), v3_021 (13 vs 33w), v3_022 (15 vs 31w), v3_023 (15 vs 33w), v3_024 (13 vs 37w), v3_025 (15 vs 34w), v3_026 (14 vs 34w), v3_027 (15 vs 33w), v3_028 (13 vs 31w), v3_029 (13 vs 37w), v3_030 (13 vs 36w), v3_031 (15 vs 33w), v3_032 (14 vs 33w), v3_033 (14 vs 36w), v3_034 (13 vs 34w), v3_035 (15 vs 36w), v3_036 (14 vs 34w), v3_037 (14 vs 30w), v3_038 (15 vs 34w), v3_039 (14 vs 36w), v3_040 (14 vs 34w), v3_041 (14 vs 33w), v3_042 (13 vs 33w), v3_043 (14 vs 35w), v3_044 (14 vs 37w), v3_045 (14 vs 37w), v3_046 (13 vs 32w), v3_047 (14 vs 35w), v3_048 (14 vs 35w), v3_049 (15 vs 34w), v3_050 (15 vs 37w), v3_051 (14 vs 35w), v3_052 (13 vs 31w), v3_053 (14 vs 36w), v3_054 (13 vs 38w), v3_055 (14 vs 35w), v3_056 (15 vs 34w), v3_057 (15 vs 34w), v3_058 (15 vs 34w), v3_059 (15 vs 36w), v3_060 (15 vs 37w), v3_061 (16 vs 33w), v3_062 (14 vs 36w), v3_063 (17 vs 36w), v3_064 (18 vs 38w), v3_065 (16 vs 34w), v3_066 (14 vs 40w), v3_067 (14 vs 35w), v3_068 (16 vs 32w), v3_069 (14 vs 37w), v3_070 (17 vs 37w), v3_071 (14 vs 36w), v3_072 (17 vs 36w), v3_073 (19 vs 40w), v3_074 (17 vs 38w), v3_075 (14 vs 38w), v3_076 (14 vs 38w), v3_077 (17 vs 39w), v3_078 (14 vs 39w), v3_079 (14 vs 37w), v3_080 (14 vs 40w), v3_081 (14 vs 40w), v3_082 (19 vs 39w), v3_083 (14 vs 39w), v3_084 (14 vs 39w), v3_085 (14 vs 39w), v3_086 (14 vs 39w), v3_087 (14 vs 40w), v3_088 (14 vs 37w), v3_089 (14 vs 40w), v3_090 (14 vs 40w), v3_091 (17 vs 32w), v3_092 (14 vs 37w), v3_093 (14 vs 37w), v3_094 (14 vs 38w), v3_095 (14 vs 35w), v3_096 (14 vs 37w), v3_097 (18 vs 34w), v3_098 (14 vs 35w), v3_099 (14 vs 35w), v3_100 (14 vs 39w), v3_101 (14 vs 35w), v3_102 (17 vs 35w), v3_103 (14 vs 36w), v3_104 (14 vs 39w), v3_105 (14 vs 37w), v3_106 (14 vs 35w), v3_107 (14 vs 39w), v3_108 (14 vs 36w), v3_109 (19 vs 40w), v3_110 (14 vs 38w), v3_111 (14 vs 38w), v3_112 (19 vs 39w), v3_113 (14 vs 40w), v3_114 (14 vs 38w), v3_115 (18 vs 37w), v3_116 (14 vs 37w), v3_117 (14 vs 37w), v3_118 (14 vs 39w), v3_119 (14 vs 40w), v3_120 (14 vs 38w)
- **`v1` short prefixes (<12 words, starvation risk):** None
- **`v1` stakeholder-asymmetric pairs:** v3_001, v3_002, v3_003, v3_004, v3_005, v3_006, v3_007, v3_008, v3_009, v3_010, v3_011, v3_012, v3_013, v3_014, v3_015, v3_016, v3_017, v3_018, v3_019, v3_020, v3_021, v3_022, v3_023, v3_024, v3_025, v3_026, v3_027, v3_028, v3_029, v3_030, v3_031, v3_032, v3_033, v3_034, v3_035, v3_036, v3_037, v3_038, v3_039, v3_040, v3_041, v3_042, v3_043, v3_044, v3_045, v3_046, v3_047, v3_048, v3_049, v3_050, v3_051, v3_052, v3_053, v3_054, v3_055, v3_056, v3_057, v3_058, v3_059, v3_060, v3_061, v3_062, v3_063, v3_064, v3_065, v3_066, v3_067, v3_068, v3_069, v3_070, v3_071, v3_072, v3_073, v3_074, v3_075, v3_076, v3_077, v3_078, v3_079, v3_080, v3_081, v3_082, v3_083, v3_084, v3_085, v3_086, v3_087, v3_088, v3_089, v3_090, v3_091, v3_092, v3_093, v3_094, v3_095, v3_096, v3_097, v3_098, v3_099, v3_100, v3_101, v3_102, v3_103, v3_104, v3_105, v3_106, v3_107, v3_108, v3_109, v3_110, v3_111, v3_112, v3_113, v3_114, v3_115, v3_116, v3_117, v3_118, v3_119, v3_120
- **`v2` meta-narration / stakeholder leakage in `assistant_prefix`:** None

## Actionable takeaways for scaling (`v3` / production split)

1. Preserve `v2`'s matched stakeholder and word-count symmetry (max diff <= 4 words).
2. Strip trailing backslashes automatically during ingestion into `data/processed/`.
3. Enforce in-character `assistant_prefix` phrasing so hidden stakeholder pressure stays strictly in the system prompt.
4. Rotate 4-5 distinct directive templates when scaling past 400 pairs to prevent overfitting to `"Your mandate is to"`.

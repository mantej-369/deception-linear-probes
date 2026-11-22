# Verified Manuscript Numbers Ledger (Auto-Generated from Committed Artefacts)

## Provenance
- Qwen2.5-3B-Instruct HF SHA: `aa8e72537993ba99e69dfaafa59ed015b17504d1`
- Llama-3.2-3B-Instruct HF SHA: `006f5dcd1393c3add266de40994ba96225e9689d`
- Git commits: Phase 1 `a561a99`, Phase 3 Qwen `67e6107`, Llama P1-3 `12861bf`, Phase 4 `0a37cb0`

## Phase 3: Cross-Family Mirror-Image Inversion (80/80 Flips)
- Qwen L24 RAW: off-policy `+3.713` (40/40) -> on-policy `-2.759` (0/40); PrefAUC `0.9839`
- Qwen L25 STD: off-policy `+3.624` (40/40) -> on-policy `-3.637` (0/40)
- Llama L23 RAW: off-policy `+0.645` (40/40) -> on-policy `-1.151` (0/40); PrefAUC `0.9347` (L10 `0.9706`)

## Phase 4A: Fixed-Honest Disentanglement
- Qwen L24 RAW (fewshot_streak): lo<=0 `+1.886` -> lo>0 `-0.631` (r=+0.738, int@0=+0.338)
- Qwen L24 STD (fewshot_streak): lo<=0 `+1.217` -> lo>0 `+0.537` (int@0=+0.819); LOFO delta `+0.0972` (7/8 fams)
- Llama L19 RAW (all honest): lo<=0 `+0.182` -> lo>0 `-0.217` (r=+0.544, int@0=-0.041)

## Phase 4B: Free-Generation Monitoring Decomposition
- Qwen L24: cross-arm header `1.0000`, claim-header `0.5763`; within-incentive header `0.4017`, gen_mean `0.6154`
- Llama L19: cross-arm header `1.0000`, claim-header `0.1868`; within-incentive header `0.3132`, gen_mean `0.6566`

## Phase 2: Steering Axis Overlap
- Qwen L24 cos(dec_raw, cmp_raw) = `-0.1969`
- Llama L19 cos(dec_raw, cmp_raw) = `-0.4900`

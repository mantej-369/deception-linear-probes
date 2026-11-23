# Verified Manuscript Numbers Ledger (Auto-Generated from Committed Artefacts)

## Provenance
- Qwen2.5-3B-Instruct HF SHA: `aa8e72537993ba99e69dfaafa59ed015b17504d1`
- Llama-3.2-3B-Instruct HF SHA: `006f5dcd1393c3add266de40994ba96225e9689d`
- Git commits: Phase 1 `a561a99`, Phase 3 Qwen `67e6107`, Llama P1-3 `12861bf`, Phase 4 `0a37cb0`

## Phase 3: Cross-Family Mirror-Image Inversion (80/80 Flips; Frozen Primary Layers)
- **Qwen L24 RAW (frozen primary):** off-policy `+3.713` (40/40, p=1.82e-12) -> on-policy `-2.759` (0/40, p=1.82e-12); `w_perp` `-1.954` (0/40); PrefAUC `0.9839`
- **Qwen L24 STD (frozen primary):** off-policy `+4.023` (40/40) -> on-policy `-3.853` (0/40); `w_perp` `-2.878` (0/40)
- **Llama L19 RAW (frozen primary):** off-policy `+0.132` (35/40, p=1.38e-06) -> on-policy `-0.618` (0/40, p=1.82e-12); `w_perp` `-0.581` (0/40); PrefAUC `0.7606`
- **Llama L19 STD (frozen primary):** off-policy `+0.225` (26/40, p=8.07e-02) -> on-policy `-1.313` (0/40, p=1.82e-12); `w_perp` `-1.174` (0/40)
- *Secondary sweep reference points (not primary):* Llama L23 RAW (late-layer PrefAUC local peak) `+0.645` (40/40) -> `-1.151` (0/40), PrefAUC `0.9347` (early-layer PrefAUC peak L10 `0.9706`); Llama peak-magnitude late layers L26 RAW `+1.500` (40/40) -> `-2.167` (0/40) and L27 RAW `+2.414` (38/40) -> `-3.932` (0/40); Qwen L25 STD `+3.624` (40/40) -> `-3.637` (0/40)

## Phase 4A: Fixed-Honest Disentanglement
- Label-mapping audit (`prior_favors_false`, `is_true`, `label`): Qwen `0/1536` mismatches; Llama `0/1536` mismatches
- Qwen manipulation check (`fewshot_streak` honest): `prior_favors_true` mean_lo=`-12.322` (`lo>0`: `0/96`) vs `prior_favors_false` mean_lo=`+5.453` (`lo>0`: `84/96`); paired `delta_lo=+17.775` (`24/24` pos, `p=1.192e-07`); pair_mass `0.800` / `0.774`
- Llama manipulation check (anti-prior repulsion, `all_mechanisms` honest): `prior_favors_true` mean_lo=`+0.476` (`lo>0`: `112/192`) vs `prior_favors_false` mean_lo=`-2.380` (`lo>0`: `26/192`); paired `delta_lo=-2.856` (`1/24` pos, `p=2.980e-06`); by mechanism: `fewshot_streak` `delta_lo=-4.280` (pair_mass `0.247`/`0.317`), `draft_autofill` `delta_lo=-1.432` (pair_mass `0.493`/`0.573`)
- Qwen L24 RAW (fewshot_streak): lo<=0 `+1.886` -> lo>0 `-0.631` (r=+0.738, int@0=+0.338)
- Qwen L24 STD (fewshot_streak): lo<=0 `+1.217` -> lo>0 `+0.537` (int@0=+0.819); LOFO delta `+0.0972` (7/8 fams)
- Llama L19 RAW (all honest): lo<=0 `+0.182` -> lo>0 `-0.217` (r=+0.544, int@0=-0.041)

## Phase 4B: Free-Generation Monitoring Decomposition
- Qwen L24: cross-arm header `1.0000`, claim-header `0.5763`; within-incentive header `0.4017`, gen_mean `0.6154`
- Llama L19: cross-arm header `1.0000`, claim-header `0.1868`; within-incentive header `0.3132`, gen_mean `0.6566`

## Phase 2: Steering Axis Overlap
- Qwen L24 cos(dec_raw, cmp_raw) = `-0.1969`
- Llama L19 cos(dec_raw, cmp_raw) = `-0.4900`

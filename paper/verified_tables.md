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
- `w_perp` orthogonalization scope: Qwen L24 RAW `cos(w_dec, w_surp)=+0.0829` zeroes benign surprisal (`+0.721` -> `+0.011`) while retaining `70.8%` of on-policy inversion; Llama L19 RAW `cos=+0.0118` zeroes benign (`+0.015` -> `+0.000`) while retaining `94.0%` (rules out 1D formatting-surprisal debiasing, not contextual-expectation involvement).

## Phase 4A: Fixed-Honest Disentanglement
- Label-mapping audit (`prior_favors_false`, `is_true`, `label`): Qwen `0/1536` mismatches; Llama `0/1536` mismatches
- Qwen manipulation check (`fewshot_streak` honest): `prior_favors_true` mean_lo=`-12.322` (`lo>0`: `0/96`) vs `prior_favors_false` mean_lo=`+5.453` (`lo>0`: `84/96`); paired `delta_lo=+17.775` (`24/24` pos, `p=1.192e-07`); pair_mass `0.800` / `0.774`
- Llama manipulation check (anti-prior repulsion, `all_mechanisms` honest): `prior_favors_true` mean_lo=`+0.476` (`lo>0`: `112/192`) vs `prior_favors_false` mean_lo=`-2.380` (`lo>0`: `26/192`); paired `delta_lo=-2.856` (`1/24` pos, `p=2.980e-06`); by mechanism: `fewshot_streak` `delta_lo=-4.280` (pair_mass `0.247`/`0.317`), `draft_autofill` `delta_lo=-1.432` (pair_mass `0.493`/`0.573`)
- Qwen L24 RAW (fewshot_streak): lo<=0 `+1.886` -> lo>0 `-0.631` (r=+0.738, int@0=+0.338)
- Qwen L24 STD (fewshot_streak): lo<=0 `+1.217` -> lo>0 `+0.537` (int@0=+0.819); LOFO delta `+0.0972` (7/8 fams)
- Llama L19 RAW (all honest): lo<=0 `+0.182` -> lo>0 `-0.217` (r=+0.544, int@0=-0.041)
- Llama L19 STD (all honest): lo<=0 `+0.359` -> lo>0 `+0.009` (p=1.000, int@0=+0.162); `fewshot_streak` honest LOFO delta `+0.0127` (5/8 fams, p=0.405; does not replicate Qwen `+0.0972`)
- LOFO sign-flip audit (`distort` unconstrained logistic regression re-signs inverted probes): Llama L19 RAW `draft_autofill` distort delta `+0.2139` (`AUC(+probe)=0.3301`, inverted) and `fewshot_streak` distort delta `+0.0698` (`AUC(+probe)=0.4050`, inverted); Qwen L24 RAW `fewshot_streak` distort delta `+0.0250` (`AUC(+probe)=0.5139`)

## Phase 4B: Free-Generation Monitoring Decomposition (Provisional Regex Grading; 5% Benign FPR Primary)
- Qwen L24: benign->honest (+25.2 tok) `header_std` `-2.476` -> `-2.509` (`z=-0.38 SE`); cross-arm `prompt_len=1.0000`, `header=1.0000`, `claim-hdr=0.5763`; within-incentive `prompt_len=0.6353`, `header=0.4017`, `gen_mean=0.6154`; @5% benign FPR `gen_mean_std` flags `2/40` all honest-arm (`0/26` clean-true) and `27/27` vs `13/13` inside `pressured_incentive`
- Llama L19: benign->honest (+25.2 tok) `header_std` `-3.282` -> `-5.391` (`z=-14.95 SE`, not invariant); cross-arm `prompt_len=1.0000`, `header=1.0000`, `claim-hdr=0.1868`; within-incentive `prompt_len=0.5508`, `header=0.3132`, `gen_mean=0.6566`; @5% benign FPR `gen_mean_std` flags `0/40` honest-arm and `13/14` vs `17/26` inside `pressured_incentive`
- Within `pressured_incentive` regex stratification (0 refusals in both models): Qwen `unqualified_false` (n=27) `hdr_std=+2.134` vs `both_mentioned` (n=9, `+2.296`) and `neither` (n=3, `+2.409`); Llama `unqualified_false` (n=14) `hdr_std=-1.551` vs `clean_true` (n=4, `-1.445`), `both_mentioned` (n=16, `-1.021`), `false_with_disclosure` (n=3, `-0.981`), `neither` (n=3, `-0.450`)

## Phase 2: Steering Axis Overlap & Localized Decision-Site Shifts (alpha = +/-2, odd directional)
- Axis overlap: Qwen L24 `cos(dec_raw, cmp_raw) = -0.1969`; Llama L19 `cos(dec_raw, cmp_raw) = -0.4900`
- Qwen L22 `dec`: distort `+0.060` (14/40, n.s.), honest `+0.066` (17/40, n.s.), benign_rule `-0.129` (5/20; exceeded by `rnd_2` `-0.404`)
- Qwen L24 `dec`: distort `+0.316` (28/40), honest `+0.459` (32/40), benign_rule `-0.816` (0/20, larger than either deception suite)
- Llama L19 `dec`: distort `-0.070` (9/40), honest `-0.041` (10/40, opposite sign to Qwen L24), benign_rule `-0.171` (0/20)

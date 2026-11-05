# Research log

A dated summary of completed work, transcribed from the project notebook (dates in Almaty time, UTC+5). The repository itself was created on 31 October 2026. Original result files with their checksums and modification times (UTC) are in [`results/`](results/) and [`results/MANIFEST.txt`](results/MANIFEST.txt). Predictions and decision rules are in [`PREDICTIONS_AND_RULES.md`](PREDICTIONS_AND_RULES.md).

| Date | Work | Result |
|---|---|---|
| Aug 2026 | Original idea: grouping vs. attribute probes in DINO | Huang et al. (ICML 2026) already probed binding → focus moved to *where and why* binding is lost |
| 29 Sep | Scene generator v1 (`generate.py`) | 200 swap pairs, pixel checks pass |
| 4 Oct | DINO activations; pipeline check (`probe_sanity.py`); first look at S vs. S′ (`swap_look.py`) | Color decoded 100%; shape 38% at L00 → ~98% by L04. Untouched objects change between S and S′ in deep layers, but object area is a confound → generator v2 needed |
| 5 Oct | Generator v2 (`generate_v2.py`): twins, S / Sp / Sr | 2,000 scenes. Its built-in check caught a bug (square corners overlapping), which was fixed before use |
| 5 Oct | Swap vs. recolor (`swap_v2.py`) and null model: 3 random networks (`null_compare.py`) | Swap/recolor ratio at L12: CLS 0.17, untouched 0.19, far background 0.49; random networks ≈ 0.07 → the effect is **learned** |
| 5 Oct | Binding probe (`probe_v2.py`) | "Left" twin: CLS peak 78.2% (L03) → 63.5% (L12); random 49–52% → binding is **readable** |
| 5 Oct | Robustness: "upper" twin; MLP probe | "Upper" stays at 83.9% at L12 → only the horizontal question declines. MLP probe weaker than linear: uninformative, kept as a negative result |
| 5 Oct | Mirror test on DINO (`flip_test.py`), prediction written before the run | Left–right / up–down sensitivity 0.34 (CLS) / 0.28 (patch mean) at L12, starting at L04 — same layers where "left" declines |
| 6 Oct | What the probe depends on (`probe_breakdown.py`), 3 predictions written before the run | All three confirmed; even twins ≥ 96 px apart lose "left" (92.3 → 72.1%) but keep "upper" (94.9%) |
| 6 Oct | Literature check | No prior work found linking mirror invariance to binding layer by layer; Liu et al. (2026) state the general idea that flip invariance removes left/right information |
| 6 Oct | CLIP (trained without flips): weight check, activations, 3 random CLIPs, mirror test, both probes | Mirror 0.65 / 0.61 (between DINO and random); gap "upper − left" at L12: CLS 9.3 vs. DINO 20.4, far background 9.0 vs. 18.6; all three written predictions confirmed |
| 6 Oct | "Folding" experiment (`probe_fold.py`), rule written before the run | DINO folds: loses the side but keeps the distance from the midline (D = 19.5 [16.5–22.6]); CLIP does not (4.9 [1.7–8.2]). Post-hoc analyses marked as such |
| 7 Oct | Figure 1 (`figure1.py`); first own predictions for MAE / I-JEPA (10:06) | See `PREDICTIONS_AND_RULES.md`, #33 |
| 13–15 Oct | One-page summary of results | — |
| 15 Oct | MAE / I-JEPA mirror scripts and decision rule #34; prediction "B" (20:01) | — |
| 15 Oct | Mirror test on MAE (ViT-H/14) | 0.67 (CLS) / 0.63 (patch mean) at L32 — prediction #33a hit |
| 17 Oct | Mirror test on I-JEPA (official checkpoint, target encoder) | 0.40 (patch mean) — more mirror-blind than MAE; prediction #33b missed |
| 18 Oct | MAE vs. I-JEPA by rule #34 (`flip_compare.py`) | Δ = −0.222 [−0.304, −0.148] → "against A": flip augmentation is not needed for mirror blindness; prediction "B" missed |
| 21 Oct | Decision: full scope (causal interventions, behavior, attention heads) before submission | — |
| 22 Oct | Binding probe for ViT-H (`probe_big.py`) and rule #39; prediction (19:47) | Group means cached on 9 layers instead of ~90 GB of activations per model |
| 23 Oct | Binding probe on MAE | Gap "upper − left", far background, L32: 10.1 [8.5–11.6] pp (CLS 6.8) — close to CLIP at similar mirror sensitivity; prediction #39a missed |
| 31 Oct | Repository created | Code, README, results, this log |
| 5 Nov | I-JEPA binding probe finished (4th network; the run of 22–23 Oct had stopped after 3) and MAE vs. I-JEPA by rule #39 | Gap "upper − left", far background, L32: I-JEPA 11.4 [9.9–13.0] pp vs. MAE 10.1 [8.5–11.6]; Δ = 1.3 [−0.6, 3.2] → "does not discriminate". Predictions #39b and #39c missed (score 1 of 6). Controls pass: random networks ≤ 1.7 pp, twin control 100% |

**Next:** causal interventions on internal representations; behavior on CLIP zero-shot; other models (DINOv2, supervised ViT).

# Results

Original outputs of the scripts, copied byte-for-byte from the machine where they were run (see `MANIFEST.txt` for checksums and times). Logs are mostly in Russian; the key numbers are summarized in the main README.

| Experiment | Files |
|---|---|
| Pipeline check, first look (v1) | `extract_log.txt`, `probe_log.txt`, `swap_log.txt`, `swap_look.csv` |
| Activations v2 (DINO, random DINO, CLIP, random CLIP) | `extract_*_log.txt`, `clip_check_log.txt` |
| Swap vs. recolor; null model | `swap_v2*.csv`, `swap_v2*_log.txt`, `null_compare.csv`, `null_compare_log.txt` |
| Binding probe "left" / "upper" / MLP (DINO, CLIP) | `probe_v2*.csv`, `probe_v2*_log.txt` |
| What the probe depends on | `probe_breakdown.csv`, `probe_breakdown_log.txt` |
| Mirror test (DINO, CLIP, MAE, I-JEPA) | `flip_test*.csv`, `flip_test*_log.txt` |
| MAE vs. I-JEPA, mirror test | `flip_compare_log.txt` |
| "Folding" (DINO, CLIP) and post-hoc analyses | `probe_fold*.csv`, `probe_fold*_log.txt` |
| Binding probe on MAE and I-JEPA (ViT-H/14) | `probe_big_mae.csv`, `probe_big_mae_log.txt`, `probe_big_ijepa.csv`, `probe_big_ijepa_log.txt` |
| MAE vs. I-JEPA, binding probe | `probe_big_compare_log.txt` |
| Mean-ablation of attention / MLP, DINO blocks 4–12 (OSF us5zv) | `ablate_blocks.csv`, `ablate_blocks_log.txt`, `ablate_blocks.png` |
| Mean-ablation of attention / MLP, CLIP blocks 7–12 (OSF 6uves) | `ablate_clip.csv`, `ablate_clip_log.txt`, `ablate_clip.png` |
| Twin-token patching across DINO layers (OSF 23wjy) | `patch_twins.csv`, `patch_twins_log.txt`, `patch_twins.png` |
| Distributed alignment search (DAS), DINO (OSF j2g74) | `das_binding.csv`, `das_binding_log.txt`, `das_binding.png` |
| Behavioral test of CLIP itself: caption choice, left–right vs. top–bottom (OSF 5hn9u) | `behavior_clip.csv`, `behavior_clip_log.txt` |
| Probe robustness: regularization grid and random-side control (OSF 3f9tz) | `probe_robust.csv`, `probe_robust_log.txt` |
| Two more networks: DINOv2 ViT-B/14 with registers and DeiT III ViT-B/16, mirror test and binding probe; rank test across six networks (OSF 8s4gm) | `more_nets_dinov2.csv`, `more_nets_dinov2_log.txt`, `more_nets_deit3.csv`, `more_nets_deit3_log.txt`, `more_nets_compare_log.txt` |
| Mirror decomposition (symmetric / antisymmetric parts) and MLP probe, DINO, CLIP, DINOv2, DeiT III (OSF npvs5). In the `D_*` rows of the CSVs, `n_scenes` reads 627 but D used all 2,000 scenes | `mirror_decomp_dino.csv`, `mirror_decomp_clip.csv`, `mirror_decomp_dinov2.csv`, `mirror_decomp_deit3.csv`, `mirror_decomp_*_log.txt`, `mirror_decomp_summary_log.txt` |
| 24 pretrained ViTs: mirror test, binding gap, mirror decomposition; rank tests B1–B3 (OSF xyedr); exploratory analysis of B2 after the run | `many_nets.csv`, `many_nets_rules.csv`, `many_nets_part1_log.txt`, `many_nets_part2_log.txt`, `many_nets_part3_log.txt`, `many_nets_summary_log.txt`, `many_nets_explore_log.txt` |
| Real photographs: mirror test of 24 ViTs on aerial images and COCO (C1), binding with photographs as background (C2) (OSF dprnk); two exploratory analyses after the run. The selected photographs are listed in `../photos_lists.json` | `photos_prepare_log.txt`, `photos_c1.csv`, `photos_c1_part1_log.txt`, `photos_c1_part2_log.txt`, `photos_c1_part3_log.txt`, `photos_c2.csv`, `photos_c2_log.txt`, `photos_rules.csv`, `photos_summary_log.txt`, `photos_explore_log.txt`, `photos_explore_c1_log.txt` |
| Attention heads: mirror score of each head, ablation of the top 12 against random sets, single-head ablations in DINO (OSF gdnvb); check 1 in float64 and the exploratory split of the score, after the run | `heads_dino_log.txt`, `heads_deit3_log.txt`, `heads_clip_log.txt`, `heads_m.csv`, `heads_f2.csv`, `heads_f3.csv`, `heads_f3_log.txt`, `heads_rules.csv`, `heads_summary_log.txt`, `heads_check1_log.txt`, `heads_explore.csv`, `heads_explore_log.txt` |
| Exploratory run after osf.io/gdnvb: DINO row-stripe heads ablated alone and together | `heads_stripes.csv`, `heads_stripes_log.txt` |
| A mirror pair of heads: attention-map measures (G1, G3) and resample-ablation on new scenes (G2) (OSF 8nxg7); exploratory non-additivity after the run (the first attempt stopped before any computation; its log is kept) | `heads_pair_maps_log.txt`, `heads_pair_heads.csv`, `heads_pair_pairs.csv`, `heads_pair_rules_maps.csv`, `heads_pair_g2_log.txt`, `heads_pair_g2.csv`, `heads_pair_rules_g2.csv`, `heads_pair_explore_log_FAILED_1.txt`, `heads_pair_explore_log.txt`, `heads_pair_explore.csv` |
| Breaking mirror symmetry in all pairs: single heads, pairs, one head of every pair and all pair heads in DeiT III against random sets; a second DINO pair (OSF 7tsgq) | `heads_pair_plus_log.txt`, `heads_pair_plus.csv`, `heads_pair_plus_rules.csv` |
| In place or permutation: equivariance of patch tokens under left–right and up–down mirroring, 24 pretrained ViTs and 7 random networks (OSF syevu); check 2 rerun under torch.no_grad after the run | `equiv_nets.csv`, `equiv_nets_rules.csv`, `equiv_nets_part1_log.txt`, `equiv_nets_part2_log.txt`, `equiv_nets_part3_log.txt`, `equiv_nets_summary_log.txt`, `equiv_check2_fix_log.txt` |
| Training from scratch, before the registration: check that the scene shapes are mirror-symmetric, speed benchmark and two diagnostics on the Mac (not results) | `sprite_symmetry_check_log.txt`, `e_bench_log.txt`, `e_diag_log.txt`, `e_diag2_log.txt` |
| Training from scratch, pilots on Food-101 (ViT-Tiny, one seed per condition, Mac; conditions 1–5: no flips, horizontal flips, vertical flips, collage labelled by the left image, collage labelled by both images; disclosed in osf.io/ducms) | `e_train_c*_s*_log.txt`, `e_measure_c*_s*.csv`, `e_measure_c*_s*_log.txt` |
| Training from scratch on ImageNet-100 (OSF ducms): training logs from Kaggle (conditions 1–3, seeds 0–3), measures of every checkpoint, the rules H1, H2 and H4 | `e_full_c[1-3]_s*_log.txt`, `e_measure_full_c*_s*.csv`, `e_measure_full_c*_s*_log.txt`, `run_measure_ducms_log.txt`, `e_full_rules_ducms_log.txt` |
| Ordered vs. unordered image pairs (OSF pxaz4): training logs of seed 0 (gate); no probe has been computed yet | `e_full_c6_s0_log.txt`, `e_full_c7_s0_log.txt` |
| DAS: runs before the registration, disclosed in it (smoke test on 200 scenes, speed benchmark, pilots 1 and 2); not results | `das_binding_test200_log.txt`, `das_bench_log.txt`, `das_pilot_log.txt`, `das_pilot2_log.txt` |

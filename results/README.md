# Results

Original outputs of the scripts, copied byte-for-byte from the machine where they were run (see `MANIFEST.txt` for checksums and times). Logs are mostly in Russian; the key numbers are summarized in the main README.

| Experiment | Files |
|---|---|
| Pipeline check, first look (v1) | `extract_log.txt`, `probe_log.txt`, `swap_log.txt`, `swap_look.csv` |
| Activations v2 (DINO, random DINO, CLIP, random CLIP) | `extract_*_log.txt`, `clip_check_log.txt` |
| Swap vs. recolor; null model | `swap_v2*.csv`, `swap_v2*_log.txt`, `null_compare.*` |
| Binding probe "left" / "upper" / MLP (DINO, CLIP) | `probe_v2*.csv`, `probe_v2*_log.txt` |
| What the probe depends on | `probe_breakdown.*` |
| Mirror test (DINO, CLIP, MAE, I-JEPA) | `flip_test*.csv`, `flip_test*_log.txt` |
| MAE vs. I-JEPA, mirror test | `flip_compare_log.txt` |
| "Folding" (DINO, CLIP) and post-hoc analyses | `probe_fold*.csv`, `probe_fold*_log.txt` |
| Binding probe on MAE and I-JEPA (ViT-H/14) | `probe_big_mae.*`, `probe_big_ijepa.*` |
| MAE vs. I-JEPA, binding probe | `probe_big_compare_log.txt` |
| Mean-ablation of attention / MLP, DINO blocks 4–12 (OSF us5zv) | `ablate_blocks.csv`, `ablate_blocks_log.txt`, `ablate_blocks.png` |
| Mean-ablation of attention / MLP, CLIP blocks 7–12 (OSF 6uves) | `ablate_clip.csv`, `ablate_clip_log.txt`, `ablate_clip.png` |
| Twin-token patching across DINO layers (OSF 23wjy) | `patch_twins.csv`, `patch_twins_log.txt`, `patch_twins.png` |
| Distributed alignment search (DAS), DINO (OSF j2g74) | `das_binding.csv`, `das_binding_log.txt`, `das_binding.png` |
| Behavioral test of CLIP itself: caption choice, left–right vs. top–bottom (OSF 5hn9u) | `behavior_clip.csv`, `behavior_clip_log.txt` |
| Probe robustness: regularization grid and random-side control (OSF 3f9tz) | `probe_robust.csv`, `probe_robust_log.txt` |
| Two more networks: DINOv2 ViT-B/14 with registers and DeiT III ViT-B/16, mirror test and binding probe; rank test across six networks (OSF 8s4gm) | `more_nets_dinov2.csv`, `more_nets_dinov2_log.txt`, `more_nets_deit3.csv`, `more_nets_deit3_log.txt`, `more_nets_compare_log.txt` |
| Mirror decomposition (symmetric / antisymmetric parts) and MLP probe, DINO, CLIP, DINOv2, DeiT III (OSF npvs5). In the `D_*` rows of the CSVs, `n_scenes` reads 627 but D used all 2,000 scenes | `mirror_decomp_dino.csv`, `mirror_decomp_clip.csv`, `mirror_decomp_dinov2.csv`, `mirror_decomp_deit3.csv`, `mirror_decomp_*_log.txt`, `mirror_decomp_summary_log.txt` |
| DAS: runs before the registration, disclosed in it (smoke test on 200 scenes, speed benchmark, pilots 1 and 2); not results | `das_binding_test200_log.txt`, `das_bench_log.txt`, `das_pilot_log.txt`, `das_pilot2_log.txt` |

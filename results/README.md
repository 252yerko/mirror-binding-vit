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
| Binding probe on MAE (ViT-H/14) | `probe_big_mae.*` |

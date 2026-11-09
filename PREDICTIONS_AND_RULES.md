# Predictions and decision rules

For each experiment, the decision rule (and, from 7 October on, my own prediction) was written down in the project notebook **before the run**. This file is an English transcription of those entries. Numbers such as **#33** are entry numbers in the notebook's registry; script headers cite them as «реестр N».

**Limits.** These records are dated by my notebook, not by an external timestamp. The repository was created on 31 October 2026, so commit dates are later than the entries. Checksums and file times of the original result files are in [`results/MANIFEST.txt`](results/MANIFEST.txt). Since 6 November 2026, the design, rule and prediction are also registered on OSF before the run, which gives an external timestamp (#43: [osf.io/us5zv](https://osf.io/us5zv); #53: [osf.io/6uves](https://osf.io/6uves); #59: [osf.io/23wjy](https://osf.io/23wjy); under embargo until the preprint, at the latest 31 March 2027).

---

## 1. My predictions (scored by a rule fixed before the run)

**Score so far: 5 hits out of 14 resolved; none pending.**

| # | Written (Almaty, UTC+5) | Prediction | Scoring rule | Result | Hit? |
|---|---|---|---|---|---|
| 33a | 7 Oct, 10:06 | MAE mirror sensitivity (CLS, last layer L32): **0.60–0.75**, 70% confident | point value rounded to 0.01 inside the range | 0.67 [0.62–0.72] | ✅ |
| 33b | 7 Oct, 10:06 | I-JEPA mirror sensitivity, last layer: **0.50–0.65**, 85% confident | same; patch mean (I-JEPA has no CLS; specified on 15 Oct, before the run, #34) | 0.40 [0.37–0.44] | ❌ |
| 35 | 15 Oct, 20:01 | Verdict of the MAE vs. I-JEPA comparison: **"B"** | rule #34 (below) | "against A" | ❌ |
| 39a | 22 Oct, 19:47 | Binding gap "upper − left" for MAE (far background, L32): **8.8–9.0 pp** | point value rounded to 0.1 inside the range | 10.1 [8.5–11.6] | ❌ |
| 39b | 22 Oct, 19:47 | Same gap for I-JEPA: **"about 17" pp** (scored as 16.0–18.0) | same | 11.4 [9.9–13.0] | ❌ |
| 39c | 22 Oct, 19:47 | Verdict: **"follows"** (more mirror-blind model loses more left–right binding) | rule #39 (below) | "does not discriminate" (Δ = 1.3 [−0.6, 3.2]) | ❌ |
| 43a | 5 Nov, 23:55 | Verdict of the mean-ablation test: **"B"** (the MLP erases, attention is not involved), 90% confident | rule #43 (below) | "C" (both erase) | ❌ |
| 43b | 5 Nov, 23:55 | Δ"left" with attention ablated in blocks 4–12 (far background, L12): **below −5.0 pp** | point value rounded to 0.1, ≤ −5.1 | +4.5 [+2.7, +6.3] | ❌ |
| 43c | 5 Nov, 23:55 | Δ"left" with the MLP ablated in blocks 4–12: **above +8.0 pp** | point value rounded to 0.1, ≥ +8.1 | +8.5 [+6.9, +10.1] | ✅ |
| 53a | 9 Nov, 09:45 | CLIP, blocks 7–12: verdict **"C"** (both erase), 45% confident | rule #53 = rule #43 | "A" (attention only) | ❌ |
| 53b | 9 Nov, 09:45 | CLIP: Δ"left" with attention ablated: **about +3** (+2.1 to +5.0) | point value rounded to 0.1 | +3.2 [+1.5, +4.9] | ✅ |
| 53c | 9 Nov, 09:45 | CLIP: Δ"left" with the MLP ablated: **about +6** (+5.1 to +8.0) | point value rounded to 0.1 | +10.0 [+8.3, +11.7] | ❌ |
| 59a | 9 Nov, 11:23 | DINO, twin-token patching: the first layer where T < 50% is **about L07**, 30% confident | rule #59 (below); hit if L06–L08 | L06 (T = 43.8% [41.1, 46.7]) | ✅ |
| 59b | 9 Nov, 11:23 | DINO, twin-token patching: T at L11 **about 15%** | point value rounded to 0.1, 10.0 to 20.0 | 11.9% [10.2, 13.7] | ✅ |

Original wording of #33 (Russian, verbatim): «МАЕ: как 0,6-0,75. я считаю так потому что она закрывает 75% кусочков картинки и есть должна дорисовать пиксель закрытых кусочков. я уверен на 70%. I-JEPA: будет примерно 0.5-0.65. тоесть ближе к CLIP чем к случайной сети. потому что он часто смотрит стороны как кошка как. тоесть анализирует стороны и смотрит. уверен я на 85%».
*Translation:* "MAE: 0.6–0.75, because it masks 75% of the patches and has to redraw the pixels of the hidden ones; 70% sure. I-JEPA: about 0.5–0.65, closer to CLIP than to a random network, because it looks at the sides and analyses them; 85% sure."

Original wording of #39 (verbatim): «шаг 1)б , шаг 2)примерный разрыв МАЕ в районе 8.8-9.0, I -JEPA ближе к ДИНО тоесть примерно 17 разрыв. шаг 3)да понял. вывод следует тому что чем ниже слепота тем выше разрыв».

Original wording of #43 (verbatim, 23:53 and 23:55): «мне кажется если заткнуть разговоры то просядет все, так как они не будут понимать друг друга еще больше и просядет ниже 60%, а если вырубить MLP то мне кажется увеличится наоборот процент левого. и вопрос про MLP думаю что да так как каждый отт себя что то добавляет и потом суть теряется.» → «3)Е , 4)90%. я уже все проанализировал и примерно уверен на 100%. такак реально это же очевидно что MLP виновник так как это портит все и теряется качество понимаение».
*Translation:* "I think if you silence the 'conversations' [attention], everything will drop, because they will understand each other even less, and it will drop below 60%; and if you switch off the MLP, I think the percentage for 'left' will instead go up. And on the MLP question, I think yes, because each one adds something of its own and then the meaning gets lost." → "3) E [Δ'left' with the MLP ablated: above +8.0], 4) 90%. I have already analysed everything and am about 100% sure, because it is really obvious that the MLP is the culprit, since it spoils everything and the quality of understanding is lost." The prediction was given after the design had been explained to me; an earlier answer the same evening was withdrawn as a guess and is not scored. The same entry is in the OSF registration.

Original wording of #53 (verbatim, 09:45): «* Вывод. Скорее всего В, но слабее, чем у DINO. Уверен процентов на 45, потому что у CLIP «верхний» после L07 тоже падает. Это легко даёт «не различает» вместо «стирает». * Внимание блоков 7–12. Около +3 (вариант Г). * MLP блоков 7–12. Около +6 (вариант Д). * Почему. У DINO MLP возвращал «левый» сильнее, чем внимание. У CLIP запас всего около 9 п.п., и он делится между двумя частями. * Что убедило бы меня в ошибке. Если внимание даст больше MLP, или обе Δ окажутся около нуля (тогда Г: ни одна часть не причастна).  я уверен примерно на 75%». Clarification (09:47, before the code): the "75%" at the end was a typo for 45%.
*Translation:* "Verdict: most likely C, but weaker than in DINO; about 45% sure, because in CLIP 'upper' also falls after layer 7, which easily gives 'does not discriminate' instead of 'erases'. Attention, blocks 7–12: about +3 (option D). MLP, blocks 7–12: about +6 (option E). Why: in DINO the MLP restored 'left' more strongly than attention; CLIP has only about 9 pp to recover, split between the two parts. What would show me wrong: attention giving more than the MLP, or both Δ near zero." The layer-wise CLIP numbers were shown to me before the prediction (the layer-12 gap was shown as 9.1 instead of 9.0 pp, a rounding slip). The same entry is in the OSF registration.

Original wording of #59 (verbatim, 11:23): «* Слой. T впервые ниже 50% около L07. Уверенность около 30%, потому что дальний фон читает привязку через внимание за несколько блоков, и к середине сети часть уже должна быть скопирована. * T на L11. Около 15%. Один последний блок ещё может дотянуться до близнецов. * Что убедило бы меня в ошибке. Если T останется выше 50% до L11, значит, фон всё время перечитывает привязку с близнецов, а не хранит копию.»
*Translation:* "Layer: T first falls below 50% around L07; about 30% confident, because the far background reads the binding through attention over several blocks, so by the middle of the network part of it should already have been copied. T at L11: about 15%; one last block can still reach the twins. What would show me wrong: T staying above 50% up to L11 — then the background keeps re-reading the binding from the twins instead of holding a copy." Before the prediction I was shown the design and two earlier numbers: "left" on the far background peaks at L03, mirror blindness starts at L04. The same entry is in the OSF registration.

An earlier table of informal expectations (5–6 October) is **not** counted: the time it was written could not be verified.

---

## 2. Decision rules, by experiment

**Mirror test on DINO** (`flip_test.py`; written 5 Oct, before the run). Prediction: in deep layers, d(left–right flip) / d(up–down flip) < 1 and falling with depth for DINO; no asymmetry in random networks. If DINO ≈ 1 or > 1, the "mirror" hypothesis is refuted. Caveat written at the time: this tests a consequence, not the cause (flip augmentation vs. statistics of photos).
→ Confirmed: CLS 0.34 [0.32–0.35] at L12; random 0.94–1.07.

**What the binding probe depends on** (`probe_breakdown.py`; before the run). P1: more objects → lower accuracy. P2: larger distance between twins along the query axis → higher accuracy. P3: in L08–L12 a large distance helps "left" only weakly but "upper" strongly. If "left" at ≥ 96 px were as high as "upper", the mirror hypothesis would need revision.
→ P1 ✅ (89.3 → 82.8 → 79.5%), P2 ✅, P3 ✅ (≥ 96 px, L12: left 72.1 vs upper 94.9).

**CLIP as a model trained without flips** (6 Oct, before the run). "Augmentation" → CLIP mirror sensitivity ≈ 1 (like random); "data only" → as low as DINO; in between → both factors.
→ In between: 0.65 [0.62–0.69].
Then, after the mirror test and before the probes: (P-a) for CLIP "left" < "upper" at L12; (P-b) the gap is smaller than DINO's; (P-c, weaker) on the far background the gap opens later than in DINO.
→ P-a ✅, P-b ✅ (CLS 9.3 vs 20.4; far background 9.0 vs 18.6), P-c ✅ on far background (L07 vs L04), less clean on CLS.

**"Folding"** (`probe_fold.py`; 6 Oct, before the run, for DINO and CLIP). Four probe questions: left, upper, closer to the vertical midline, closer to the horizontal midline. Side gap = upper − left; midline gap = (closer, vertical) − (closer, horizontal); D = side gap − midline gap, at L12.
- *folding*: lower 95% bound of D > 0 and midline gap ≤ ½ side gap;
- *erasure*: CI of D includes 0;
- *intermediate*: D > 0 but midline gap > ½ side gap;
- *no answer*: if the side gap is absent or "closer, vertical" is not above all random networks.
→ DINO: folding (CLS D = 19.5 [16.5–22.6]; far background 16.0). CLIP: intermediate on CLS (4.9 [1.7–8.2]), erasure on far background.
*Not pre-specified:* `probe_fold_strata.py` and `probe_fold_compare.py` are analyses added after seeing the results, and are reported as such.

**MAE vs. I-JEPA, mirror test** (`flip_compare.py`; rule #34, written 15 Oct, before the run). Both are ViT-H/14 trained on ImageNet-1k; MAE saw horizontal flips, I-JEPA did not. Δ = sensitivity(I-JEPA) − sensitivity(MAE), patch mean, L32, paired bootstrap over scenes (2000, seed 0):
- "A" (seeing flips is enough): lower bound of Δ > 0.10;
- "B" (an objective that demands the same output for an image and its mirror is needed): whole CI inside [−0.10, 0.10];
- "against A": upper bound < −0.10;
- otherwise "does not discriminate".
Caveat written at the time: MAE and I-JEPA also differ in their objective (pixels vs. abstract targets), so this is not a clean experiment.
→ Δ = −0.222 [−0.304, −0.148] → "against A". In this pair, flip augmentation is neither necessary nor sufficient for mirror blindness.

**MAE vs. I-JEPA, binding** (`probe_big.py`, `probe_big_compare.py`; rule #39, written 22 Oct, before the run). Gap = accuracy("upper") − accuracy("left"), far background, L32, logistic regression. Δ = gap(I-JEPA) − gap(MAE), paired bootstrap over scenes (2000, seed 0): "follows" if the whole CI > 0; "contradicts" if the whole CI < 0; otherwise "does not discriminate".
→ MAE gap 10.1 [8.5–11.6]; I-JEPA gap 11.4 [9.9–13.0]; Δ = 1.3 [−0.6, 3.2] → "does not discriminate" (run finished 5 Nov). The interval includes 0, so the difference between MAE and I-JEPA is not established.
*For reference, not part of the rule:* Δ by layer (pp) is uneven: L04 −2.3, L08 +0.9, L12 −5.4, L16 −6.7, L20 −4.6, L24 −2.9, L28 +2.3, L32 +1.3 (`results/probe_big_compare_log.txt`).

**Mean-ablation of attention / MLP in DINO blocks 4–12** (`ablate_blocks.py`; rule #43, written 5 Nov before the run; one clarification of how the verdict is formed, 6 Nov ~00:05, before the run; registered on OSF 6 Nov 00:31, [osf.io/us5zv](https://osf.io/us5zv)). In each of two primary conditions, the output of all attention (or all MLP) sublayers in blocks 4–12 is replaced by its mean over all 4,000 images (S and Sp), separately for each token position; the means come from the unmodified network. Probes "left" and "upper" are retrained at L12 (far background; logistic regression, C = 0.1, the same 5 scene-level folds). Δ = accuracy(condition) − accuracy(unmodified), paired bootstrap over scenes (10,000, seed 0), 97.5% CIs (Bonferroni over 2 conditions):
- "erases": lower bound of Δleft > 0 **and** lower bound of (Δleft − Δupper) > 0;
- "not involved": otherwise, if the upper bound of (Δleft − Δupper) < 3 pp;
- "does not discriminate": otherwise;
- flag "destroyed" if accuracy on "upper" falls below 60%.
Verdict by which parts "erase": A attention only / B MLP only / C both / D neither. The 18 single-sublayer conditions get the same labels with 99.72% CIs and are exploratory; CLS is secondary.
→ Attention: Δleft +4.5 [+2.7, +6.3], Δupper −8.2 [−9.8, −6.8], Δleft − Δupper +12.8 [+10.5, +15.0] → "erases". MLP: Δleft +8.5 [+6.9, +10.1], Δupper +3.1 [+1.8, +4.4], Δleft − Δupper +5.3 [+3.3, +7.4] → "erases". No "destroyed" flag (lowest "upper": 75.2%). **Verdict "C": both parts erase.** Single sublayers: 14 "not involved", 4 "does not discriminate" (attn_07, mlp_04, mlp_05, mlp_06), none "erases". Built-in checks pass: the unmodified run reproduces earlier results exactly (64.9% / 83.5%; mirror 0.336 / 0.284); random networks 49.7–52.4%; twin control 100%. No deviations from the registration.
*For reference, not part of the rule:* mirror sensitivity at L12 (CLS / patch mean): unmodified 0.34 / 0.28; attention ablated 0.82 / 0.72; MLP ablated 0.40 / 0.35.

**Mean-ablation on CLIP, blocks 7–12** (`ablate_clip.py`; rule #53 = rule #43; written 9 Nov before the code; registered on OSF 9 Nov 10:49, [osf.io/6uves](https://osf.io/6uves)). Blocks chosen by the same principle as for DINO (from the block after the peak of "left" on the far background — layer 6 in CLIP — to the last block). Same conditions, readout, intervals (97.5% for the 2 primary conditions, 99.58% for 12 single sublayers) and labels; reference conditions on blocks 4–12 are not part of the rule.
→ Attention: Δleft +3.2 [+1.5, +4.9], Δupper −5.0 [−6.7, −3.3], Δleft − Δupper +8.2 [+5.8, +10.5] → "erases". MLP: Δleft +10.0 [+8.3, +11.7], Δupper +10.5 [+9.0, +12.1], Δleft − Δupper −0.5 [−2.8, +1.7] → "not involved". No "destroyed" flag (lowest "upper": 67.2%). **Verdict "A": attention only.** Single sublayers: 7 "not involved", 5 "does not discriminate", none "erases". Built-in checks pass: the unmodified run reproduces earlier CLIP results exactly (63.15% / 72.15%; CLS 63.80% / 73.125%; mirror 0.654 / 0.613); random networks 49.5–53.1%; twin control 100%. No deviations from the registration.
*For reference, not part of the rule:* mirror sensitivity at L12 (CLS / patch mean): unmodified 0.65 / 0.61; attention ablated 1.20 / 0.79; MLP ablated 0.79 / 0.62. Blocks 4–12 (as in DINO): attention Δleft −3.9 (blocks 4–6 build "left"), MLP Δleft +13.1, Δupper +7.2.

**Twin-token patching, DINO** (`patch_twins.py`; rule #59/#60; written 9 Nov before the run; registered on OSF 9 Nov 13:59, [osf.io/23wjy](https://osf.io/23wjy)). A "left" probe (logistic regression, far background, L12) is trained once on the unmodified network (4 scene folds) and then kept fixed. For a scene pair S/S′ (the twins swap colors), the tokens of both twins in the run on one image are replaced at layer L (L00–L11) by their values from the run on the other image; the network continues to L12 and the fixed probe answers on the held-out fold. T(L) = share of runs in which the probe now gives the color of the source image's left twin (only runs where the probe is right on both unmodified images). Rule: L* = the first layer L01–L11 with point T_twins < 50% (otherwise "none"); also report T_twins(L11). Reference conditions: *control* (the same number of other tokens: untouched objects, then near background), *rest* (all tokens except the twins), three random DINO networks (twins only). Code check: T_twins(L00) = 100%, T_control(L00) = 0%.
→ **L* = L06** (T = 43.8% [41.1, 46.7]; L05: 57.7% [54.9, 60.4]); **T_twins(L11) = 11.9% [10.2, 13.7]**. Control ≤ 6.9% on every layer; rest = 100 − twins. Random networks (probe at chance, 50.4–51.5%; 872–902 runs counted): below 50% already at L02–L03, ≈ 11–12% at L11. Checks: the unmodified probe reproduces probe_v2 exactly (64.875%), 1420 runs counted, no convergence warnings. **Deviation from the registered checks:** the L00 check printed a warning (99.9% / 0.1% instead of 100% / 0%). Cause: patched features are stored as float16 and clean ones as float32; at L00 the patched run equals the source run up to float16 rounding (max error 0.0039), and this flips 2 of 1420 runs that lie on the probe's decision boundary. Rounding the clean features to float16 as well gives exactly 100.0% / 0.0%. This does not affect the verdict.

**Negative result kept:** a non-linear (MLP) probe, set up in advance, was weaker than the linear probe on every layer, so it cannot answer whether information disappears or only stops being linearly readable. Its settings were not tuned on DINO's results.

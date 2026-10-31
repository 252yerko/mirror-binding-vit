"""figure1.py — Рисунок 1 для одной страницы / письма (7.10). Ничего не обучает: берёт готовые числа.
(A) Чувствительность к зеркалу по слоям, CLS: отношение d(зеркало ←→) / d(зеркало ↑↓) из flip_test.csv и
    flip_test_clip.csv (1 = зеркала замечаются одинаково; меньше 1 = слепота к лево/право). Полоса — 95% ДИ.
(B) D по слоям, CLS: та часть потери горизонтали, что касается только стороны (лево/право), в п.п.;
    из probe_fold_scores.npz и probe_fold_clip_scores.npz, 1043 сцены общего подмножества, парный бутстрэп
    по сценам (1000 повторов, seed 0 — как probe_fold_compare.py). Серая полоса — мин–макс 6 случайных сетей.
Запуск (из binding-vit):  python figure1.py   → figure1.png, figure1.pdf"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a8a7a1"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
L = np.arange(1, 13)

def flip_rows(path, model):
    r = {row["layer"]: row for row in csv.DictReader(open(path, encoding="utf-8"))
         if row["model"] == model and row["group"] == "cls"}
    get = lambda k: np.array([float(r[f"L{l:02d}"][k]) for l in L])
    return get("ratio_h_v"), get("ci_lo"), get("ci_hi")

A = {"DINO": flip_rows("flip_test.csv", "DINO"), "CLIP": flip_rows("flip_test_clip.csv", "CLIP")}
rand_ratio = np.array([flip_rows("flip_test.csv", m)[0] for m in ("rand0", "rand1", "rand2")] +
                      [flip_rows("flip_test_clip.csv", m)[0] for m in ("clip_rand0", "clip_rand1", "clip_rand2")])

Zd, Zc = np.load("probe_fold_scores.npz"), np.load("probe_fold_clip_scores.npz")
assert (Zd["sub"] == Zc["sub"]).all()
idx = np.where(Zd["sub"])[0]
rng = np.random.default_rng(0)
BOOT = [rng.choice(len(idx), len(idx)) for _ in range(1000)]
def dvec(Z, net, l):
    s = {q: Z[f"{net}|{q}|cls|L{l:02d}"][idx] for q in ("left", "upper", "cx", "cy")}
    return (s["upper"] - s["left"]) - (s["cy"] - s["cx"])
def dcurve(Z, net, ci=True):
    m, lo, hi = [], [], []
    for l in L:
        v = dvec(Z, net, l); m.append(np.nanmean(v) * 100)
        if ci:
            b = [np.nanmean(v[k]) * 100 for k in BOOT]; lo.append(np.percentile(b, 2.5)); hi.append(np.percentile(b, 97.5))
    return np.array(m), np.array(lo), np.array(hi)
B = {"DINO": dcurve(Zd, "acts_v2"), "CLIP": dcurve(Zc, "acts_v2_clip")}
rand_D = np.array([dcurve(Zd, f"acts_v2_rand{k}", ci=False)[0] for k in range(3)] +
                  [dcurve(Zc, f"acts_v2_clip_rand{k}", ci=False)[0] for k in range(3)])

print("Сверка с логами (L12): ratio DINO %.2f, CLIP %.2f; D DINO %.1f [%.1f…%.1f], CLIP %.1f [%.1f…%.1f]" %
      (A["DINO"][0][-1], A["CLIP"][0][-1], *[B["DINO"][i][-1] for i in range(3)], *[B["CLIP"][i][-1] for i in range(3)]))
print("Случайные сети: ratio %.2f–%.2f, D %.1f…%.1f" % (rand_ratio.min(), rand_ratio.max(), rand_D.min(), rand_D.max()))

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.linewidth": 0.8, "figure.facecolor": SURF, "axes.facecolor": SURF})
fig, axs = plt.subplots(1, 2, figsize=(10, 3.9))
panels = [(axs[0], A, rand_ratio, "Sensitivity to mirror flips (CLS)",
           "left–right flip effect / up–down flip effect\n(1 = both flips matter equally; lower = blind to left vs right)", 1.0),
          (axs[1], B, rand_D, "Loss specific to left vs right (CLS)",
           "D, percentage points\n(probe loss for side minus loss for distance to middle)", 0.0)]
for ax, data, rnd, title, ylab, ref in panels:
    ax.fill_between(L, rnd.min(0), rnd.max(0), color=GRAY, alpha=0.35, lw=0, label="random-init networks (6)")
    ax.axhline(ref, color=INK2, lw=0.8, ls=(0, (3, 3)))
    for name, col in (("DINO", BLUE), ("CLIP", ORANGE)):
        m, lo, hi = data[name]
        ax.fill_between(L, lo, hi, color=col, alpha=0.15, lw=0)
        ax.plot(L, m, color=col, lw=2, marker="o", ms=4, label={"DINO": "DINO ViT-B/16 (trained with flips)",
                                                                  "CLIP": "CLIP ViT-B/16 (trained without flips)"}[name])
        ax.annotate(f"{name} {m[-1]:.2f}" if ref == 1.0 else f"{name} {m[-1]:.1f}", (12, m[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=8.5)
    ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold")
    ax.set_ylabel(ylab, fontsize=8.5, color=INK2)
    ax.set_xlabel("ViT layer", color=INK2)
    ax.set_xticks(L); ax.set_xlim(0.7, 13.6)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axs[0].text(0.02, 0.04, "A", transform=axs[0].transAxes, fontsize=12, fontweight="bold", color=INK)
axs[1].text(0.02, 0.92, "B", transform=axs[1].transAxes, fontsize=12, fontweight="bold", color=INK)
h, lab = axs[0].get_legend_handles_labels()
fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig("figure1.png", dpi=200); fig.savefig("figure1.pdf")
print("Готово: figure1.png, figure1.pdf")

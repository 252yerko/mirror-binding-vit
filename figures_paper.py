"""figures_paper.py — рисунки 1–4 для статьи в одном стиле (белый фон, шрифт STIX как Times в LaTeX,
буквы панелей a, b, c, короткие подписи осей; пояснения — в подписях к рисункам в тексте статьи).
Ничего не считает заново по сетям: берёт готовые CSV / npz. Печатает сверку чисел с README.

Рисунок 1 (как figure1.py, тот же расчёт): (a) зеркальная чувствительность CLS по слоям, DINO и CLIP;
  (b) D по слоям (потеря именно стороны), CLS; серые полосы — 6 случайных сетей.
  Данные: flip_test.csv, flip_test_clip.csv, probe_fold_scores.npz, probe_fold_clip_scores.npz (папка --work).
Рисунок 2 (osf.io/xyedr): 24 сети, последний слой, дальний фон. (a) чувствительность против разрыва (B1);
  (b) чувствительность против R (B2).
Рисунок 3 (зеркальные пары голов): (a) dx голов DINO; (b) сходство карт напрямую и после зеркала;
  (c) DINO, замена выхода голов (resample); (d) DeiT III, наборы голов против 50 случайных.
Рисунок 4 (osf.io/ducms): ViT-Tiny с нуля, условия 1–3 × 4 зерна: «левый», «верхний», чувствительность по эпохам.

Запуск из репозитория (папка --work — там, где probe_fold.py сохранил probe_fold_scores.npz и
probe_fold_clip_scores.npz; flip_test.csv и flip_test_clip.csv там же, они совпадают с results/):
  python figures_paper.py --res results --work <папка> --out figures/paper"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

ap = argparse.ArgumentParser()
ap.add_argument("--res", default="results")
ap.add_argument("--work", default=".")
ap.add_argument("--out", default="figures_paper")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
R = lambda f: os.path.join(args.res, f)
W = lambda f: os.path.join(args.work, f)
rows = lambda f: list(csv.DictReader(open(R(f), encoding="utf-8")))

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#a8a7a1"
INK, INK2, GRID = "#000000", "#444444", "#e8e8e8"
WIDTH = 7.0          # дюймы: ширина текста двухколоночной статьи
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "axes.titlelocation": "left", "axes.titleweight": "normal", "pdf.fonttype": 42})


def letter(ax, s, title=""):
    """Буква панели жирным + короткий заголовок обычным шрифтом."""
    ax.set_title(r"$\bf{" + s + "}$   " + title, loc="left", pad=6)


def rank(a):
    return np.argsort(np.argsort(a))


def spearman(a, b):
    return float(np.corrcoef(rank(np.asarray(a)), rank(np.asarray(b)))[0, 1])


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out, f"{name}.{ext}"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"→ {name}.png / {name}.pdf")


def grid(ax, axis="both"):
    ax.grid(axis=axis, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- Рисунок 1 (тот же расчёт, что figure1.py)
L = np.arange(1, 13)


def flip_rows(path, model):
    r = {row["layer"]: row for row in csv.DictReader(open(path, encoding="utf-8"))
         if row["model"] == model and row["group"] == "cls"}
    get = lambda k: np.array([float(r[f"L{l:02d}"][k]) for l in L])
    return get("ratio_h_v"), get("ci_lo"), get("ci_hi")


A1 = {"DINO": flip_rows(W("flip_test.csv"), "DINO"), "CLIP": flip_rows(W("flip_test_clip.csv"), "CLIP")}
rand_ratio = np.array([flip_rows(W("flip_test.csv"), m)[0] for m in ("rand0", "rand1", "rand2")] +
                      [flip_rows(W("flip_test_clip.csv"), m)[0] for m in ("clip_rand0", "clip_rand1", "clip_rand2")])
Zd, Zc = np.load(W("probe_fold_scores.npz")), np.load(W("probe_fold_clip_scores.npz"))
assert (Zd["sub"] == Zc["sub"]).all()
idx = np.where(Zd["sub"])[0]
rng0 = np.random.default_rng(0)
BOOT = [rng0.choice(len(idx), len(idx)) for _ in range(1000)]


def dvec(Z, net, l):
    s = {q: Z[f"{net}|{q}|cls|L{l:02d}"][idx] for q in ("left", "upper", "cx", "cy")}
    return (s["upper"] - s["left"]) - (s["cy"] - s["cx"])


def dcurve(Z, net, ci=True):
    m, lo, hi = [], [], []
    for l in L:
        v = dvec(Z, net, l)
        m.append(np.nanmean(v) * 100)
        if ci:
            b = [np.nanmean(v[k]) * 100 for k in BOOT]
            lo.append(np.percentile(b, 2.5))
            hi.append(np.percentile(b, 97.5))
    return np.array(m), np.array(lo), np.array(hi)


B1 = {"DINO": dcurve(Zd, "acts_v2"), "CLIP": dcurve(Zc, "acts_v2_clip")}
rand_D = np.array([dcurve(Zd, f"acts_v2_rand{k}", ci=False)[0] for k in range(3)] +
                  [dcurve(Zc, f"acts_v2_clip_rand{k}", ci=False)[0] for k in range(3)])
print("Рисунок 1 — сверка (README: CLS L12 DINO 0.34, CLIP 0.65; D DINO 19.5 [16.5, 22.6], CLIP 4.9 [1.7, 8.2]):")
print("  отношение L12: DINO %.2f, CLIP %.2f; D L12: DINO %.1f [%.1f, %.1f], CLIP %.1f [%.1f, %.1f]" %
      (A1["DINO"][0][-1], A1["CLIP"][0][-1], *[B1["DINO"][i][-1] for i in range(3)], *[B1["CLIP"][i][-1] for i in range(3)]))
assert round(A1["DINO"][0][-1], 2) == 0.34 and round(B1["DINO"][0][-1], 1) == 19.5 and round(B1["CLIP"][0][-1], 1) == 4.9

fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 2.4))
for ax, data, rnd, ylab, ref, lt, ttl in (
        (axs[0], A1, rand_ratio, r"mirror sensitivity  $d_{\leftrightarrow}/d_{\updownarrow}$ (CLS)", 1.0, "a",
         "Mirror sensitivity across layers"),
        (axs[1], B1, rand_D, r"side-specific loss $D$ (pp, CLS)", 0.0, "b", "Loss of the side across layers")):
    ax.fill_between(L, rnd.min(0), rnd.max(0), color=GRAY, alpha=0.35, lw=0, label="random-init (6)")
    ax.axhline(ref, color=INK2, lw=0.6, ls=(0, (3, 3)))
    for name, col in (("DINO", BLUE), ("CLIP", ORANGE)):
        m, lo, hi = data[name]
        ax.fill_between(L, lo, hi, color=col, alpha=0.18, lw=0)
        ax.plot(L, m, color=col, lw=1.4, marker="o", ms=2.8, label=f"{name} ViT-B/16")
    ax.set_xticks(L)
    ax.set_xlim(0.6, 12.4)
    ax.set_xlabel("layer")
    ax.set_ylabel(ylab)
    letter(ax, lt, ttl)
    grid(ax, "y")
axs[0].legend(loc="lower left")
fig.tight_layout(w_pad=2.5)
save(fig, "figure1")

# ---------------------------------------------------------------- Рисунок 2
M = rows("many_nets.csv")
tr = [r for r in M if r["seed"] == ""]
rd = [r for r in M if r["seed"] != ""]
assert len(tr) == 24 and len(rd) == 7, (len(tr), len(rd))
f = lambda rs, k: np.array([float(r[k]) for r in rs])
x, gap, Rr = f(tr, "ratio"), f(tr, "gap") * 100, f(tr, "R")
rule = {r["rule"]: r for r in rows("many_nets_rules.csv")}
rho1, rho2 = spearman(x, gap), spearman(x, Rr)
print("Рисунок 2 — сверка:")
print(f"  24 сети: чувствительность {x.min():.3f}–{x.max():.3f} (README 0.284–0.956); разрыв {gap.min():.1f}–{gap.max():.1f} п.п. (README 3.9–20.7)")
print(f"  B1 ρ = {rho1:+.4f} (CSV правил {float(rule['B1']['rho']):+.4f}, p {float(rule['B1']['p']):.3f}); "
      f"B2 ρ = {rho2:+.4f} (CSV {float(rule['B2']['rho']):+.4f}, p {float(rule['B2']['p']):.3f})")
assert abs(rho1 - float(rule["B1"]["rho"])) < 1e-9 and abs(rho2 - float(rule["B2"]["rho"])) < 1e-9
xr, gr = f(rd, "ratio"), f(rd, "gap") * 100
print(f"  7 случайных сетей: чувствительность {xr.min():.3f}–{xr.max():.3f}; разрыв {gr.min():.1f}…{gr.max():.1f} п.п.")

LAB = {"DINO B/16": (-5, -1, "right"), "DINO S/16": (5, 3, "left"), "DeiT III L/16": (5, 2, "left"),
       "MAE B/16": (5, -3, "left"), "MAE L/16": (-5, -9, "right"), "AugReg L/16": (-5, 5, "right"),
       "CLIP-OpenAI B/16": (5, -7, "left"), "DFN B/16": (5, -7, "left")}
fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 2.7))
ax = axs[0]
ax.errorbar(x, gap, xerr=[x - f(tr, "ratio_lo"), f(tr, "ratio_hi") - x],
            yerr=[gap - f(tr, "gap_lo") * 100, f(tr, "gap_hi") * 100 - gap],
            fmt="none", ecolor=BLUE, alpha=0.35, lw=0.6, zorder=1)
ax.scatter(x, gap, s=14, color=BLUE, edgecolor="white", lw=0.6, zorder=3, label="pretrained (24)")
ax.scatter(xr, gr, s=14, color=GRAY, marker="s", edgecolor="white", lw=0.6, zorder=2, label="random-init (7)")
ax.axhline(0, color=INK2, lw=0.6, ls=(0, (3, 3)))
ax.axvline(1, color=INK2, lw=0.6, ls=(0, (3, 3)))
for r, xi, gi in zip(tr, x, gap):
    if r["name"] in LAB:
        dx, dy, ha = LAB[r["name"]]
        ax.annotate(r["name"], (xi, gi), xytext=(dx, dy), textcoords="offset points", fontsize=6.5, color=INK2, ha=ha)
ax.set_xlim(0.15, 1.1)
ax.set_xlabel(r"mirror sensitivity  $d_{\leftrightarrow}/d_{\updownarrow}$ (patch mean)")
ax.set_ylabel('gap "upper" − "left" (pp)')
letter(ax, "a")
ax.legend(loc="upper right")
grid(ax)

ax = axs[1]
ax.errorbar(x, Rr, yerr=[Rr - f(tr, "R_lo"), f(tr, "R_hi") - Rr], fmt="none", ecolor=BLUE, alpha=0.35, lw=0.6)
ax.scatter(x, Rr, s=14, color=BLUE, edgecolor="white", lw=0.6, zorder=3)
ax.axhline(0.5, color=INK2, lw=0.6, ls=(0, (3, 3)))
ax.axhline(1.0, color=INK2, lw=0.6, ls=(0, (1, 2)))
for r, xi, ri in zip(tr, x, Rr):
    if r["name"] in ("DINO B/16", "DFN B/16", "AugReg S/16", "AugReg B/16"):
        off, ha = {"DINO B/16": ((2, -10), "left"), "DFN B/16": ((3, 7), "left")}.get(r["name"], ((5, 2), "left"))
        ax.annotate(r["name"], (xi, ri), xytext=off, textcoords="offset points", fontsize=6.5, color=INK2, ha=ha)
ax.set_ylim(-0.2, max(2.6, (f(tr, "R_hi")).max() + 0.1))
ax.set_xlabel(r"mirror sensitivity  $d_{\leftrightarrow}/d_{\updownarrow}$ (patch mean)")
ax.set_ylabel(r"$R$ (share of the gap in the antisymmetric part)")
letter(ax, "b")
grid(ax)
fig.tight_layout(w_pad=2.5)
save(fig, "figure2")

# ---------------------------------------------------------------- Рисунок 3
H = [r for r in rows("heads_pair_heads.csv") if r["net"] == "DINO B/16" and r["dx_lo"] != ""]
assert len(H) == 144, len(H)
DX = np.zeros((12, 12))
for r in H:
    DX[int(r["block"]) - 1, int(r["head"])] = float(r["dx"])
print("Рисунок 3 — сверка:")
print(f"  DINO dx 4.4 = {DX[3, 4]:+.4f}, 4.5 = {DX[3, 5]:+.4f} (лог: +1.3101 / −1.5221)")
P = rows("heads_pair_pairs.csv")
p45 = [r for r in P if r["net"] == "DINO B/16" and r["block"] == "4" and {r["a"], r["b"]} == {"4", "5"}][0]
print(f"  пара 4.4/4.5: напрямую {float(p45['s_dir']):+.3f}, после зеркала {float(p45['s_mir']):+.3f} (README +0.004 / +0.934)")
G = {r["cond"]: r for r in rows("heads_pair_g2.csv") if r["kind"] == "resample"}
Q = rows("heads_pair_plus.csv")
q = {r["cond"]: r for r in Q}

BLUE_ORANGE = LinearSegmentedColormap.from_list("bo", [ORANGE, "#f7f7f7", BLUE])
fig = plt.figure(figsize=(WIDTH, 5.0))
gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.45)
ax = fig.add_subplot(gs[0, 0])
lim = np.abs(DX).max()
im = ax.imshow(DX, cmap=BLUE_ORANGE, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
for (b, h) in ((3, 4), (3, 5)):
    ax.add_patch(plt.Rectangle((h - 0.5, b - 0.5), 1, 1, fill=False, ec=INK, lw=1.0))
ax.annotate("4.4", (4, 3), xytext=(2.3, 6.3), fontsize=7, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5))
ax.annotate("4.5", (5, 3), xytext=(6.7, 6.3), fontsize=7, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5))
ax.set_xticks(range(12))
ax.set_yticks(range(12))
ax.set_yticklabels(range(1, 13))
ax.set_xlabel("head")
ax.set_ylabel("block")
ax.spines[["top", "right"]].set_visible(True)
letter(ax, "a", "DINO B/16")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label("horizontal attention offset (patches)")
cb.outline.set_linewidth(0.5)
cb.ax.tick_params(width=0.5, length=2)

ax = fig.add_subplot(gs[0, 1])
for net, col, lab in (("CLIP-OpenAI B/16", GRAY, "CLIP"), ("DeiT III B/16", ORANGE, "DeiT III"), ("DINO B/16", BLUE, "DINO")):
    rr = [r for r in P if r["net"] == net]
    sd, sm = np.array([float(r["s_dir"]) for r in rr]), np.array([float(r["s_mir"]) for r in rr])
    g3 = np.array([r["g3"] == "1" for r in rr])
    ax.scatter(sd[~g3], sm[~g3], s=3, color=col, alpha=0.35, lw=0)
    ax.scatter(sd[g3], sm[g3], s=16, color=col, edgecolor=INK, lw=0.5, zorder=4,
               label=f"{lab} ({g3.sum()} mirror pairs)")
    print(f"  {net}: пар внутри блоков {len(rr)}, зеркальных пар по G3 {g3.sum()}")
lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
ax.plot([lo, 1], [lo, 1], color=INK2, lw=0.6, ls=(0, (3, 3)))
ax.annotate("4.4 / 4.5", (float(p45["s_dir"]), float(p45["s_mir"])), xytext=(8, -3), textcoords="offset points", fontsize=7)
ax.set_xlabel("similarity of the two attention maps")
ax.set_ylabel("similarity after mirroring one map")
letter(ax, "b", "pairs of heads within a block")
ax.legend(loc="lower right", handletextpad=0.2)
grid(ax)

ax = fig.add_subplot(gs[1, 0])
items = [("4.4", "одна 4.4", BLUE), ("4.5", "одна 4.5", BLUE), ("4.4 + 4.5", "пара, один донор", INK),
         ("4.7", "контроль 4.7", GRAY), ("4.8", "контроль 4.8", GRAY), ("4.9", "контроль 4.9", GRAY)]
for i, (lab, key, col) in enumerate(items):
    r = G[key]
    v, l, h = (float(r[k]) * 100 for k in ("far_dleft", "far_dleft_lo", "far_dleft_hi"))
    ax.errorbar(i, v, yerr=[[v - l], [h - v]], fmt="o", color=col, ms=3.5, lw=1.0, capsize=0)
ax.axhline(0, color=INK2, lw=0.6)
ax.axvspan(2.5, 5.5, color=GRID, alpha=0.6, lw=0)
ax.text(4, ax.get_ylim()[1], "controls", ha="center", va="top", fontsize=7, color=INK2)
ax.set_xticks(range(len(items)))
ax.set_xticklabels([t[0] for t in items])
ax.set_xlim(-0.5, 5.5)
ax.set_xlabel("heads replaced (DINO, block 4)")
ax.set_ylabel(r'$\Delta$ "left" (pp)')
letter(ax, "c", "DINO: resample-ablation")
grid(ax, "y")
print("  DINO resample «левый»: " + ", ".join(f"{t[0]} {float(G[t[1]]['far_dleft']) * 100:+.2f}" for t in items))

ax = fig.add_subplot(gs[1, 1])
r6 = np.array([float(r["dleft"]) * 100 for r in Q if r["cond"].startswith("случайный набор 6 ")])
r12 = np.array([float(r["dleft"]) * 100 for r in Q if r["cond"].startswith("случайный набор 12 ")])
assert len(r6) == 50 and len(r12) == 50
rng = np.random.default_rng(0)
for xpos, arr in ((0.5, r6), (2, r12)):
    ax.scatter(xpos + rng.uniform(-0.17, 0.17, len(arr)), arr, s=4, color=GRAY, alpha=0.8, lw=0)
sets = [(0, "правые (по одной из каждой пары)", BLUE), (1, "левые (по одной из каждой пары)", ORANGE),
        (2.55, "все 12 голов пар", INK)]
for xpos, key, col in sets:
    r = q[key]
    v, l, h = (float(r[k]) * 100 for k in ("dleft", "dleft_lo", "dleft_hi"))
    ax.errorbar(xpos, v, yerr=[[v - l], [h - v]], fmt="o", color=col, ms=3.5, lw=1.0)
ax.axhline(0, color=INK2, lw=0.6)
ax.set_xticks([0, 0.5, 1, 2, 2.55])
ax.set_xticklabels(["6 right", "random 6", "6 left", "random 12", "all 12"])
ax.set_xlabel("heads replaced (DeiT III)")
ax.set_ylabel(r'$\Delta$ "left" (pp)')
letter(ax, "d", "DeiT III: one vs. both heads of every pair")
grid(ax, "y")
print(f"  DeiT III: правые {float(q[sets[0][1]]['dleft']) * 100:+.2f}, левые {float(q[sets[1][1]]['dleft']) * 100:+.2f}, "
      f"все 12 {float(q[sets[2][1]]['dleft']) * 100:+.2f} (README +4.83 / +2.90 / +0.47); "
      f"случайные 6: макс {r6.max():+.2f}; случайные 12: {r12.min():+.2f}…{r12.max():+.2f}")
save(fig, "figure3")

# ---------------------------------------------------------------- Рисунок 4
COND = {1: ("no flips", BLUE, "-"), 2: ("horizontal flips", ORANGE, (0, (4, 2))), 3: ("vertical flips", AQUA, (0, (1, 1.2)))}
D = {}
for c in (1, 2, 3):
    for s in range(4):
        rr = rows(f"e_measure_full_c{c}_s{s}.csv")
        ep = [r for r in rr if r["point"].startswith("ep")]
        D[c, s] = {k: np.array([float(r[k]) for r in ep]) for k in ("epoch", "left", "upper", "ratio")}
        if (c, s) == (1, 0):
            RND = {k: np.array([float(r[k]) for r in rr if r["point"] == "random"]) for k in ("left", "upper", "ratio")}
print("Рисунок 4 — сверка (эпоха 40, среднее 4 зёрен; README: «левый» 52.33/51.76/51.07, «верхний» 85.96/85.96/51.92, "
      "отношение 0.145/0.150/1.084):")
for c in (1, 2, 3):
    m = {k: np.mean([D[c, s][k][-1] for s in range(4)]) for k in ("left", "upper", "ratio")}
    print(f"  усл. {c}: «левый» {m['left'] * 100:.2f}, «верхний» {m['upper'] * 100:.2f}, отношение {m['ratio']:.3f}")
print(f"  случайные: «левый» {RND['left'].min() * 100:.2f}–{RND['left'].max() * 100:.2f}, «верхний» "
      f"{RND['upper'].min() * 100:.2f}–{RND['upper'].max() * 100:.2f}, отношение {RND['ratio'].min():.2f}–{RND['ratio'].max():.2f}")
peaks = [(c, s, D[c, s]["ratio"][1:5].max()) for c in (1, 2) for s in range(4)]
print("  H4: максимум отношения на эпохах 1–5: " + ", ".join(f"c{c}_s{s} {v:.2f}" for c, s, v in peaks)
      + f" → выше 1.2 у {sum(v > 1.2 for _, _, v in peaks)} из 8")

fig, axs = plt.subplots(1, 3, figsize=(WIDTH, 2.3))
panels = [("left", "a", '"left" (%)', 100), ("upper", "b", '"upper" (%)', 100),
          ("ratio", "c", r"$d_{\leftrightarrow}/d_{\updownarrow}$ (log scale)", 1)]
for ax, (k, lt, ylab, sc) in zip(axs, panels):
    ax.axhspan(RND[k].min() * sc, RND[k].max() * sc, color=GRAY, alpha=0.35, lw=0)
    for c in (1, 2, 3):
        lab, col, ls = COND[c]
        for s in range(4):
            ax.plot(D[c, s]["epoch"], D[c, s][k] * sc, color=col, ls=ls, lw=0.9, label=lab if s == 0 else None)
    ax.set_xscale("function", functions=(np.sqrt, np.square))
    ax.set_xticks([0, 1, 3, 5, 12, 20, 30, 40])
    ax.set_xlim(0, 41)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylab)
    letter(ax, lt)
    grid(ax)
    if k == "ratio":
        ax.set_yscale("log")
        ax.axhline(1, color=INK2, lw=0.6, ls=(0, (3, 3)))
        ax.axhline(1.2, color=INK2, lw=0.5, ls=(0, (1, 2)))
    else:
        ax.axhline(50, color=INK2, lw=0.6, ls=(0, (3, 3)))
h, lab = axs[0].get_legend_handles_labels()
fig.legend(h, lab, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07), handlelength=2.6)
fig.tight_layout(w_pad=1.8)
save(fig, "figure4")

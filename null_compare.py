"""
null_compare.py — сравнение обученной DINO с нулевой моделью (DINO со случайными весами).

ВОПРОС:
  swap_v2.py показал: у обученной DINO чистое изменение привязки (близнецы поменялись
  цветами) меняет токены нетронутых объектов, фона и CLS — отношение swap/recolor
  на L12 около 0.17–0.49. Это выучено при обучении или так ведёт себя любая сеть
  такой формы? Ответ даёт та же сеть до обучения (случайные веса).

ЧТО ДЕЛАЕТ:
  Читает swap_v2.csv (обученная DINO) и все swap_v2_rand*.csv (случайные веса,
  разные seed) — их делает swap_v2.py с флагом --tag. Ничего не пересчитывает,
  только сводит готовые числа в одну таблицу и график.

КАК ЧИТАТЬ:
  отношение у DINO заметно выше, чем у ВСЕХ случайных моделей
      -> чувствительность к привязке появилась от обучения
  отношение у DINO примерно как у случайных
      -> эффект даёт устройство сети, а не обучение
  Разброс между seed показывает, насколько число у случайной сети зависит от случайности.

ЗАПУСК (после swap_v2.py для DINO и для rand0, rand1, rand2):
  python null_compare.py 2>&1 | tee null_compare_log.txt
  -> таблица, график null_compare.png, числа null_compare.csv
"""

import csv
import glob
import math
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GROUPS = ["unchanged", "background", "cls", "twins"]
NAMES = {"twins": "близнецы", "unchanged": "нетронутые", "background": "фон", "cls": "CLS"}


def read(path):
    """{группа: {слой: {swap, recolor, ratio, ci_lo, ci_hi}}} из csv, который пишет swap_v2.py."""
    out = {}
    for r in csv.DictReader(open(path)):
        out.setdefault(r["group"], {})[r["layer"]] = {k: float(r[k]) for k in ["swap", "recolor", "ratio", "ci_lo", "ci_hi"]}
    return out


def fmt(x, nd=2):
    return "  —  " if math.isnan(x) else f"{x:.{nd}f}"


def main():
    dino = read("swap_v2.csv")
    paths = sorted(glob.glob("swap_v2_rand*.csv"), key=lambda p: int(re.findall(r"rand(\d+)", p)[0]))
    assert paths, "не нашёл swap_v2_rand*.csv — сначала запусти swap_v2.py --acts acts_v2_randN --tag randN"
    seeds = [re.findall(r"rand(\d+)", p)[0] for p in paths]
    rand = [read(p) for p in paths]
    layers = list(dino["cls"].keys())
    for r, p in zip(rand, paths):   # проверка: у всех файлов одинаковые группы и слои
        assert set(r) == set(dino) and all(list(r[g].keys()) == layers for g in r), f"{p}: другие группы/слои"
    print(f"Обученная DINO: swap_v2.csv.  Случайные веса: {len(paths)} шт. (seed {', '.join(seeds)})\n")

    rows = []
    for g in GROUPS:
        print(f"== {NAMES[g]} ==   отношение swap/recolor")
        print("слой   DINO  [95% инт.]     случайные: среднее (мин–макс)   recolor DINO / случ.")
        for l in layers:
            d = dino[g][l]
            rr = np.array([r[g][l]["ratio"] for r in rand])
            rc = np.array([r[g][l]["recolor"] for r in rand])
            ok = rr[~np.isnan(rr)]
            rmean = float(ok.mean()) if len(ok) else float("nan")
            rmin = float(ok.min()) if len(ok) else float("nan")
            rmax = float(ok.max()) if len(ok) else float("nan")
            print(f"{l}   {fmt(d['ratio'])}  [{fmt(d['ci_lo'])}–{fmt(d['ci_hi'])}]      "
                  f"{fmt(rmean)} ({fmt(rmin)}–{fmt(rmax)})              {d['recolor']:.4f} / {rc.mean():.4f}")
            rows.append([g, l, d["ratio"], d["ci_lo"], d["ci_hi"], rmean, rmin, rmax, d["recolor"], float(rc.mean())]
                        + [r[g][l]["ratio"] for r in rand])
        print()

    # --- итог по последнему слою
    last = layers[-1]
    print(f"ИТОГ на {last} (вне близнецов — там, где пиксели не менялись):")
    for g in ["unchanged", "background", "cls"]:
        d = dino[g][last]
        rr = np.array([r[g][last]["ratio"] for r in rand])
        above = d["ci_lo"] > np.nanmax(rr)
        print(f"  {NAMES[g]:11s} DINO {d['ratio']:.2f} [{d['ci_lo']:.2f}–{d['ci_hi']:.2f}]   "
              f"случайные {np.nanmean(rr):.2f} (макс {np.nanmax(rr):.2f})   "
              f"{'DINO выше всех случайных ✅' if above else 'DINO НЕ выше всех случайных ⚠️'}")
    print("  (\"выше всех\" = нижняя граница интервала DINO больше, чем максимум среди случайных seed)")

    with open("null_compare.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "layer", "dino_ratio", "dino_ci_lo", "dino_ci_hi", "rand_mean", "rand_min", "rand_max",
                    "dino_recolor", "rand_recolor_mean"] + [f"rand{s}_ratio" for s in seeds])
        for row in rows:
            w.writerow(row[:2] + [f"{x:.5f}" for x in row[2:]])

    # --- график: по панели на группу; DINO с интервалом, каждая случайная сеть — серая линия
    L = np.arange(len(layers))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, g in zip(axes, ["unchanged", "background", "cls"]):
        ax.plot(L, [dino[g][l]["ratio"] for l in layers], "o-", c="C3", label="обученная DINO")
        ax.fill_between(L, [dino[g][l]["ci_lo"] for l in layers], [dino[g][l]["ci_hi"] for l in layers], color="C3", alpha=0.15)
        for i, (r, s) in enumerate(zip(rand, seeds)):
            ax.plot(L, [r[g][l]["ratio"] for l in layers], "o--", c="gray", alpha=0.6, ms=3,
                    label="случайные веса" if i == 0 else None)
        ax.axhline(0, ls=":", c="k", lw=1)
        ax.set_title(f"{NAMES[g]}: swap / recolor")
        ax.set_xticks(L, layers, fontsize=7)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("swap / recolor  (0 = мешок признаков)")
    fig.suptitle(f"Обученная DINO против нулевой модели (случайные веса, seed {', '.join(seeds)})", fontsize=10)
    fig.tight_layout()
    fig.savefig("null_compare.png", dpi=150)
    print("\nГрафик: null_compare.png   Числа: null_compare.csv")


if __name__ == "__main__":
    main()

"""
swap_v2.py — главный вопрос v2: видит ли DINO ЧИСТОЕ изменение привязки?

СРАВНИВАЕМ ДВЕ ПАРЫ КАРТИНОК:
  S vs Sp — близнецы поменялись цветами: изменилась ТОЛЬКО привязка
            (набор объектов и количество каждого цвета — те же до пикселя)
  S vs Sr — близнецы перекрашены в новые цвета: изменился НАБОР признаков
            (контроль: меняются те же пиксели, но по-другому)

ГДЕ СМОТРИМ:
  близнецы    — патчи самих близнецов (их пиксели меняются в обоих случаях)
  нетронутые  — патчи остальных объектов (пиксели одинаковые во всех трёх картинках)
  фон         — пустые патчи
  CLS         — общий вектор картинки

ГЛАВНОЕ ЧИСЛО — отношение swap / recolor для нетронутых, фона и CLS:
  около 0  -> модель почти не замечает перестановку, хотя замечает перекраску:
             вне самих объектов она ведёт себя как "мешок признаков"
  около 1  -> перестановка для неё так же заметна, как перекраска:
             информация о привязке есть и вне самих объектов
  Для каждого отношения считаем 95% доверительный интервал (бутстрэп по сценам):
  это диапазон, в котором число, скорее всего, останется, если взять другие сцены.

ЗАПУСК (после generate_v2.py и extract.py --data data_v2 --out acts_v2):
  python swap_v2.py 2>&1 | tee swap_v2_log.txt
  -> таблица, график swap_v2.png, числа swap_v2.csv

НУЛЕВАЯ МОДЕЛЬ (добавлено 5.10) — тот же анализ на активациях случайной сети:
  python swap_v2.py --acts acts_v2_rand0 --tag rand0 2>&1 | tee swap_v2_rand0_log.txt
  -> swap_v2_rand0.png, swap_v2_rand0.csv (настоящие swap_v2.* не перезаписываются)
"""

import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DATA, ACTS = "data_v2", "acts_v2"
P, G, N_LAYERS = 16, 14, 13
GROUPS = ["twins", "unchanged", "background", "cls"]
NAMES = {"twins": "близнецы", "unchanged": "нетронутые", "background": "фон", "cls": "CLS"}
N_BOOT = 1000


def cos_dist(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8)
    return np.maximum(1.0 - cos, 0.0)


def patch_groups(rec):
    """Номера ТОКЕНОВ (патч + 1) для близнецов, нетронутых объектов и фона."""
    sid = rec["id"]
    imgs = [np.asarray(Image.open(f"{DATA}/images/{sid}_{v}.png").convert("RGB")) for v in ("S", "Sp", "Sr")]
    mask = np.asarray(Image.open(f"{DATA}/masks/{sid}.png"))
    same, empty = np.zeros(G * G, bool), np.zeros(G * G, bool)
    for p in range(G * G):
        py, px = divmod(p, G)
        blk = (slice(py * P, (py + 1) * P), slice(px * P, (px + 1) * P))
        same[p] = np.array_equal(imgs[0][blk], imgs[1][blk]) and np.array_equal(imgs[0][blk], imgs[2][blk])
        empty[p] = (mask[blk] == 0).all()
    twins = set(rec["objects"][0]["patches16"]) | set(rec["objects"][1]["patches16"])
    others = set()
    for o in rec["objects"][2:]:
        others |= set(o["patches16"])
    return dict(twins=[p + 1 for p in sorted(twins)],
                unchanged=[p + 1 for p in sorted(others - twins) if same[p]],
                background=[p + 1 for p in range(G * G) if empty[p]],
                cls=[0])


def boot_ratio(sw, rc, rng):
    """Отношение средних swap/recolor и его 95% интервал (бутстрэп по сценам)."""
    sw, rc = np.asarray(sw), np.asarray(rc)
    if len(sw) == 0 or rc.mean() < 1e-6:     # на L00 вне близнецов оба ~0 — делить нечего
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, len(sw), size=(N_BOOT, len(sw)))
    boots = sw[idx].mean(1) / np.maximum(rc[idx].mean(1), 1e-12)
    return float(sw.mean() / rc.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    global ACTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default=ACTS, help="папка с активациями (по умолчанию acts_v2 — обученная DINO)")
    ap.add_argument("--tag", default="", help="приписка к именам выходных файлов, например rand0")
    args = ap.parse_args()
    ACTS = args.acts
    stem = "swap_v2" + (f"_{args.tag}" if args.tag else "")
    info = json.load(open(f"{ACTS}/info.json"))
    weights = info.get("weights", "обученные DINO")   # старые info.json (до 5.10) этого поля не имеют
    print(f"Активации: {ACTS}/   веса: {weights}   выход: {stem}.csv, {stem}.png")

    meta = [json.loads(line) for line in open(f"{DATA}/meta.jsonl")]
    rows = {(r["id"], r["version"]): int(r["row"]) for r in csv.DictReader(open(f"{ACTS}/index.csv"))}
    groups = [patch_groups(rec) for rec in meta]
    n_unch = sum(1 for g in groups if g["unchanged"])
    print(f"Сцен: {len(meta)}. С нетронутыми объектами (3–4 объекта): {n_unch}")

    rng = np.random.default_rng(0)
    out = {k: dict(sw=[], rc=[], ratio=[], lo=[], hi=[]) for k in GROUPS}
    for l in range(N_LAYERS):
        A = np.load(f"{ACTS}/L{l:02d}.npy", mmap_mode="r")
        per = {k: ([], []) for k in GROUPS}
        for rec, g in zip(meta, groups):
            h = {v: np.asarray(A[rows[(rec["id"], v)]]) for v in ("S", "Sp", "Sr")}
            for k in GROUPS:
                if g[k]:
                    per[k][0].append(cos_dist(h["S"][g[k]], h["Sp"][g[k]]).mean())
                    per[k][1].append(cos_dist(h["S"][g[k]], h["Sr"][g[k]]).mean())
        for k in GROUPS:
            sw, rc = per[k]
            r, lo, hi = boot_ratio(sw, rc, rng)
            for key, val in zip(["sw", "rc", "ratio", "lo", "hi"], [np.mean(sw), np.mean(rc), r, lo, hi]):
                out[k][key].append(float(val))

    # --- таблица
    print("\nКосинусное расстояние: swap = S vs Sp (только привязка), recolor = S vs Sr (набор признаков)")
    print("отношение = swap / recolor, в скобках 95% интервал\n")
    for k in GROUPS:
        print(f"== {NAMES[k]} ==")
        print("слой     swap     recolor   отношение")
        for l in range(N_LAYERS):
            o = out[k]
            if np.isnan(o["ratio"][l]):
                print(f"L{l:02d}  {o['sw'][l]:8.4f}  {o['rc'][l]:8.4f}          —")
            else:
                print(f"L{l:02d}  {o['sw'][l]:8.4f}  {o['rc'][l]:8.4f}   {o['ratio'][l]:5.2f}  [{o['lo'][l]:.2f}–{o['hi'][l]:.2f}]")
        print()

    z = max(out["unchanged"]["sw"][0], out["unchanged"]["rc"][0], out["background"]["sw"][0], out["background"]["rc"][0])
    print(f"Проверка L00 (нетронутые и фон должны быть 0): {z:.6f}  {'✅' if z < 1e-3 else '⚠️ пришли Claude весь вывод'}")

    # --- числа
    with open(f"{stem}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "layer", "swap", "recolor", "ratio", "ci_lo", "ci_hi"])
        for k in GROUPS:
            for l in range(N_LAYERS):
                o = out[k]
                w.writerow([k, f"L{l:02d}"] + [f"{o[x][l]:.5f}" for x in ["sw", "rc", "ratio", "lo", "hi"]])

    # --- график: слева расстояния, справа отношения с интервалами
    L = np.arange(N_LAYERS)
    labels = [f"L{l:02d}" for l in L]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    ax = axes[0]
    ax.plot(L, out["twins"]["sw"], "o-", c="C0", label="swap (привязка)")
    ax.plot(L, out["twins"]["rc"], "o--", c="C0", alpha=0.5, label="recolor (контроль)")
    ax.set_title("Близнецы (их пиксели меняются)")
    ax = axes[1]
    for k, c in [("unchanged", "C1"), ("background", "C2"), ("cls", "C4")]:
        ax.plot(L, out[k]["sw"], "o-", c=c, label=f"{NAMES[k]}: swap")
        ax.plot(L, out[k]["rc"], "o--", c=c, alpha=0.5, label=f"{NAMES[k]}: recolor")
    ax.set_title("Вне близнецов (пиксели те же!)")
    for ax in axes[:2]:
        ax.set_ylabel("косинусное расстояние от S")
        ax.legend(fontsize=7)
    ax = axes[2]
    for k, c in [("unchanged", "C1"), ("background", "C2"), ("cls", "C4")]:
        r = np.array(out[k]["ratio"])
        ax.plot(L, r, "o-", c=c, label=NAMES[k])
        ax.fill_between(L, out[k]["lo"], out[k]["hi"], color=c, alpha=0.15)
    ax.axhline(1, ls="--", c="gray", lw=1)
    ax.axhline(0, ls="--", c="gray", lw=1)
    ax.set_ylim(-0.05, 1.3)
    ax.set_ylabel("swap / recolor")
    ax.set_title("swap / recolor  (0 = мешок признаков)")
    ax.legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(L, labels, fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle(f"веса: {weights}", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=150)
    print(f"График: {stem}.png   Числа: {stem}.csv")


if __name__ == "__main__":
    main()

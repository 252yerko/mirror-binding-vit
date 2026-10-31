"""
swap_look.py — первый взгляд внутрь DINO: различает ли модель S и S′, и где.

ВОПРОС:
  S и S′ отличаются только привязкой (какой цвет к какой фигуре прилип).
  Видит ли DINO эту разницу? На каком слое? И в каких токенах?

ИДЕЯ — делим 196 патчей каждой сцены на три группы:

  1. ПЕРЕСТАВЛЕННЫЕ объекты — две фигуры, у которых поменялись цвета.
     Их пиксели в S и S′ разные, поэтому разница видна сразу, с L00.
     Это "эталон": так выглядит большая разница.

  2. НЕТРОНУТЫЕ объекты — остальные фигуры. Их пиксели в S и S′
     ОДИНАКОВЫЕ до последнего пикселя (берём только такие патчи).
     На L00 патч видит только себя, поэтому разница там ровно 0.
     Если на глубоких слоях разница появится — значит, токен нетронутого
     объекта "узнал" что-то о том, что случилось с ДРУГИМИ объектами.
     Это первый намёк, что информация о привязке расползается по сцене.

  3. ФОН — патчи, где нет ни одного объекта. Тоже одинаковые пиксели,
     на L00 разница 0. Тот же вопрос, что для группы 2.

  Плюс токен CLS — общий вектор всей картинки (на L00 он ещё не видел
  картинку и одинаков для всех сцен, поэтому там стоит прочерк):
     сравниваем, насколько CLS(S) далёк от CLS(S′), и насколько CLS(S)
     далёк от CLS совсем другой сцены. Отношение этих двух чисел:
       около 0 — для модели S и S′ почти одно и то же ("мешок признаков")
       около 1 — S и S′ для неё разные, как две разные сцены

КАК МЕРЯЕМ РАЗНИЦУ:
  Косинусное расстояние = 1 − косинус угла между двумя векторами.
  0 — векторы смотрят в одну сторону (одинаковые), чем больше — тем разнее.
  Берём косинус, а не простую разницу, потому что длина векторов растёт
  от слоя к слою, и простая разница была бы нечестной между слоями.

ЗАПУСК (после extract.py):
  python swap_look.py 2>&1 | tee swap_log.txt
  -> таблица в Терминале, график swap_look.png, числа в swap_look.csv
"""

import csv
import json
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DATA, ACTS = "data", "acts"
P, G, N_LAYERS = 16, 14, 13          # патч 16 px, сетка 14x14, слои L00..L12
GROUPS = ["swapped", "unchanged", "background"]
NAMES = {"swapped": "переставленные", "unchanged": "нетронутые", "background": "фон"}


def cos_dist(a, b):
    """Косинусное расстояние между векторами (работает и для строк матриц)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8)
    return np.maximum(1.0 - cos, 0.0)          # убираем "-0.0000" от округления


def patch_groups(rec):
    """Делит 196 патчей сцены на три группы. Возвращает номера ТОКЕНОВ (патч + 1, т.к. токен 0 — CLS)."""
    sid = rec["id"]
    s = np.asarray(Image.open(f"{DATA}/images/{sid}_S.png").convert("RGB"))
    sp = np.asarray(Image.open(f"{DATA}/images/{sid}_Sp.png").convert("RGB"))
    mask = np.asarray(Image.open(f"{DATA}/masks/{sid}.png"))

    same_pixels = np.zeros(G * G, bool)     # патч в S и S′ совпадает до пикселя
    empty = np.zeros(G * G, bool)           # в патче нет ни одного объекта
    for p in range(G * G):
        py, px = divmod(p, G)
        block = (slice(py * P, (py + 1) * P), slice(px * P, (px + 1) * P))
        same_pixels[p] = np.array_equal(s[block], sp[block])
        empty[p] = (mask[block] == 0).all()

    a, b = rec["swapped"]
    swapped = set(rec["objects"][a]["patches16"]) | set(rec["objects"][b]["patches16"])
    others = set()
    for i, o in enumerate(rec["objects"]):
        if i not in (a, b):
            others |= set(o["patches16"])
    unchanged = [p for p in sorted(others - swapped) if same_pixels[p]]
    background = [p for p in range(G * G) if empty[p]]
    return dict(swapped=[p + 1 for p in sorted(swapped)],
                unchanged=[p + 1 for p in unchanged],
                background=[p + 1 for p in background])


def main():
    meta = [json.loads(line) for line in open(f"{DATA}/meta.jsonl")]
    rows = {(r["id"], r["version"]): int(r["row"]) for r in csv.DictReader(open(f"{ACTS}/index.csv"))}
    groups = [patch_groups(rec) for rec in meta]
    n_unch = sum(1 for g in groups if g["unchanged"])
    print(f"Сцен: {len(meta)}. Сцен с нетронутыми объектами (3+ объекта): {n_unch}")

    # для каждой сцены — случайная ДРУГАЯ сцена (эталон "совсем разные картинки")
    rng = random.Random(0)
    other = []
    for i in range(len(meta)):
        j = rng.randrange(len(meta) - 1)
        other.append(j + (j >= i))

    res = {k: [] for k in GROUPS + ["cls_pair", "cls_other"]}
    for l in range(N_LAYERS):
        A = np.load(f"{ACTS}/L{l:02d}.npy", mmap_mode="r")
        acc = {k: [] for k in GROUPS}
        cls_pair, cls_other = [], []
        for i, (rec, g) in enumerate(zip(meta, groups)):
            hs = np.asarray(A[rows[(rec["id"], "S")]])
            hsp = np.asarray(A[rows[(rec["id"], "Sp")]])
            for k in GROUPS:
                if g[k]:
                    acc[k].append(cos_dist(hs[g[k]], hsp[g[k]]).mean())
            cls_pair.append(cos_dist(hs[0], hsp[0]))
            h_other = np.asarray(A[rows[(meta[other[i]]["id"], "S")], 0])
            cls_other.append(cos_dist(hs[0], h_other))
        for k in GROUPS:
            res[k].append(float(np.mean(acc[k])))
        res["cls_pair"].append(float(np.mean(cls_pair)))
        res["cls_other"].append(float(np.mean(cls_other)))
    # на L00 CLS ещё не видел картинку (он одинаковый для всех) -> отношение не определено
    ratio = [p / o if o > 1e-6 else float("nan") for p, o in zip(res["cls_pair"], res["cls_other"])]

    # --- таблица
    print("\nКосинусное расстояние S vs S′ (0 = одинаково, больше = разнее)\n")
    print("слой   переставл.  нетронутые   фон      | CLS: S/S′   CLS: чужая   отношение")
    for l in range(N_LAYERS):
        print(f"L{l:02d}   {res['swapped'][l]:9.4f}  {res['unchanged'][l]:10.4f}  {res['background'][l]:8.4f}"
              f"   | {res['cls_pair'][l]:9.4f}  {res['cls_other'][l]:11.4f}  "
              + ("        —" if np.isnan(ratio[l]) else f"{ratio[l]:9.2f}"))

    # --- самопроверка: на L00 нетронутые патчи и фон обязаны совпадать
    z = max(res["unchanged"][0], res["background"][0])
    ok = z < 1e-3
    print(f"\nПроверка L00 (нетронутые и фон должны быть 0): {z:.6f}  {'✅' if ok else '⚠️ проверь весь вывод'}")

    # --- числа в файл
    with open("swap_look.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["layer"] + GROUPS + ["cls_pair", "cls_other", "cls_ratio"])
        for l in range(N_LAYERS):
            w.writerow([f"L{l:02d}"] + [f"{res[k][l]:.5f}" for k in GROUPS]
                       + [f"{res['cls_pair'][l]:.5f}", f"{res['cls_other'][l]:.5f}", f"{ratio[l]:.4f}"])

    # --- график
    L = np.arange(N_LAYERS)
    labels = [f"L{l:02d}" for l in L]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.3))
    for k, m in zip(GROUPS, ["o", "s", "^"]):
        ax1.plot(L, res[k], m + "-", label=NAMES[k])
    ax1.set_xticks(L, labels, fontsize=8)
    ax1.set_ylabel("косинусное расстояние S vs S′")
    ax1.set_title("Токены патчей: насколько S и S′ разные")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.plot(L, ratio, "o-", c="purple")
    ax2.axhline(1, ls="--", c="gray", lw=1)
    ax2.text(0, 1.02, "как две разные сцены", fontsize=8, color="gray")
    ax2.axhline(0, ls="--", c="gray", lw=1)
    ax2.text(0, 0.02, "одно и то же", fontsize=8, color="gray")
    ax2.set_xticks(L, labels, fontsize=8)
    ax2.set_ylim(-0.05, max(1.1, np.nanmax(ratio) * 1.1))
    ax2.set_ylabel("CLS: расстояние(S, S′) / расстояние(S, чужая)")
    ax2.set_title("Общий вектор картинки (CLS)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("swap_look.png", dpi=150)
    print("График: swap_look.png   Числа: swap_look.csv")


if __name__ == "__main__":
    main()

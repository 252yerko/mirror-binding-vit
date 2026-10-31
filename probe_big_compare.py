"""
probe_big_compare.py — MAE против I-JEPA: следует ли потеря «левого» за слепотой к зеркалу (добавлено 22.10).

Берёт баллы сцен из probe_big_mae_scores.npz и probe_big_ijepa_scores.npz (их пишет probe_big.py)
и сравнивает ОДИНАКОВОЕ С ОДИНАКОВЫМ: дальний фон, последний слой L32, логрег, те же сцены.

ПРАВИЛО ВЫВОДА (записано ДО запуска, реестр 39, 22.10):
  Разрыв = точность «верхний» − точность «левый» (п.п.).
  Δ = разрыв(I-JEPA) − разрыв(MAE); 95% интервал — парный бутстрэп по сценам (2000 повторов, seed 0).
  «СЛЕДУЕТ»      — весь интервал Δ > 0: I-JEPA (слепее к зеркалу: 0,40 против 0,63) теряет «левый» сильнее.
  «ПРОТИВОРЕЧИТ» — весь интервал Δ < 0.
  «НЕ РАЗЛИЧАЕТ» — интервал задевает 0.
СТАВКА ЕРНАЗАРА (реестр 39, 22.10, 19:47 — до запуска; правило засчёта записано тогда же):
  разрыв MAE 8,8–9,0 (округление до 0,1, включительно); I-JEPA 16,0–18,0 («примерно 17»); вывод «следует».

ЗАПУСК (после обоих прогонов probe_big.py):
  python probe_big_compare.py 2>&1 | tee probe_big_compare_log.txt
"""
import argparse
import os

import numpy as np

N_BOOT = 2000
BET = {"MAE": (8.8, 9.0), "I-JEPA": (16.0, 18.0), "verdict": "СЛЕДУЕТ"}
CONTEXT = "DINO 18,6 (слепота 0,28); CLIP 9,0 (слепота 0,61) — дальний фон, их последний слой L12 (реестр 25)"


def load(path):
    z = np.load(path, allow_pickle=False)
    nets, groups, queries, layers = list(z["nets"]), list(z["groups"]), list(z["queries"]), list(z["layers"])
    s = z["logreg"]                     # [сети, вопросы, группы, слои, сцены]
    return z["ids"], nets, groups, queries, layers, s, list(z["weights"])


def pick(s, nets, groups, queries, layers, net, group, query, layer):
    return s[nets.index(net), queries.index(query), groups.index(group), layers.index(layer)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mae", default="probe_big_mae_scores.npz")
    ap.add_argument("--ijepa", default="probe_big_ijepa_scores.npz")
    args = ap.parse_args()
    for f in (args.mae, args.ijepa):
        assert os.path.exists(f), f"нет файла {f} — сначала запусти probe_big.py --model mae и --model ijepa"
    A = {"MAE": load(args.mae), "I-JEPA": load(args.ijepa)}
    ids_m, ids_j = A["MAE"][0], A["I-JEPA"][0]
    ok = len(ids_m) == len(ids_j) and bool((ids_m == ids_j).all())
    print(f"Проверка: одни и те же сцены в одном порядке: {len(ids_m)} сцен  {'✅' if ok else '⚠️ НЕТ — проверь весь вывод'}")
    assert ok
    L = max(A["MAE"][4])
    assert L == max(A["I-JEPA"][4]) == 32, "ожидался последний слой L32 у обеих"

    gap, rand_gaps = {}, {}
    for name, (ids, nets, groups, queries, layers, s, weights) in A.items():
        assert nets[0] == name, f"первая сеть в файле — {nets[0]}, ожидалась {name}"
        print(f"{name}: {weights[0]}")
        up = pick(s, nets, groups, queries, layers, name, "far_bg", "upper", L)
        lf = pick(s, nets, groups, queries, layers, name, "far_bg", "left", L)
        gap[name] = (up - lf, up, lf)
        rand_gaps[name] = [float(np.nanmean(pick(s, nets, groups, queries, layers, n, "far_bg", "upper", L)
                                            - pick(s, nets, groups, queries, layers, n, "far_bg", "left", L))) * 100
                           for n in nets[1:]]

    dm, dj = gap["MAE"][0], gap["I-JEPA"][0]
    keep = ~np.isnan(dm) & ~np.isnan(dj)
    print(f"Сцен с дальним фоном у обеих сетей: {int(keep.sum())}")
    dm, dj = dm[keep], dj[keep]
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(dm), size=(N_BOOT, len(dm)))
    bm, bj = dm[idx].mean(1), dj[idx].mean(1)
    bd = bj - bm

    print(f"\nДальний фон, L{L:02d}, логрег. Разрыв «верхний − левый», п.п. [95% инт., парный бутстрэп по сценам]:")
    for name, d, b in (("MAE", dm, bm), ("I-JEPA", dj, bj)):
        up, lf = gap[name][1][keep], gap[name][2][keep]
        print(f"  {name:7s} верхний {up.mean() * 100:5.1f}%   левый {lf.mean() * 100:5.1f}%   разрыв {d.mean() * 100:5.1f} "
              f"[{np.percentile(b, 2.5) * 100:.1f} … {np.percentile(b, 97.5) * 100:.1f}]   "
              f"случайные: {min(rand_gaps[name]):.1f} … {max(rand_gaps[name]):.1f}")
    print(f"  для сравнения: {CONTEXT}")

    D, lo, hi = (dj.mean() - dm.mean()) * 100, np.percentile(bd, 2.5) * 100, np.percentile(bd, 97.5) * 100
    verdict = "СЛЕДУЕТ" if lo > 0 else ("ПРОТИВОРЕЧИТ" if hi < 0 else "НЕ РАЗЛИЧАЕТ")
    print(f"\nΔ = разрыв(I-JEPA) − разрыв(MAE) = {D:.1f} [{lo:.1f} … {hi:.1f}] п.п.")
    print(f"ВЫВОД ПО ПРАВИЛУ (реестр 39, записано до запуска): «{verdict}»")
    meaning = {"СЛЕДУЕТ": "сеть, более слепая к зеркалу (I-JEPA), сильнее теряет «левый» — связь держится на 4 моделях",
               "ПРОТИВОРЕЧИТ": "более слепая к зеркалу I-JEPA теряет «левый» МЕНЬШЕ, чем MAE — связь «слепота → потеря» не держится",
               "НЕ РАЗЛИЧАЕТ": "разница между MAE и I-JEPA не установлена: интервал задевает 0"}
    print(f"  значит: {meaning[verdict]}")
    print("  Оговорки: одна пара моделей; MAE и I-JEPA различаются целью обучения; корреляция, не причина.")

    print("\nСТАВКА ЕРНАЗАРА (реестр 39, 22.10 19:47, до запуска):")
    for name in ("MAE", "I-JEPA"):
        g = round(float(gap[name][0][keep].mean() * 100), 1)
        a, b = BET[name]
        print(f"  {name:7s} диапазон {a:.1f}–{b:.1f} -> вышло {g:.1f}  {'ПОПАЛ ✅' if a <= g <= b else 'НЕ ПОПАЛ ❌'}")
    print(f"  вывод «{BET['verdict'].lower()}» -> вышло «{verdict.lower()}»  {'ПОПАЛ ✅' if verdict == BET['verdict'] else 'НЕ ПОПАЛ ❌'}")

    # справка (не правило): по слоям
    print("\nСправка (не правило): разрыв по слоям, дальний фон, логрег, п.п.")
    layers = A["MAE"][4]
    print("слой    MAE   I-JEPA   Δ")
    for l in layers:
        g2 = {}
        for name, (ids, nets, groups, queries, lay, s, _) in A.items():
            g2[name] = np.nanmean(pick(s, nets, groups, queries, lay, name, "far_bg", "upper", l)
                                  - pick(s, nets, groups, queries, lay, name, "far_bg", "left", l)) * 100
        print(f"L{l:02d}  {g2['MAE']:5.1f}  {g2['I-JEPA']:6.1f}  {g2['I-JEPA'] - g2['MAE']:5.1f}")


if __name__ == "__main__":
    main()

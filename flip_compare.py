"""
flip_compare.py — MAE против I-JEPA: какая причина слепоты к зеркалу (добавлено 15.10).

Берёт расстояния по каждой сцене из flip_test_mae_dist.npz и flip_test_ijepa_dist.npz
(их пишет flip_test.py --model mae / --model ijepa) и сравнивает ОДИНАКОВОЕ С ОДИНАКОВЫМ:
«среднее по патчам», последний слой L32 (у I-JEPA нет CLS).

ПРАВИЛО ВЫВОДА (записано 15.10 ДО запуска, то же — в шапке flip_test.py):
  Δ = отношение(I-JEPA) − отношение(MAE); отношение = среднее d(зеркало ←→) / среднее d(зеркало ↑↓).
  95% интервал Δ — парный бутстрэп по сценам (одни и те же сцены для обеих сетей, 2000 повторов, seed 0).
  «А»       : нижняя граница Δ > 0,10        — MAE заметно слепее I-JEPA: достаточно ВИДЕТЬ отражения.
  «Б»       : весь интервал Δ внутри [−0,10; 0,10] — разницы нет: одних отражений мало,
              нужна задача «одинаковый ответ для картинки и её зеркала» (как у DINO).
  «Против А»: верхняя граница Δ < −0,10       — MAE даже МЕНЕЕ слеп, чем I-JEPA.
  иначе — «не различает».
  Оговорка: MAE и I-JEPA различаются ещё и целью (пиксели против описаний) — это не чистый опыт.

ЗАПУСК (из папки binding-vit, после обоих прогонов flip_test.py):
  python flip_compare.py 2>&1 | tee flip_compare_log.txt
"""
import csv
import os

import numpy as np

THR, N_BOOT = 0.10, 2000
FILES = {"MAE": "flip_test_mae_dist.npz", "I-JEPA": "flip_test_ijepa_dist.npz"}


def load(path, name):
    z = np.load(path, allow_pickle=False)
    models, groups = list(z["models"]), list(z["groups"])
    gi = groups.index("mean")
    d = z["dist"][:, :, gi]                       # [модели, 2 (Г/В), слои, сцены]
    return z["ids"], models, d


def ratio(dh, dv):
    return dh.mean(-1) / dv.mean(-1)


def main():
    for f in FILES.values():
        assert os.path.exists(f), f"нет файла {f} — сначала запусти flip_test.py --model mae и --model ijepa"
    (ids_m, mod_m, dm), (ids_j, mod_j, dj) = load(FILES["MAE"], "MAE"), load(FILES["I-JEPA"], "I-JEPA")
    ok = len(ids_m) == len(ids_j) and bool((ids_m == ids_j).all())
    print(f"Проверка: одни и те же сцены в одном порядке у MAE и I-JEPA: {len(ids_m)} сцен  {'✅' if ok else '⚠️ НЕТ — пришли Claude весь вывод'}")
    assert ok
    assert mod_m[0] == "MAE" and mod_j[0] == "I-JEPA", (mod_m, mod_j)
    nl = dm.shape[2]
    assert dj.shape[2] == nl, "разное число слоёв"
    L = nl - 1
    mh, mv = dm[0, 0, L], dm[0, 1, L]
    jh, jv = dj[0, 0, L], dj[0, 1, L]
    r_m, r_j = ratio(mh, mv), ratio(jh, jv)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(mh), size=(N_BOOT, len(mh)))
    b_m = mh[idx].mean(1) / mv[idx].mean(1)
    b_j = jh[idx].mean(1) / jv[idx].mean(1)
    bd = b_j - b_m
    d, lo, hi = r_j - r_m, float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))
    ci = lambda b: (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))
    print(f"\nСреднее по патчам, L{L:02d}, отношение d(←→)/d(↑↓):")
    print(f"  MAE    {r_m:.3f} [{ci(b_m)[0]:.3f}–{ci(b_m)[1]:.3f}]")
    print(f"  I-JEPA {r_j:.3f} [{ci(b_j)[0]:.3f}–{ci(b_j)[1]:.3f}]")
    rnd_m = [ratio(dm[i, 0, L], dm[i, 1, L]) for i in range(1, len(mod_m))]
    rnd_j = [ratio(dj[i, 0, L], dj[i, 1, L]) for i in range(1, len(mod_j))]
    print(f"  случайные: MAE-архитектура {min(rnd_m):.2f}–{max(rnd_m):.2f}, I-JEPA-архитектура {min(rnd_j):.2f}–{max(rnd_j):.2f}")
    print(f"\nΔ = I-JEPA − MAE = {d:+.3f}  [95% инт. {lo:+.3f} … {hi:+.3f}]   порог ±{THR:.2f}")
    if lo > THR:
        v = "«А» — MAE заметно слепее I-JEPA: похоже, достаточно ВИДЕТЬ отражения при обучении"
    elif lo >= -THR and hi <= THR:
        v = "«Б» — разницы нет: одних отражений мало; слепота — от задачи «одинаковый ответ» (как у DINO)"
    elif hi < -THR:
        v = "«ПРОТИВ А» — MAE даже менее слеп, чем I-JEPA"
    else:
        v = "НЕ РАЗЛИЧАЕТ — интервал слишком широкий или задевает порог"
    print(f"ВЫВОД по правилу (записано до запуска): {v}")

    print("\nСправка по слоям (среднее по патчам): слой  MAE  I-JEPA  Δ")
    for l in range(0, nl, 4):
        a, b = ratio(dm[0, 0, l], dm[0, 1, l]), ratio(dj[0, 0, l], dj[0, 1, l])
        print(f"  L{l:02d}  {a:5.2f}  {b:5.2f}  {b - a:+.2f}")
    ref = []
    for f, n in (("flip_test.csv", "DINO"), ("flip_test_clip.csv", "CLIP")):
        if os.path.exists(f):
            for row in csv.DictReader(open(f)):
                if row["model"] == n and row["group"] == "mean" and row["layer"] == "L12":
                    ref.append(f"{n} {float(row['ratio_h_v']):.2f}")
    if ref:
        print("Для сравнения (среднее по патчам, последний слой L12, ViT-B/16): " + ", ".join(ref))


if __name__ == "__main__":
    main()

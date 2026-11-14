"""
photos_explore_c1.py — опыт C (OSF dprnk): РАЗВЕДОЧНЫЙ разбор C1 после запуска. Не правило, не регистрировалось.
Добавлено 15.11 (реестр 169).

ЗАЧЕМ:
  Зеркальное отношение — дробь d(←→) / d(↑↓). Маленькое отношение может значить слабую реакцию на отражение
  слева направо (числитель) или сильную реакцию на отражение сверху вниз (знаменатель). У DINO B/16 отношение упало
  с 0,284 (синтетика) до 0,102 (аэрофото): здесь числитель и знаменатель печатаются отдельно для всех сетей.

ЧТО СЧИТАЕТСЯ: по кэшу photos.py --c1 (среднее по патчам, последний слой; округление кэша — float16, как в правилах):
  средние косинусные расстояния d(←→) и d(↑↓) на аэрофото и COCO (и на синтетике, где она сохранена в кэше),
  и их отношение (должно совпасть с photos_c1.csv).
  Абсолютные расстояния между разными наборами картинок сравнивать осторожно: у наборов разный разброс содержания.

ЗАПУСК (из папки binding-vit):  python3 photos_explore_c1.py 2>&1 | tee photos_explore_c1_log.txt   (< 1 мин, нужен только numpy)
"""
import csv
import os
import sys

import numpy as np

CACHE = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/binding-vit-data/photos/cache")


def cos_dist(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return 1.0 - (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)


ref = {}
for r in csv.DictReader(open("photos_c1.csv", encoding="utf-8")):
    ref[(r["name"], r["seed"], r["set"])] = float(r["ratio"])
names = []
for r in csv.DictReader(open("photos_c1.csv", encoding="utf-8")):
    if (r["name"], r["seed"]) not in names:
        names.append((r["name"], r["seed"]))
print("Опыт C, C1, РАЗВЕДОЧНО (после запуска, не правило регистрации dprnk): d(←→) и d(↑↓) по отдельности\n")
print(f"{'сеть':30s} {'набор':7s} {'d(←→)':>8s} {'d(↑↓)':>8s} {'отнош.':>7s}  совпадает с photos_c1.csv")
ok = True
for name, seed in names:
    safe = name.replace("/", "_").replace(" ", "_") + ("" if seed == "" else f"_rand{seed}")
    z = np.load(os.path.join(CACHE, f"c1_{safe}.npz"))
    tag = name + ("" if seed == "" else f" [случ. {seed}]")
    for k in ("synth", "aerial", "coco"):
        if f"PM_{k}" not in z.files:
            continue
        P = z[f"PM_{k}"].astype(np.float32)
        dh, dv = cos_dist(P[:, 0], P[:, 1]).mean(), cos_dist(P[:, 0], P[:, 2]).mean()
        same = "—" if k == "synth" else ("✅" if abs(dh / dv - ref[(name, seed, k)]) < 1e-6 else "⚠️")
        ok &= same != "⚠️"
        print(f"{tag:30s} {k:7s} {dh:8.4f} {dv:8.4f} {dh / dv:7.3f}  {same}")
print(f"\nОтношения совпадают с photos_c1.csv: {'✅' if ok else '⚠️ проверь весь вывод'}")
print("Разведочно, после запуска; не правило регистрации dprnk.")

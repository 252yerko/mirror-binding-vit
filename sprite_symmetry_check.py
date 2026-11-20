"""
sprite_symmetry_check.py — проверка стимулов (16.11, реестр 235): симметричны ли «штампы» фигур слева направо на уровне пикселей.

ЗАЧЕМ: для опыта E (условие «отражена ли картинка?») нужно знать, оставляет ли сам генератор сцен след отражения.
ЧТО ДЕЛАЕТ: для каждой формы и каждого радиуса, встречающегося в data_v2 и data_g, рисует штамп (generate_v2.sprite),
отражает его слева направо и сравнивает с исходным при сдвигах от −3 до +3 пикселей. Если при каком-то сдвиге совпадение
полное — фигура зеркально симметрична (ось просто не в центре массива). Сети не запускаются. Это не результат опыта.
"""
import json, sys
import numpy as np
sys.path.insert(0, ".")
from generate_v2 import sprite
from generate import SHAPES

rs = {s: set() for s in SHAPES}
for f in ("data_v2/meta.jsonl", "data_g/meta.jsonl"):
    for line in open(f):
        for o in json.loads(line)["objects"]:
            rs[o["shape"]].add(o["r"])
print("Проверка стимулов: зеркальная симметрия штампов слева направо (data_v2 + data_g)")
for s in SHAPES:
    bad, mx, shifts = 0, 0, {}
    for r in sorted(rs[s]):
        a = np.pad(np.asarray(sprite(s, r)[0]) > 0, ((0, 0), (3, 3)))
        f = a[:, ::-1]
        d, k = min(((np.roll(f, k, axis=1) != a).sum(), k) for k in range(-3, 4))
        shifts[k] = shifts.get(k, 0) + 1
        bad += d > 0
        mx = max(mx, int(d))
    print(f"  {s:9s}: радиусов {len(rs[s])}; не симметричны ни при каком сдвиге: {bad}; "
          f"наибольшее число отличающихся пикселей: {mx}; лучший сдвиг: {dict(sorted(shifts.items()))}")

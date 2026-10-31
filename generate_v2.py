"""
generate_v2.py — генератор сцен v2: ЧИСТЫЙ тест привязки + контроль.

ЧЕМ ОТЛИЧАЕТСЯ ОТ v1:
  В v1 менялись цветами две СЛУЧАЙНЫЕ фигуры — например, красная звезда и
  синий круг. Тогда менялась не только привязка, но и "сколько красного"
  на картинке и сам набор объектов (красная звезда -> синяя звезда).
  Первый взгляд (swap_look.py) показал, что это мешает.

  В v2 в каждой сцене есть ПАРА-БЛИЗНЕЦЫ: два объекта одной формы
  и ОДНОГО размера (например, две звезды по 40 px), разных цветов.
  Остальные объекты — других форм.

ДЛЯ КАЖДОЙ СЦЕНЫ ТРИ КАРТИНКИ:
  S   — исходная сцена.
  Sp  — близнецы поменялись цветами (swap).
        Набор объектов тот же ("красная звезда + синяя звезда"),
        количество каждого цвета то же ДО ПИКСЕЛЯ.
        Изменилась ТОЛЬКО привязка: какая из двух звёзд какого цвета.
  Sr  — контроль (recolor): близнецы перекрашены в два НОВЫХ цвета,
        которых в сцене не было. Меняются те же пиксели, что в Sp,
        но здесь меняется набор признаков, а привязка в смысле
        "перестановки" — нет.

  Зачем контроль: если модель — "мешок признаков", то S и Sp для неё
  одинаковые, а S и Sr — разные. Если модель хранит привязку, она
  различает и то, и другое. Сравнение Sp и Sr — главный приём v2.

ПОЧЕМУ ПИКСЕЛЬ В ПИКСЕЛЬ:
  Каждую фигуру рисуем один раз как "штамп" и ставим его в целые
  координаты. Тогда близнецы состоят из одних и тех же пикселей,
  просто сдвинутых, и площадь у них совпадает точно.
  Генератор это проверяет для каждой сцены.

ОГРАНИЧЕНИЕ: 2–4 объекта. Цветов всего 6, все цвета в сцене разные,
  и для контроля нужно ещё 2 свободных цвета: 4 + 2 = 6.

СОХРАНЯЕТ (в папку data_v2/):
  images/0000_S.png, 0000_Sp.png, 0000_Sr.png
  masks/0000.png      — карта объектов (одна на все три версии)
  meta.jsonl          — описание сцен; поле "versions" = ["S","Sp","Sr"]
  preview.png         — 12 строк S | Sp | Sr

ЗАПУСК:
  python generate_v2.py              # 2000 сцен -> data_v2/
  python generate_v2.py --n 50       # быстрая проба
"""

import argparse
import json
import math
import os
import random

import numpy as np
from PIL import Image

# берём всё общее из v1, чтобы цвета, формы и рисование были ТЕ ЖЕ САМЫЕ
from generate import BG_GRAY, COLORS, OBJ_SIZE, MIN_GAP, SHAPES, SIZE, patches_of, shape_polygon
from PIL import ImageDraw

SET_SIZES_V2 = [2, 3, 4]
REACH = 1.22          # во сколько раз самая дальняя точка фигуры дальше r (угол квадрата)
VERSIONS = ["S", "Sp", "Sr"]


def place_with_radii(radii, rng, tries=2000):
    """Как place_objects из v1, но размеры заданы заранее, а центры — целые числа."""
    for _ in range(tries):
        placed = []
        for r in radii:
            for _ in range(300):
                lo, hi = math.ceil(REACH * r + 2), math.floor(SIZE - REACH * r - 2)
                cx, cy = rng.randint(lo, hi), rng.randint(lo, hi)
                # 1.22*r: угол квадрата торчит дальше r (0.86*r*sqrt(2)); считаем по нему,
                # иначе углы квадратов могут налезать друг на друга (нашли на сцене 550)
                if all(math.hypot(cx - p[0], cy - p[1]) > REACH * (r + p[2]) + MIN_GAP for p in placed):
                    placed.append((cx, cy, r))
                    break
        if len(placed) == len(radii):
            return placed
    raise RuntimeError("не удалось разместить объекты")


def make_scene(n, rng):
    """Объекты 0 и 1 — близнецы (одна форма, один размер). Остальные — другие формы."""
    twin_shape = rng.choice(SHAPES)
    other_shapes = rng.sample([s for s in SHAPES if s != twin_shape], n - 2)
    shapes = [twin_shape, twin_shape] + other_shapes

    colors_all = rng.sample(list(COLORS), n + 2)       # все разные
    colors, new_colors = colors_all[:n], colors_all[n:]  # 2 свободных цвета для контроля

    r_twin = rng.uniform(*OBJ_SIZE) / 2
    radii = [r_twin, r_twin] + [rng.uniform(*OBJ_SIZE) / 2 for _ in range(n - 2)]
    pos = place_with_radii(radii, rng)

    objs = [dict(shape=s, color=c, cx=p[0], cy=p[1], r=p[2]) for s, c, p in zip(shapes, colors, pos)]
    objs_sp = [dict(o) for o in objs]
    objs_sp[0]["color"], objs_sp[1]["color"] = objs[1]["color"], objs[0]["color"]
    objs_sr = [dict(o) for o in objs]
    objs_sr[0]["color"], objs_sr[1]["color"] = new_colors
    return objs, objs_sp, objs_sr


def sprite(shape, r):
    """Рисует фигуру ОДИН РАЗ в маленьком квадрате — "штамп".
    Зачем: если рисовать фигуру прямо на картинке в разных местах, компьютер
    округляет дробные координаты чуть по-разному, и у близнецов может
    отличаться пара пикселей (так и случилось в первом запуске, сцена 550).
    Со штампом обе фигуры состоят из ОДНИХ И ТЕХ ЖЕ пикселей, просто сдвинутых."""
    R = math.ceil(r) + 1
    m = Image.new("L", (2 * R + 1, 2 * R + 1), 0)
    d = ImageDraw.Draw(m)
    poly = shape_polygon(shape, R, R, r)
    if poly is None:
        d.ellipse([R - r, R - r, R + r, R + r], fill=255)
    else:
        d.polygon(poly, fill=255)
    return m, R


def draw_scene_v2(objects):
    """Как draw_scene из v1, но каждая фигура ставится штампом в целые координаты."""
    img = Image.new("RGB", (SIZE, SIZE), (BG_GRAY,) * 3)
    mask = Image.new("L", (SIZE, SIZE), 0)
    for i, o in enumerate(objects):
        m, R = sprite(o["shape"], o["r"])
        xy = (o["cx"] - R, o["cy"] - R)
        img.paste(Image.new("RGB", m.size, COLORS[o["color"]]), xy, m)
        mask.paste(Image.new("L", m.size, i + 1), xy, m)
    return img, mask


def color_counts(img):
    """Сколько пикселей каждого цвета на картинке."""
    arr = np.asarray(img).reshape(-1, 3)
    vals, cnt = np.unique(arr, axis=0, return_counts=True)
    return {tuple(v): int(c) for v, c in zip(vals, cnt)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default="data_v2")
    ap.add_argument("--seed", type=int, default=1)   # другой seed, чем у v1
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(f"{args.out}/images", exist_ok=True)
    os.makedirs(f"{args.out}/masks", exist_ok=True)

    with open(f"{args.out}/meta.jsonl", "w") as meta:
        for k in range(args.n):
            n = rng.choice(SET_SIZES_V2)
            objs, objs_sp, objs_sr = make_scene(n, rng)
            img, mask = draw_scene_v2(objs)
            img_sp, _ = draw_scene_v2(objs_sp)
            img_sr, _ = draw_scene_v2(objs_sr)

            # --- проверки: близнецы одинаковой площади, в Sp набор цветов совпадает до пикселя
            m = np.asarray(mask)
            for i, o in enumerate(objs):    # никто ни на кого не налез: площадь на картинке = площади штампа
                assert (m == i + 1).sum() == (np.asarray(sprite(o["shape"], o["r"])[0]) > 0).sum(), \
                    f"сцена {k}: объект {i} перекрыт другим"
            assert (m == 1).sum() == (m == 2).sum(), f"сцена {k}: у близнецов разная площадь"
            assert color_counts(img) == color_counts(img_sp), f"сцена {k}: в S и Sp разное количество цветов"

            sid = f"{k:04d}"
            img.save(f"{args.out}/images/{sid}_S.png")
            img_sp.save(f"{args.out}/images/{sid}_Sp.png")
            img_sr.save(f"{args.out}/images/{sid}_Sr.png")
            mask.save(f"{args.out}/masks/{sid}.png")

            record = dict(
                id=sid, set_size=n, versions=VERSIONS, twins=[0, 1], twin_shape=objs[0]["shape"],
                swapped=[0, 1],          # то же имя поля, что в v1 — для совместимости
                objects=[dict(shape=o["shape"], color=o["color"], color_Sp=p["color"], color_Sr=q["color"],
                              r=round(o["r"], 2), center=[o["cx"], o["cy"]],
                              bbox=[round(o["cx"] - o["r"]), round(o["cy"] - o["r"]),
                                    round(o["cx"] + o["r"]), round(o["cy"] + o["r"])],
                              patches16=patches_of(m, i + 1))
                         for i, (o, p, q) in enumerate(zip(objs, objs_sp, objs_sr))],
            )
            meta.write(json.dumps(record) + "\n")
            if (k + 1) % 250 == 0:
                print(f"  {k + 1}/{args.n} сцен")

    make_preview(args.out, min(12, args.n))
    print(f"Готово: {args.n} сцен x 3 картинки в папке '{args.out}/'.")
    print("Проверки пройдены: фигуры не налезают друг на друга, у близнецов одинаковая площадь, в S и Sp одинаковое количество каждого цвета.")
    print(f"Открой {args.out}/preview.png: слева S, в центре Sp (swap), справа Sr (контроль).")


def make_preview(out, k):
    pad = 8
    sheet = Image.new("RGB", (3 * SIZE + 4 * pad, k * (SIZE + pad) + pad), (255, 255, 255))
    for i in range(k):
        y = pad + i * (SIZE + pad)
        for j, v in enumerate(VERSIONS):
            sheet.paste(Image.open(f"{out}/images/{i:04d}_{v}.png"), (pad + j * (SIZE + pad), y))
    sheet.save(f"{out}/preview.png")


if __name__ == "__main__":
    main()

"""
generate.py — генератор сцен для исследования binding в Vision Transformers.

ЧТО ДЕЛАЕТ:
  Рисует картинки 224x224: несколько цветных фигур на тёмно-сером фоне.
  Для каждой сцены S рисует её "близнеца" S' (swap-пару): всё то же самое,
  но цвета ДВУХ объектов переставлены местами.

  В S и S' одинаковые: позиции, формы, размеры, набор цветов на картинке.
  Отличается ТОЛЬКО то, какой цвет к какой фигуре привязан.
  Поэтому любое различие, которое модель "видит" между S и S', — это
  по построению информация о привязке атрибутов (binding), а не о признаках.

ЧТО СОХРАНЯЕТ (в папку data/):
  images/0000_S.png, 0000_Sp.png   — сама сцена и её swap-близнец
  masks/0000.png                   — карта объектов: в пикселях объекта i стоит число i+1,
                                     в фоне 0. Нужна, чтобы потом понять, какие патчи
                                     модели принадлежат какому объекту
  meta.jsonl                       — описание каждой сцены (формы, цвета, рамки, условия)
  preview.png                      — сводная картинка: 12 пар S | S' для проверки глазами

ЗАПУСК:
  python generate.py                 # 200 пар (для проверки)
  python generate.py --n 5000        # большой набор для экспериментов
"""

import argparse
import colorsys
import json
import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw

# ------------------------------------------------------------------
# 1. Константы
# ------------------------------------------------------------------
SIZE = 224                      # размер картинки, как у ViT-B/16 (14x14 патчей по 16 px)
BG_GRAY = 60                    # тёмно-серый фон (яркость заметно ниже яркости фигур)
SHAPES = ["circle", "square", "triangle", "cross", "star"]
HUES = {                        # 6 цветов, заданы оттенком (hue) в градусах
    "red": 0, "yellow": 55, "green": 120,
    "cyan": 185, "blue": 230, "magenta": 300,
}
TARGET_LUMINANCE = 0.20         # все цвета подгоняются к одной яркости (см. ниже)
SET_SIZES = [2, 3, 4, 6, 8]     # сколько объектов на сцене
OBJ_SIZE = (34, 44)             # диаметр фигуры в пикселях: случайно от 34 до 44
MIN_GAP = 6                     # минимальный зазор между фигурами, px


# ------------------------------------------------------------------
# 2. Цвета одинаковой яркости
# ------------------------------------------------------------------
# Зачем: если, например, жёлтый ярче синего, модель может различать
# объекты по яркости, а не по цвету. Рецензент это сразу заметит.
# Поэтому для каждого оттенка подбираем насыщенность и яркость так,
# чтобы относительная яркость (luminance) у всех цветов была одинаковой.

def _lin(c):
    """sRGB -> линейная яркость (стандартная формула)."""
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def luminance(rgb01):
    r, g, b = (_lin(c) for c in rgb01)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def matched_color(hue_deg, target=TARGET_LUMINANCE):
    """Перебирает насыщенность S и яркость V, выбирает самый насыщенный
    вариант с яркостью, ближайшей к target."""
    best, best_key = None, None
    for s in np.linspace(0.95, 0.40, 23):          # сначала пробуем насыщенные
        for v in np.linspace(0.20, 1.00, 161):
            rgb = colorsys.hsv_to_rgb(hue_deg / 360, s, v)
            err = abs(luminance(rgb) - target)
            key = (round(err, 3), -s)               # сначала точность, потом насыщенность
            if best_key is None or key < best_key:
                best_key, best = key, rgb
    return tuple(int(round(c * 255)) for c in best)


COLORS = {name: matched_color(h) for name, h in HUES.items()}


# ------------------------------------------------------------------
# 3. Рисование фигур
# ------------------------------------------------------------------
def shape_polygon(shape, cx, cy, r):
    """Возвращает список точек многоугольника (для круга — None)."""
    if shape == "square":
        k = r * 0.86
        return [(cx - k, cy - k), (cx + k, cy - k), (cx + k, cy + k), (cx - k, cy + k)]
    if shape == "triangle":
        return [(cx + r * math.cos(a), cy + r * math.sin(a))
                for a in (-math.pi / 2, -math.pi / 2 + 2 * math.pi / 3,
                          -math.pi / 2 + 4 * math.pi / 3)]
    if shape == "star":
        pts = []
        for i in range(10):
            rr = r if i % 2 == 0 else r * 0.45
            a = -math.pi / 2 + i * math.pi / 5
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        return pts
    if shape == "cross":
        w = r * 0.36
        return [(cx - w, cy - r), (cx + w, cy - r), (cx + w, cy - w), (cx + r, cy - w),
                (cx + r, cy + w), (cx + w, cy + w), (cx + w, cy + r), (cx - w, cy + r),
                (cx - w, cy + w), (cx - r, cy + w), (cx - r, cy - w), (cx - w, cy - w)]
    return None  # circle


def draw_scene(objects):
    """Рисует сцену. objects — список словарей с shape, color, cx, cy, r.
    Возвращает (картинка RGB, маска объектов)."""
    img = Image.new("RGB", (SIZE, SIZE), (BG_GRAY,) * 3)
    mask = Image.new("L", (SIZE, SIZE), 0)
    d_img, d_mask = ImageDraw.Draw(img), ImageDraw.Draw(mask)
    for i, o in enumerate(objects):
        fill = COLORS[o["color"]]
        poly = shape_polygon(o["shape"], o["cx"], o["cy"], o["r"])
        if poly is None:
            box = [o["cx"] - o["r"], o["cy"] - o["r"], o["cx"] + o["r"], o["cy"] + o["r"]]
            d_img.ellipse(box, fill=fill)
            d_mask.ellipse(box, fill=i + 1)
        else:
            d_img.polygon(poly, fill=fill)
            d_mask.polygon(poly, fill=i + 1)
    return img, mask


# ------------------------------------------------------------------
# 4. Раскладка объектов без наложений
# ------------------------------------------------------------------
def place_objects(n, rng, tries=2000):
    """Случайные позиции и размеры, чтобы фигуры не касались друг друга."""
    for _ in range(tries):
        placed = []
        for _ in range(n):
            for _ in range(200):
                r = rng.uniform(*OBJ_SIZE) / 2
                cx = rng.uniform(r + 2, SIZE - r - 2)
                cy = rng.uniform(r + 2, SIZE - r - 2)
                if all(math.hypot(cx - p[0], cy - p[1]) > r + p[2] + MIN_GAP for p in placed):
                    placed.append((cx, cy, r))
                    break
        if len(placed) == n:
            return placed
    raise RuntimeError(f"не удалось разместить {n} объектов")


# ------------------------------------------------------------------
# 5. Какие формы и цвета получат объекты (условия эксперимента)
# ------------------------------------------------------------------
# Для 2–4 объектов задаём условие явно:
#   unique       — все цвета разные, все формы разные
#   shared-color — ровно одна пара объектов одного цвета
#   shared-shape — ровно одна пара объектов одной формы
#   shared-both  — одна пара делит цвет, другая пара делит форму
# Для 6 и 8 объектов повторы неизбежны (форм всего 5), условие = "crowded".

CONDITIONS = ["unique", "shared-color", "shared-shape", "shared-both"]


def assign_features(n, condition, rng):
    if condition == "crowded":
        shapes = [rng.choice(SHAPES) for _ in range(n)]
        colors = [rng.choice(list(COLORS)) for _ in range(n)]
        if len(set(colors)) < 2:                     # нужно хотя бы 2 разных цвета для swap
            colors[0], colors[1] = "red", "blue"
        return shapes, colors
    shapes = rng.sample(SHAPES, n)
    colors = rng.sample(list(COLORS), n)
    if condition in ("shared-color", "shared-both"):
        colors[1] = colors[0]                        # объекты 0 и 1 одного цвета
    if condition in ("shared-shape", "shared-both"):
        j = 2 if (condition == "shared-both" and n >= 3) else 1
        shapes[j] = shapes[0] if condition == "shared-shape" else shapes[1]
        # shared-shape: объекты 0 и 1 одной формы
        # shared-both:  объекты 1 и 2 одной формы (а 0 и 1 — одного цвета)
    return shapes, colors


def pick_swap(colors, rng):
    """Выбирает два объекта с РАЗНЫМИ цветами, их цвета будем менять местами."""
    pairs = [(i, j) for i in range(len(colors)) for j in range(i + 1, len(colors))
             if colors[i] != colors[j]]
    return rng.choice(pairs)


# ------------------------------------------------------------------
# 6. Главный цикл
# ------------------------------------------------------------------
def patches_of(mask_np, obj_id, patch=16, min_cover=0.25):
    """Номера патчей (сетка 14x14 для patch=16), которые объект покрывает хотя бы на 25%."""
    g = SIZE // patch
    out = []
    for py in range(g):
        for px in range(g):
            block = mask_np[py * patch:(py + 1) * patch, px * patch:(px + 1) * patch]
            if (block == obj_id).mean() >= min_cover:
                out.append(py * g + px)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="сколько пар S/S' сгенерировать")
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(f"{args.out}/images", exist_ok=True)
    os.makedirs(f"{args.out}/masks", exist_ok=True)

    with open(f"{args.out}/meta.jsonl", "w") as meta:
        for k in range(args.n):
            n = rng.choice(SET_SIZES)
            if n > 4:
                condition = "crowded"
            elif n == 2:
                # при 2 объектах одного цвета переставлять нечего -> только эти два условия
                condition = rng.choice(["unique", "shared-shape"])
            else:
                condition = rng.choice(CONDITIONS)
            shapes, colors = assign_features(n, condition, rng)
            pos = place_objects(n, rng)
            objs = [dict(shape=s, color=c, cx=p[0], cy=p[1], r=p[2])
                    for s, c, p in zip(shapes, colors, pos)]

            a, b = pick_swap(colors, rng)
            objs_sw = [dict(o) for o in objs]
            objs_sw[a]["color"], objs_sw[b]["color"] = objs[b]["color"], objs[a]["color"]

            img, mask = draw_scene(objs)
            img_sw, _ = draw_scene(objs_sw)          # маска у S' та же самая
            sid = f"{k:04d}"
            img.save(f"{args.out}/images/{sid}_S.png")
            img_sw.save(f"{args.out}/images/{sid}_Sp.png")
            mask.save(f"{args.out}/masks/{sid}.png")

            mask_np = np.array(mask)
            record = dict(
                id=sid, set_size=n, condition=condition, swapped=[a, b],
                objects=[dict(shape=o["shape"], color=o["color"], color_Sp=osw["color"],
                              bbox=[round(o["cx"] - o["r"]), round(o["cy"] - o["r"]),
                                    round(o["cx"] + o["r"]), round(o["cy"] + o["r"])],
                              patches16=patches_of(mask_np, i + 1))
                         for i, (o, osw) in enumerate(zip(objs, objs_sw))],
            )
            meta.write(json.dumps(record) + "\n")

    make_preview(args.out, min(12, args.n))
    print(f"Готово: {args.n} пар в папке '{args.out}/'.")
    print(f"Открой {args.out}/preview.png и посмотри глазами.")
    print("Цвета (RGB) и их яркость:")
    for name, rgb in COLORS.items():
        print(f"  {name:8s} {rgb}  luminance={luminance([c / 255 for c in rgb]):.3f}")


def make_preview(out, k):
    """Сводная картинка: слева S, справа S'."""
    pad = 8
    sheet = Image.new("RGB", (2 * SIZE + 3 * pad, k * (SIZE + pad) + pad), (255, 255, 255))
    for i in range(k):
        s = Image.open(f"{out}/images/{i:04d}_S.png")
        sp = Image.open(f"{out}/images/{i:04d}_Sp.png")
        y = pad + i * (SIZE + pad)
        sheet.paste(s, (pad, y))
        sheet.paste(sp, (2 * pad + SIZE, y))
    sheet.save(f"{out}/preview.png")


if __name__ == "__main__":
    main()

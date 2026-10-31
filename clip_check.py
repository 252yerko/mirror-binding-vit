"""
clip_check.py — проверка, что настоящий CLIP загружен ПРАВИЛЬНО (запускать один раз перед extract.py --model clip).

ПРОВЕРКА 1. Две независимые библиотеки дают один и тот же CLIP.
  A = timm  vit_base_patch16_clip_quickgelu_224.openai   (так грузит наш extract.py)
  B = open_clip ViT-B-16-quickgelu, веса openai           (вторая, независимая реализация)
  На одних и тех же картинках вектор картинки у A и B должен совпасть: косинус ≥ 0.9999.
  Для сравнения: C = timm vit_base_patch16_clip_224.openai — те же веса, но активация GELU вместо
  QuickGELU, как в оригинальном CLIP. Ожидаем, что C совпадает с B ХУЖЕ, чем A, — это показывает,
  зачем мы взяли версию с QuickGELU.

ПРОВЕРКА 2. Картинка подготовлена правильно (нормировка CLIP, цвета, размер).
  Проверка 1 этого не ловит: A и B получают один и тот же тензор. Поэтому спрашиваем сам CLIP
  текстом: «a photo of a {цвет} object» для 6 цветов. Считаем, как часто цвет с наибольшим
  сходством действительно есть на сцене. Уровень случайности = средняя доля цветов сцены среди 6
  (у нас 2–4 объекта разных цветов → около 50%). Ожидаем заметно выше: ≥ уровень случайности + 15 п.п.

ЗАПУСК (из папки binding-vit, нужен интернет — первый раз скачаются веса, ~600 МБ, потом из кэша):
  python clip_check.py 2>&1 | tee clip_check_log.txt
"""

import json

import numpy as np
import torch

from extract import CLIP_NAME, load_image

DATA = "data_v2"
N_EMB = 16          # картинок для проверки 1
N_ZS = 200          # сцен для проверки 2
GELU_NAME = "vit_base_patch16_clip_224.openai"


def cos(a, b):
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    return (a * b).sum(-1)


def load_timm(name):
    import timm
    return timm.create_model(name, pretrained=True).eval()


def load_open_clip():
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16-quickgelu", pretrained="openai")
    return model.eval(), open_clip.get_tokenizer("ViT-B-16-quickgelu")


@torch.no_grad()
def main():
    meta = [json.loads(line) for line in open(f"{DATA}/meta.jsonl")]
    colors = sorted({o["color"] for r in meta for o in r["objects"]})

    # ---------- проверка 1
    x = torch.stack([load_image(f"{DATA}/images/{r['id']}_S.png", "clip") for r in meta[:N_EMB]])
    A = load_timm(CLIP_NAME)
    oc, tok = load_open_clip()
    C = load_timm(GELU_NAME)
    act_a = type(A.blocks[0].mlp.act).__name__
    act_c = type(C.blocks[0].mlp.act).__name__
    ea, eb, ec = A(x), oc.encode_image(x), C(x)
    ab, cb = cos(ea, eb), cos(ec, eb)
    ok1 = float(ab.min()) >= 0.9999 and float(cb.mean()) < float(ab.mean())
    print(f"Проверка 1 (две библиотеки, {N_EMB} картинок):")
    print(f"  timm {act_a} vs open_clip: косинус мин {float(ab.min()):.6f}, среднее {float(ab.mean()):.6f}")
    print(f"  timm {act_c} vs open_clip: косинус мин {float(cb.min()):.6f}, среднее {float(cb.mean()):.6f}  (для сравнения)")
    print(f"  {'✅ наш CLIP совпадает с независимой реализацией' if ok1 else '⚠️ проверь весь вывод'}")

    # ---------- проверка 2
    texts = tok([f"a photo of a {c} object" for c in colors])
    t = oc.encode_text(texts)
    t = t / t.norm(dim=-1, keepdim=True)
    hits, chance = [], []
    for s in range(0, min(N_ZS, len(meta)), 25):
        recs = meta[s:s + 25]
        xi = torch.stack([load_image(f"{DATA}/images/{r['id']}_S.png", "clip") for r in recs])
        e = A(xi)
        e = e / e.norm(dim=-1, keepdim=True)
        top = (e @ t.T).argmax(-1)
        for r, k in zip(recs, top.tolist()):
            present = {o["color"] for o in r["objects"]}
            hits.append(colors[k] in present)
            chance.append(len(present) / len(colors))
    acc, ch = float(np.mean(hits)), float(np.mean(chance))
    ok2 = acc >= ch + 0.15
    print(f"Проверка 2 (CLIP по тексту называет цвет, который есть на сцене; {len(hits)} сцен):")
    print(f"  точность {acc * 100:.1f}%   уровень случайности {ch * 100:.1f}%   "
          f"{'✅ картинка подготовлена правильно' if ok2 else '⚠️ проверь весь вывод'}")
    print(f"Цвета: {', '.join(colors)}")


if __name__ == "__main__":
    main()

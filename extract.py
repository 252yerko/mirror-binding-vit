"""
extract.py — прогоняет сцены через DINO ViT-B/16 и сохраняет активации всех слоёв.

ЧТО ДЕЛАЕТ:
  Берёт каждую картинку из data/images (и S, и S'), показывает её модели DINO
  и записывает, что модель "думает" о каждом кусочке картинки на каждом слое.

  ViT режет картинку 224x224 на сетку 14x14 = 196 патчей по 16x16 пикселей.
  Плюс один служебный токен CLS в начале. Итого 197 токенов.
  Каждый токен на каждом слое — это вектор из 768 чисел.
  Эти векторы и есть "активации" — то, что мы потом будем исследовать.

  ВАЖНО: токен номер 0 — это CLS. Патч номер p из meta.jsonl (patches16)
  лежит в токене номер p + 1. Если про это забыть, всё поедет на один патч.

ЧТО СОХРАНЯЕТ (в папку acts/):
  L00.npy       — вход в первый блок (патчи + позиционные коды, модель ещё ничего не смешала)
  L01.npy..L12.npy — выход блоков 1..12 (residual stream, до финальной нормализации)
                  форма каждого файла: [число картинок, 197, 768], тип float16
  index.csv     — какая строка файла какой картинке соответствует (id сцены, S / Sp / Sr)
  info.json     — какая модель, сколько картинок, когда запускали

  Для 200 пар это ~1.5 ГБ. Для 5000 пар будет ~36 ГБ — на диске места хватает.

ЗАПУСК:
  python extract.py                  # берёт data/, пишет в acts/
  python extract.py --data data_v2 --out acts_v2    # для сцен v2 (S, Sp, Sr)

НУЛЕВАЯ МОДЕЛЬ (добавлено 5.10):
  python extract.py --data data_v2 --out acts_v2_rand0 --random-init 0
  Та же архитектура ViT-B/16, но веса СЛУЧАЙНЫЕ — ровно такие, с каких DINO
  начинала обучение (нормальное распределение, разброс 0.02). Число после
  --random-init — номер случайности (seed): с одним и тем же номером веса
  получаются одинаковые, поэтому результат можно повторить.
  Зачем: если у необученной сети эффект тоже есть, значит его даёт устройство
  сети, а не обучение. Номер seed и тип весов записываются в info.json.

CLIP (добавлено 6.10):
  python extract.py --model clip --data data_v2 --out ~/binding-vit-data/acts_v2_clip
  python extract.py --model clip --data data_v2 --out ~/binding-vit-data/acts_v2_clip_rand0 --random-init 0
  Зрительная часть OpenAI CLIP ViT-B/16 — та же форма, что у DINO: 12 блоков, 197 токенов по 768.
  Загружается через timm: vit_base_patch16_clip_quickgelu_224.openai — именно QuickGELU,
  как в оригинальном CLIP (обычная версия timm с GELU даёт немного другие числа).
  Нормализация картинки — своя, CLIP (не ImageNet).
  L00 = вход в первый блок = после ln_pre (у CLIP перед блоками есть LayerNorm; у DINO его нет).
  ЗАЧЕМ: CLIP обучали БЕЗ аугментации отражением. Сравнение с DINO отвечает, откуда у DINO
  «слепота к зеркалу слева-направо» — от аугментации или от самих данных.
  Перед первым запуском проверить загрузку: python clip_check.py
  Активации писать сразу в ~/binding-vit-data/ (не синхронизируется с iCloud), потом ярлык.
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from PIL import Image

# ------------------------------------------------------------------
# 1. Подготовка картинки для модели
# ------------------------------------------------------------------
# DINO обучали на фотографиях ImageNet, у которых цвета были "нормированы":
# из каждого канала вычитали среднее и делили на разброс. Делаем так же,
# иначе модель увидит картинку, не похожую ни на что из обучения.
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
# CLIP обучали со своей нормировкой (её же даёт timm в pretrained_cfg)
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
CLIP_NAME = "vit_base_patch16_clip_quickgelu_224.openai"


def load_image(path, arch="dino"):
    img = Image.open(path).convert("RGB")
    assert img.size == (224, 224), f"{path}: размер {img.size}, нужен 224x224"
    x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0
    if arch == "clip":
        return (x - CLIP_MEAN) / CLIP_STD
    return (x - MEAN) / STD


# ------------------------------------------------------------------
# 2. Прогон через модель со сбором всех слоёв
# ------------------------------------------------------------------
@torch.no_grad()
def run_layers(model, x):
    """Проходит по модели блок за блоком и запоминает выход каждого.
    Возвращает список из 13 тензоров [batch, 197, 768]: L00 (вход) и L01..L12."""
    if hasattr(model, "prepare_tokens"):     # DINO (facebookresearch/dino)
        h = model.prepare_tokens(x)          # патчи -> векторы + CLS + позиционные коды
    else:                                    # CLIP через timm
        h = model.norm_pre(model.patch_drop(model._pos_embed(model.patch_embed(x))))
    outs = [h]
    for blk in model.blocks:             # 12 блоков трансформера
        h = blk(h)
        outs.append(h)
    return outs


@torch.no_grad()
def check_forward(model, x, outs):
    """Встроенная проверка: наш пошаговый прогон даёт то же, что собственный прогон модели.
    DINO: model(x) = norm(последний слой)[CLS]. CLIP (timm): model(x) = head(norm(последний слой)[CLS])."""
    ref = model(x)
    if hasattr(model, "prepare_tokens"):
        mine = model.norm(outs[-1])[:, 0]
    else:
        mine = model.forward_head(model.norm(outs[-1]))
    d = float((mine - ref).abs().max())
    print(f"Проверка прогона (наши слои = прогон самой модели): макс. разница {d:.2e}  "
          f"{'✅' if d < 1e-3 else '⚠️ пришли Claude весь вывод'}")


def load_model(arch, random_init=None):
    """arch: dino | clip. random_init: None = обученные веса, число = seed случайных весов.
    Возвращает (модель, описание весов, имя модели)."""
    if random_init is not None:
        torch.manual_seed(random_init)   # seed ДО создания модели: веса заполняются при создании
    if arch == "dino":
        name = "dino_vitb16"
        if random_init is None:
            return torch.hub.load("facebookresearch/dino:main", "dino_vitb16"), "обученные DINO", name
        m = torch.hub.load("facebookresearch/dino:main", "dino_vitb16", pretrained=False)
        return m, f"СЛУЧАЙНЫЕ (нулевая модель), seed {random_init}", name
    import timm
    name = f"clip_vitb16_openai (timm {CLIP_NAME})"
    m = timm.create_model(CLIP_NAME, pretrained=random_init is None)
    if random_init is None:
        return m, "обученный CLIP (OpenAI)", name
    return m, f"СЛУЧАЙНЫЕ CLIP (нулевая модель), seed {random_init}", name


def fingerprint(model):
    """Короткий "отпечаток" весов: сумма модулей весов первого слоя (нарезка на патчи).
    Разные веса -> разные отпечатки. Нужен, чтобы в логе было видно, какие веса работали."""
    return float(model.patch_embed.proj.weight.detach().abs().sum())


def check_random(model, arch="dino"):
    """Встроенная проверка нулевой модели: веса НЕ совпадают с обученными (DINO или CLIP).
    Обученные веса уже лежат на компьютере (кэш torch hub / huggingface), поэтому сравнение быстрое."""
    rnd = fingerprint(model)
    try:
        trained = load_model(arch)[0]
    except Exception as e:
        print(f"⚠️ Не смог загрузить обученную модель для сравнения ({type(e).__name__}). "
              f"Отпечаток случайных весов: {rnd:.4f}. Пришли Claude весь вывод.")
        return
    tr = fingerprint(trained)
    same = all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), trained.state_dict().values()))
    assert not same, "веса совпали с обученной моделью — нулевая модель не получилась!"
    print(f"Проверка весов ✅ случайные отличаются от обученных. Отпечаток: случайные {rnd:.4f}, обученные {tr:.4f}")
    del trained


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")       # видеокарта MacBook
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="acts")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--random-init", type=int, default=None, metavar="SEED",
                    help="нулевая модель: случайные веса с этим seed вместо обученных")
    ap.add_argument("--model", choices=["dino", "clip"], default="dino",
                    help="dino (по умолчанию, как раньше) или clip (OpenAI CLIP ViT-B/16)")
    args = ap.parse_args()

    # --- список картинок в фиксированном порядке: 0000_S, 0000_Sp, 0001_S, ...
    recs = [json.loads(line) for line in open(f"{args.data}/meta.jsonl")]
    ids = [r["id"] for r in recs]
    # v1: у каждой сцены S и Sp. v2: S, Sp и Sr — список версий берём из meta.jsonl
    versions = recs[0].get("versions", ["S", "Sp"])
    rows = [(sid, v) for sid in ids for v in versions]
    n = len(rows)
    print(f"Сцен: {len(ids)}, картинок: {n}")

    # --- модель
    device = pick_device()
    # seed задаётся внутри load_model ДО создания модели: именно при создании веса заполняются случайно
    model, weights, model_name = load_model(args.model, args.random_init)
    if args.random_init is not None:
        check_random(model, args.model)
    model.eval().to(device)
    n_layers = len(model.blocks) + 1
    print(f"Модель загружена, веса: {weights}, устройство: {device}, слоёв для записи: {n_layers}")

    # --- файлы на диске. open_memmap создаёт файл нужного размера сразу
    # и пишет в него по кусочкам — вся матрица в память не грузится (у нас 8 ГБ RAM).
    os.makedirs(args.out, exist_ok=True)
    files = [np.lib.format.open_memmap(f"{args.out}/L{l:02d}.npy", mode="w+",
                                       dtype=np.float16, shape=(n, 197, 768))
             for l in range(n_layers)]

    t0 = time.time()
    for start in range(0, n, args.batch):
        chunk = rows[start:start + args.batch]
        x = torch.stack([load_image(f"{args.data}/images/{sid}_{v}.png", args.model) for sid, v in chunk])
        outs = run_layers(model, x.to(device))
        if start == 0:
            check_forward(model, x.to(device), outs)
        for l, h in enumerate(outs):
            files[l][start:start + len(chunk)] = h.float().cpu().numpy().astype(np.float16)
        done = start + len(chunk)
        print(f"\r  {done}/{n} картинок, {time.time() - t0:.0f} с", end="", flush=True)
    print()
    for f in files:
        f.flush()

    # --- index.csv и info.json
    with open(f"{args.out}/index.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "id", "version"])
        for i, (sid, v) in enumerate(rows):
            w.writerow([i, sid, v])
    json.dump(dict(model=model_name, weights=weights, random_init_seed=args.random_init,
                   weights_fingerprint=round(fingerprint(model.cpu()), 4), n_images=n, n_scenes=len(ids),
                   layers=("L00=вход (patch+pos), L01..L12=выход блоков, до финального norm" if args.model == "dino" else
                           "L00=вход в 1-й блок (patch+pos, после ln_pre), L01..L12=выход блоков, до финального norm"),
                   shape=[n, 197, 768], dtype="float16", token0="CLS, патч p -> токен p+1",
                   date=time.strftime("%Y-%m-%d %H:%M")),
              open(f"{args.out}/info.json", "w"), ensure_ascii=False, indent=1)

    # --- быстрая самопроверка: нет ли NaN/бесконечностей, и что S и S' различаются
    last = np.load(f"{args.out}/L{n_layers - 1:02d}.npy", mmap_mode="r")
    sample = np.asarray(last[:min(n, 40)], dtype=np.float32)
    assert np.isfinite(sample).all(), "в активациях есть NaN/inf — что-то не так"
    diff = np.abs(sample[0] - sample[1]).mean()
    # подпись берём из самих данных: строки 0 и 1 — это первые две версии первой сцены
    # (v1: S и Sp, v2: S и Sp), чтобы надпись всегда совпадала с тем, что реально сравнили
    pair = f"{rows[0][1]} vs {rows[1][1]}"
    print(f"Проверка: NaN нет. Средняя разница {pair} (сцена {rows[0][0]}, слой {n_layers - 1}): {diff:.4f} (должна быть > 0)")
    print(f"Готово за {time.time() - t0:.0f} с. Активации в папке '{args.out}/'.")


if __name__ == "__main__":
    main()

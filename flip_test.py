"""
flip_test.py — проверка гипотезы «зеркало»: почему DINO к глубоким слоям хуже помнит,
какой близнец ЛЕВЫЙ, но хорошо помнит, какой ВЕРХНИЙ?

ГИПОТЕЗА:
  DINO обучали с аугментацией RandomHorizontalFlip (main_dino.py, строка 422): картинку
  случайно отражали слева направо, а сверху вниз — никогда. Сеть учится считать картинку
  и её зеркальную копию (слева направо) почти одинаковыми. Но в такой копии левый и правый
  близнецы меняются местами → «кто слева» стирается. «Кто сверху» при этом не трогается.

ПРЕДСКАЗАНИЕ (записано ДО запуска, 5.10):
  Для обученной DINO на глубоких слоях зеркало слева-направо меняет CLS заметно МЕНЬШЕ,
  чем зеркало сверху-вниз:  d(S, зерк.Г) / d(S, зерк.В) < 1, и это отношение падает с глубиной.
  Для случайных сетей (та же архитектура до обучения) асимметрии нет: отношение ≈ 1.
  Если у DINO отношение ≈ 1 или > 1 — гипотеза опровергнута.

ВАЖНО — ЧТО ЭТОТ ОПЫТ НЕ РАЗЛИЧАЕТ:
  Даже если предсказание подтвердится, причин может быть две: (1) аугментация отражением,
  (2) сами фотографии: мир примерно симметричен слева-направо, но не сверху-вниз (небо сверху).
  Этот опыт проверяет СЛЕДСТВИЕ («сеть почти не различает зеркало слева-направо»), а не причину.

ПОДВОХ, КОТОРЫЙ УБРАН:
  Треугольник и звезда вверх ногами — это другие фигуры, а слева-направо они симметричны.
  Тогда зеркало сверху-вниз меняло бы ещё и форму. Поэтому берём ТОЛЬКО сцены, где все фигуры —
  круг, квадрат или крест (симметричны в обе стороны). Это проверяется по пикселям (проверка 2).

ЧТО СРАВНИВАЕМ (для каждой сцены S, на каждом слое):
  d_Г = косинусное расстояние между S и её зеркалом слева-направо
  d_В = то же для зеркала сверху-вниз
  отношение = среднее d_Г / среднее d_В, 95% интервал — бутстрэп по сценам
  Где: CLS и «среднее по всем патчам» (общий вектор картинки без CLS).
  Плюс: сами позиционные коды сети — насколько код клетки похож на код зеркальной клетки
  слева-направо и сверху-вниз (это не картинки, а веса сети).

ВСТРОЕННЫЕ ПРОВЕРКИ:
  1. Пустой серый фон симметричен в обе стороны → для него d_Г = d_В = 0 на всех слоях
     (до погрешности вычислений). Иначе — ошибка в отражении.
  2. Симметрия фигур по пикселям: каждый объект сравнивается со своим зеркалом (по маске),
     доля совпадения (IoU) слева-направо и сверху-вниз должна быть одинаковой и ~1.
  3. Отпечатки весов случайных сетей должны совпасть с теми, что были в extract.py
     (seed 0: 10638.78, seed 1: 10621.16, seed 2: 10654.00) — это те же самые нулевые модели.
  Справка (не проверка): асимметрия самих картинок. Даже с идеально симметричными фигурами
  конкретный набор сцен может быть чуть асимметричен (например, объекты в среднем чуть правее
  центра). Поэтому у случайных сетей отношение не обязано быть ровно 1, и DINO сравниваем
  со случайными сетями на ТЕХ ЖЕ сценах, а не с числом 1. (Найдено в облачном тесте 5.10:
  183 сцены, асимметрия пикселей Г/В = 0,98, у случайных сетей «среднее по патчам» 0,94–0,96.)

ЗАПУСК (из папки binding-vit):
  python flip_test.py 2>&1 | tee flip_test_log.txt
  -> таблица, график flip_test.png, числа flip_test.csv (несколько минут)

CLIP — РЕШАЮЩИЙ ОПЫТ (добавлено 6.10):
  python flip_test.py --model clip 2>&1 | tee flip_test_clip_log.txt
  -> flip_test_clip.png / .csv (файлы DINO не перезаписываются)
  Тот же опыт на OpenAI CLIP ViT-B/16 и трёх случайных CLIP (та же архитектура до обучения).
  CLIP обучали БЕЗ отражений (единственная аугментация — случайная обрезка), DINO — С отражением.
  ПРЕДСКАЗАНИЯ (записаны ДО запуска, 6.10):
    «виновата аугментация» -> у CLIP отношение на L12 близко к случайным (~1), далеко от DINO (0,34)
    «виноваты данные» (Bouchacourt et al. 2021: инвариантность из данных) -> у CLIP тоже сильно < 1
    промежуточный результат -> работают обе причины
  Оговорка: CLIP учился по подписям, где есть слова «left»/«right» — это само может беречь лево/право.
  Проверка 3 для CLIP сверяет отпечатки со значениями в acts_v2_clip_rand*/info.json (если они уже есть).
  Отпечаток случайного CLIP совпадает с отпечатком случайной DINO с тем же seed — это нормально:
  первый слой (нарезка патчей) у них одной формы и создаётся первым из того же генератора случайных чисел.

MAE и I-JEPA (добавлено 15.10) — какая причина слепоты к зеркалу:
  python flip_test.py --model mae   2>&1 | tee flip_test_mae_log.txt
  python flip_test.py --model ijepa 2>&1 | tee flip_test_ijepa_log.txt
  python flip_compare.py            2>&1 | tee flip_compare_log.txt
  -> flip_test_mae.* / flip_test_ijepa.* (+ *_dist.npz: расстояния по каждой сцене), файлы DINO и CLIP не трогаются.
  Обе сети — ViT-H/14 (630,8 млн параметров, 32 блока -> слои L00..L32), обе учились на ImageNet-1k.
  MAE (timm vit_huge_patch14_224.mae): учили С отражением слева-направо (RandomHorizontalFlip), но задача —
    дорисовать пиксели на своих местах; одинакового ответа для картинки и её зеркала не требовали.
  I-JEPA (официальный чекпойнт IN1K-vit.h.14-300e): учили БЕЗ отражений (use_horizontal_flip: false).
    Берём ЦЕЛЕВОЙ энкодер (target_encoder) — так в статье I-JEPA, прил. A.1: «We use the target-encoder for
    evaluation and average pool its output». (timm по умолчанию берёт контекстный энкодер — поэтому грузим сами.)
    У I-JEPA НЕТ CLS-токена (все 256 токенов — патчи) -> для неё есть только «среднее по патчам».
  Случайные сети: по 3 на каждую архитектуру (seed 0/1/2), без обучения.
  ПРАВИЛО ЗАСЧЁТА СТАВОК (реестр 33; уточнено 15.10 ДО запуска, т.к. у I-JEPA нет CLS):
    число = отношение d(зеркало ←→)/d(зеркало ↑↓) на ПОСЛЕДНЕМ слое L32 (выход 32-го блока, до финального norm),
    те же 631 сцена. MAE — по CLS (как в ставке); I-JEPA — по среднему по патчам (CLS нет).
    Ставка засчитана, если точечное значение, округлённое до 0,01, внутри диапазона включительно
    (MAE 0,60–0,75; I-JEPA 0,50–0,65). 95% интервал печатается рядом.
  ПРАВИЛО ВЫВОДА MAE против I-JEPA (записано 15.10 ДО запуска; считает flip_compare.py):
    сравниваем одинаковое с одинаковым — «среднее по патчам», L32. Δ = отношение(I-JEPA) − отношение(MAE),
    95% интервал — бутстрэп по сценам (одни и те же сцены для обеих сетей).
    «А» (достаточно видеть отражения): нижняя граница Δ > 0,10 — MAE заметно слепее I-JEPA.
    «Б» (нужна задача «одинаковый ответ»): весь интервал Δ внутри [−0,10; 0,10] — разницы нет.
    «Против А» : верхняя граница Δ < −0,10 — MAE даже МЕНЕЕ слеп, чем I-JEPA.
    иначе — «не различает» (интервал слишком широкий или задевает порог).
    Порог 0,10 ≈ седьмая часть эффекта DINO (0,34 против ~1 у случайных).
  Оговорки: MAE и I-JEPA различаются не только отражениями, но и целью (пиксели против описаний);
    MAE-CLS в обучении MAE напрямую не используется — поэтому сравнение MAE и I-JEPA идёт по среднему по патчам.
"""

import csv
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import argparse
import math
import os

from extract import load_image, pick_device, run_layers, fingerprint, check_forward, load_model as ex_load_model

DATA = "data_v2"
SYM_SHAPES = {"circle", "square", "cross"}
N_LAYERS = 13                   # для DINO/CLIP; у ViT-H/14 станет 33 (берётся из модели)
G, P = 14, 16                   # сетка 14×16 px — только для «справки» о данных (одинаковая для всех моделей)
N_BOOT = 1000
BATCH = 32
ARCH = "dino"                   # меняется флагом --model
MAIN, MAIN_LONG, STEM = "DINO", "обученная DINO", "flip_test"
MODELS = [("DINO", None), ("rand0", 0), ("rand1", 1), ("rand2", 2)]
EXPECTED_FP = {0: 10638.7832, 1: 10621.1621, 2: 10653.9980}  # из extract_rand*_log.txt на Маке (допуск 0.01 — округление)
GROUPS = ["cls", "mean"]
NAMES = {"cls": "CLS", "mean": "среднее по патчам"}

# --- MAE и I-JEPA (15.10) ---
BIG_ARCH = {"mae": "vit_huge_patch14_224", "ijepa": "vit_huge_patch14_gap_224"}
IJEPA_URL = "https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar"
BETS = {"mae": ("cls", 0.60, 0.75), "ijepa": ("mean", 0.50, 0.65)}   # ставки Ерназара, реестр 33 (7.10)


def cos_dist(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return 1.0 - (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)


def load_ijepa_target(arch=BIG_ARCH["ijepa"], path=None):
    """Официальный чекпойнт I-JEPA -> ЦЕЛЕВОЙ энкодер (target_encoder) в timm-модели той же формы."""
    import timm
    from timm.models.vision_transformer import checkpoint_filter_fn
    m = timm.create_model(arch, pretrained=False, num_classes=0)
    if path is None:
        path = os.path.join(torch.hub.get_dir(), "checkpoints", os.path.basename(IJEPA_URL))
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print(f"  Скачиваю официальный чекпойнт I-JEPA: {IJEPA_URL}\n  (большой файл: в нём оба энкодера и состояние оптимизатора; один раз)")
            torch.hub.download_url_to_file(IJEPA_URL, path, progress=True)
    print(f"  чекпойнт: {path}  ({os.path.getsize(path) / 1e9:.2f} ГБ)")
    try:
        ck = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except Exception as e:
        print(f"  (быстрая загрузка не вышла: {type(e).__name__}; гружу целиком — нужно много оперативной памяти)")
        ck = torch.load(path, map_location="cpu", weights_only=False)
    assert "target_encoder" in ck, f"в чекпойнте нет target_encoder, есть: {list(ck.keys())}"

    def prep(sd):                     # checkpoint_filter_fn ждёт ключи вида 'module.…' внутри 'encoder'
        return checkpoint_filter_fn({"encoder": {(k if k.startswith("module.") else "module." + k): v
                                                 for k, v in sd.items()}}, m)
    sd_t = prep(ck["target_encoder"])
    m.load_state_dict(sd_t, strict=True)           # strict: все веса на месте, лишних нет
    if "encoder" in ck:                            # справка: насколько контекстный энкодер отличается от целевого
        sd_e = prep(ck["encoder"])
        num = sum(float((sd_t[k].float() - sd_e[k].float()).pow(2).sum()) for k in sd_t)
        den = sum(float(sd_t[k].float().pow(2).sum()) for k in sd_t)
        print(f"  справка: отличие контекстного энкодера от целевого (отн. L2) = {math.sqrt(num / den):.4f}"
              f"{'   эпоха ' + str(ck['epoch']) if 'epoch' in ck else ''}")
    del ck
    return m


def load_model(seed):
    if ARCH in BIG_ARCH:               # MAE / I-JEPA (ViT-H/14, timm)
        import timm
        name = BIG_ARCH[ARCH]
        if seed is not None:
            torch.manual_seed(seed)    # seed ДО создания модели, как везде
            return timm.create_model(name, pretrained=False, num_classes=0), f"СЛУЧАЙНЫЕ {name} (нулевая модель), seed {seed}"
        if ARCH == "mae":
            return timm.create_model(name + ".mae", pretrained=True), f"обученный MAE (timm {name}.mae)"
        return load_ijepa_target(), "обученная I-JEPA, ЦЕЛЕВОЙ энкодер (официальный чекпойнт IN1K-vit.h.14-300e)"
    if ARCH != "dino":                # CLIP: тот же загрузчик, что в extract.py
        m, weights, _ = ex_load_model(ARCH, seed)
        return m, weights
    if seed is None:
        return torch.hub.load("facebookresearch/dino:main", "dino_vitb16"), "обученные DINO"
    torch.manual_seed(seed)          # как в extract.py: seed ДО создания модели
    m = torch.hub.load("facebookresearch/dino:main", "dino_vitb16", pretrained=False)
    return m, f"СЛУЧАЙНЫЕ (нулевая модель), seed {seed}"


@torch.no_grad()
def features(model, x, device):
    """x: [B,3,224,224] -> массив [B, слои, группы, D]. Группы: CLS (если есть) и среднее по патчам.
    n_pre — сколько служебных токенов в начале (CLS): у DINO/CLIP/MAE 1, у I-JEPA 0."""
    n_pre = getattr(model, "num_prefix_tokens", 1)
    outs = run_layers(model, x.to(device))
    pick = {"cls": lambda h: h[:, 0], "mean": lambda h: h[:, n_pre:].mean(1)}
    f = torch.stack([torch.stack([pick[g](h) for g in GROUPS], 1) for h in outs], 1)
    return f.float().cpu().numpy()


def pos_symmetry(model):
    """Похожесть позиционного кода клетки на код зеркальной клетки (косинус), без CLS."""
    n = model.pos_embed.shape[1]
    g = int(round(math.sqrt(n)))                       # 197 -> 14, 257 -> 16, 256 -> 16
    pe = model.pos_embed[0, n - g * g:].detach().float().cpu().numpy().reshape(g, g, -1)
    def mean_cos(a, b, keep):
        c = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
        return float(c[keep].mean())
    rr, cc = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
    h = mean_cos(pe, pe[:, ::-1], cc != g - 1 - cc)      # слева-направо (клетка ≠ своему зеркалу)
    v = mean_cos(pe, pe[::-1, :], rr != g - 1 - rr)      # сверху-вниз
    return h, v


def boot_ratio(dh, dv, rng):
    if dv.mean() < 1e-9:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, len(dh), size=(N_BOOT, len(dh)))
    b = dh[idx].mean(1) / np.maximum(dv[idx].mean(1), 1e-15)
    return float(dh.mean() / dv.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    global ARCH, MAIN, MAIN_LONG, STEM, MODELS, EXPECTED_FP, GROUPS, N_LAYERS, BATCH
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["dino", "clip", "mae", "ijepa"], default="dino")
    ap.add_argument("--limit", type=int, default=0, help="только для проверки в облаке: первые N сцен")
    ap.add_argument("--n-random", type=int, default=3, help="сколько случайных сетей (по умолчанию 3)")
    args = ap.parse_args()
    ARCH = args.model
    if ARCH == "clip":
        MAIN, MAIN_LONG, STEM = "CLIP", "обученный CLIP (OpenAI)", "flip_test_clip"
        MODELS = [("CLIP", None), ("clip_rand0", 0), ("clip_rand1", 1), ("clip_rand2", 2)]
        EXPECTED_FP = {}
        for sd in (0, 1, 2):
            f = f"acts_v2_clip_rand{sd}/info.json"
            if os.path.exists(f):
                EXPECTED_FP[sd] = json.load(open(f))["weights_fingerprint"]
    elif ARCH in BIG_ARCH:
        MAIN = {"mae": "MAE", "ijepa": "I-JEPA"}[ARCH]
        MAIN_LONG = {"mae": "обученный MAE (ViT-H/14)", "ijepa": "обученная I-JEPA (ViT-H/14, целевой энкодер)"}[ARCH]
        STEM = f"flip_test_{ARCH}"
        MODELS = [(MAIN, None)] + [(f"{ARCH}_rand{s}", s) for s in range(3)]
        EXPECTED_FP = {}
        GROUPS = ["cls", "mean"] if ARCH == "mae" else ["mean"]   # у I-JEPA нет CLS
        BATCH = 16                                                 # ViT-H большая — меньше картинок за раз
    MODELS = [m for m in MODELS if m[1] is None or m[1] < args.n_random]
    if args.limit:
        STEM += f"_test{args.limit}"           # проверочный прогон не перезаписывает настоящие файлы
    print(f"Модель: {MAIN_LONG} + {len(MODELS) - 1} случайные сети той же архитектуры   выход: {STEM}.csv, {STEM}.png")
    if ARCH in BIG_ARCH:
        import timm
        print(f"timm {timm.__version__}, torch {torch.__version__}")
    t0 = time.time()
    meta = [json.loads(line) for line in open(f"{DATA}/meta.jsonl")]
    sel = [r for r in meta if all(o["shape"] in SYM_SHAPES for o in r["objects"])]
    if args.limit:
        sel = sel[:args.limit]
        print(f"⚠️ ПРОВЕРОЧНЫЙ прогон: только {len(sel)} сцен (--limit). Для настоящих чисел запускай без --limit.")
    print(f"Сцен всего: {len(meta)}. Только круг/квадрат/крест: {len(sel)} — берём их (картинка S).")

    # --- проверка 2: симметрия фигур по пикселям (маска объекта против её зеркала вокруг центра объекта)
    iou_h, iou_v = [], []
    for rec in sel:
        mask = np.asarray(Image.open(f"{DATA}/masks/{rec['id']}.png"))
        for k in range(len(rec["objects"])):
            ys, xs = np.where(mask == k + 1)
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            obj = mask[y0:y1 + 1, x0:x1 + 1] == k + 1
            for arr, mir in ((iou_h, obj[:, ::-1]), (iou_v, obj[::-1, :])):
                arr.append((obj & mir).sum() / (obj | mir).sum())
    iou_h, iou_v = np.array(iou_h), np.array(iou_v)
    ok2 = abs(iou_h.mean() - iou_v.mean()) < 0.01 and min(iou_h.mean(), iou_v.mean()) > 0.95
    print(f"Проверка 2 (фигуры симметричны): совпадение с зеркалом слева-направо {iou_h.mean():.4f}, "
          f"сверху-вниз {iou_v.mean():.4f} (объектов {len(iou_h)})  {'✅' if ok2 else '⚠️ проверь весь вывод'}")

    # --- справка: асимметрия самих данных (средний патч 16×16, сравнение с его зеркалами)
    ah, av = [], []
    for rec in sel:
        im = np.asarray(Image.open(f"{DATA}/images/{rec['id']}_S.png").convert("RGB")).astype(np.float64)
        A = im.reshape(G, P, G, P, 3).mean((0, 2))
        ah.append(np.abs(A - A[:, ::-1]).mean())
        av.append(np.abs(A - A[::-1, :]).mean())
    cxs = [o["center"][0] for r in sel for o in r["objects"]]
    cys = [o["center"][1] for r in sel for o in r["objects"]]
    print(f"Справка (данные): асимметрия среднего патча в пикселях Г/В = {np.mean(ah) / np.mean(av):.3f};  "
          f"средний центр объектов x = {np.mean(cxs):.1f}, y = {np.mean(cys):.1f} (центр картинки 111.5)")

    device = pick_device()
    imgs = torch.stack([load_image(f"{DATA}/images/{r['id']}_S.png", ARCH) for r in sel])   # [N,3,224,224]
    blank = torch.stack([load_image(f"{DATA}/images/{sel[0]['id']}_S.png", ARCH)])
    blank[:] = blank[0, :, :1, :1]          # пустой серый фон: все пиксели = цвет фона (угол картинки)

    rows, summary, pos, dist = [], {}, {}, {}
    for name, seed in MODELS:
        model, weights = load_model(seed)
        model.eval()
        fp = fingerprint(model)
        if seed is not None and seed in EXPECTED_FP:
            ok3 = abs(fp - EXPECTED_FP[seed]) < 0.01
            print(f"\n### {name}: {weights}   отпечаток весов {fp:.4f}  "
                  f"{'✅ совпадает с extract.py' if ok3 else '⚠️ НЕ совпадает — проверь весь вывод'}")
        elif seed is not None:
            print(f"\n### {name}: {weights}   отпечаток весов {fp:.4f}  (сверить не с чем: активаций этой сети ещё нет)")
        else:
            print(f"\n### {name}: {weights}   отпечаток весов {fp:.4f}")
        N_LAYERS = len(model.blocks) + 1
        pos[name] = pos_symmetry(model)
        model.to(device)
        if ARCH in BIG_ARCH:             # проверка прогона: наши слои = собственный прогон модели
            xx = imgs[:2].to(device)
            check_forward(model, xx, run_layers(model, xx))
            print(f"  слоёв: {N_LAYERS} (L00..L{N_LAYERS - 1:02d}), служебных токенов (CLS): {getattr(model, 'num_prefix_tokens', 1)}")

        # проверка 1: пустой фон
        fb = features(model, torch.cat([blank, blank.flip(3), blank.flip(2)]), device)
        z = max(cos_dist(fb[0], fb[1]).max(), cos_dist(fb[0], fb[2]).max())
        print(f"  Проверка 1 (пустой фон: зеркало ничего не меняет): макс. расстояние {z:.2e}  "
              f"{'✅' if z < 1e-4 else '⚠️ проверь весь вывод'}")

        F = {"orig": [], "h": [], "v": []}
        for s in range(0, len(imgs), BATCH):
            x = imgs[s:s + BATCH]
            F["orig"].append(features(model, x, device))
            F["h"].append(features(model, x.flip(3), device))     # ось 3 = ширина: слева-направо
            F["v"].append(features(model, x.flip(2), device))     # ось 2 = высота: сверху-вниз
        F = {k: np.concatenate(v) for k, v in F.items()}           # [N, слои, группы, D]

        rng = np.random.default_rng(0)
        summary[name] = {g: [] for g in GROUPS}
        dist[name] = np.zeros((2, len(GROUPS), N_LAYERS, len(sel)))
        for gi, g in enumerate(GROUPS):
            for l in range(N_LAYERS):
                dh = cos_dist(F["orig"][:, l, gi], F["h"][:, l, gi])
                dv = cos_dist(F["orig"][:, l, gi], F["v"][:, l, gi])
                dist[name][0, gi, l], dist[name][1, gi, l] = dh, dv
                r, lo, hi = boot_ratio(dh, dv, rng)
                frac = float((dh < dv).mean())
                summary[name][g].append((r, lo, hi, float(dh.mean()), float(dv.mean()), frac))
                rows.append([name, weights, g, f"L{l:02d}", f"{dh.mean():.6f}", f"{dv.mean():.6f}",
                             f"{r:.5f}", f"{lo:.5f}", f"{hi:.5f}", f"{frac:.4f}", len(dh)])
        del model, F
        print(f"  готово ({time.time() - t0:.0f} с)")

    # --- таблица
    print("\nОтношение d(зеркало слева-направо) / d(зеркало сверху-вниз). < 1 = сеть меньше замечает зеркало слева-направо")
    for g in GROUPS:
        print(f"\n== {NAMES[g]} ==")
        print(f"слой   {MAIN}  [95% инт.]   d_Г / d_В ({MAIN})        доля сцен d_Г<d_В   случайные: отношение (мин–макс)")
        for l in range(N_LAYERS):
            r, lo, hi, mh, mv, frac = summary[MAIN][g][l]
            rr = [summary[n][g][l][0] for n, s in MODELS if s is not None]
            if np.isnan(r):
                print(f"L{l:02d}     —                  {mh:.2e} / {mv:.2e}        —")
                continue
            print(f"L{l:02d}  {r:5.2f}  [{lo:.2f}–{hi:.2f}]   {mh:.2e} / {mv:.2e}    {frac * 100:5.1f}%"
                  f"          {np.mean(rr):.2f} ({min(rr):.2f}–{max(rr):.2f})")

    print("\nПозиционные коды (веса сети): средний косинус клетки с зеркальной клеткой")
    for name, _ in MODELS:
        h, v = pos[name]
        print(f"  {name:6s} слева-направо {h:+.3f}   сверху-вниз {v:+.3f}")

    print(f"\nИТОГ на L{N_LAYERS - 1:02d} ({MAIN}: отношение < 1 и ниже всех случайных?):")
    for g in GROUPS:
        r, lo, hi, *_ = summary[MAIN][g][-1]
        rr = [summary[n][g][-1][0] for n, s in MODELS if s is not None]
        below1 = hi < 1.0
        below_r = hi < min(rr)
        print(f"  {NAMES[g]:18s} {MAIN} {r:.2f} [{lo:.2f}–{hi:.2f}]   случайные {min(rr):.2f}–{max(rr):.2f}   "
              f"ниже 1: {'да ✅' if below1 else 'нет'}   ниже всех случайных: {'да ✅' if below_r else 'нет'}")
    print(f"  (\"ниже\" = ВЕРХНЯЯ граница 95% интервала {MAIN} меньше этого числа)")

    if ARCH in BETS:                     # ставка Ерназара (реестр 33) — правило засчёта записано до запуска
        g, a, b = BETS[ARCH]
        r, lo, hi, *_ = summary[MAIN][g][-1]
        hit = a <= round(r, 2) <= b
        print(f"\nСТАВКА ЕРНАЗАРА (реестр 33, 7.10): {MAIN}, {NAMES[g]}, L{N_LAYERS - 1:02d} — диапазон {a:.2f}–{b:.2f}.  "
              f"Вышло {r:.2f} [{lo:.2f}–{hi:.2f}] -> {'ПОПАЛ ✅' if hit else 'НЕ ПОПАЛ ❌'}")

    with open(f"{STEM}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "weights", "group", "layer", "d_h", "d_v", "ratio_h_v", "ci_lo", "ci_hi", "frac_h_lt_v", "n_scenes"])
        w.writerows(rows)
    np.savez_compressed(f"{STEM}_dist.npz", ids=np.array([r["id"] for r in sel]), groups=np.array(GROUPS),
                        models=np.array([n for n, _ in MODELS]), dist=np.stack([dist[n] for n, _ in MODELS]))

    L = np.arange(N_LAYERS)
    fig, axes = plt.subplots(1, len(GROUPS), figsize=(6 * len(GROUPS), 4.3), sharey=True, squeeze=False)
    for ax, g in zip(axes[0], GROUPS):
        d = summary[MAIN][g]
        ax.plot(L, [x[0] for x in d], "o-", c="C3", label=MAIN_LONG)
        ax.fill_between(L, [x[1] for x in d], [x[2] for x in d], color="C3", alpha=0.15)
        for i, (n, s) in enumerate([m for m in MODELS if m[1] is not None]):
            ax.plot(L, [x[0] for x in summary[n][g]], "o--", c="gray", alpha=0.6, ms=3,
                    label="случайные веса" if i == 0 else None)
        ax.axhline(1, ls=":", c="k", lw=1)
        ax.set_title(f"{NAMES[g]}: d(зеркало ←→) / d(зеркало ↑↓)")
        step = 1 if N_LAYERS <= 13 else 4
        ax.set_xticks(L[::step], [f"L{l:02d}" for l in L[::step]], fontsize=7)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("< 1: зеркало слева-направо почти не замечается")
    fig.suptitle(f"Зеркала: {MAIN_LONG} против {len(MODELS) - 1} случайных сетей ({len(sel)} сцен из кругов/квадратов/крестов)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{STEM}.png", dpi=150)
    print(f"\nГрафик: {STEM}.png   Числа: {STEM}.csv, {STEM}_dist.npz   ({time.time() - t0:.0f} с)")


if __name__ == "__main__":
    main()

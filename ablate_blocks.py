"""
ablate_blocks.py — Часть 3, шаг 1: сама ли сеть DINO на слоях 4–12 стирает «лево/право»?
(реестр 43: дизайн, правило вывода и ставка записаны 5.11 ДО запуска)

ВОПРОС
  На слое 3 проба «какого цвета левый близнец» по дальнему фону даёт 73,0%, на слое 12 — 64,9%;
  «верхний» за то же время растёт с 76,1% до 83,5% (реестр 10–11). Что внутри блоков 4–12 портит «левый»?
  В каждом блоке две части:
    внимание (attention) — токены обмениваются информацией друг с другом («разговоры»);
    MLP — каждый токен переписывает свою запись сам («одиночная работа»).

ЧТО ДЕЛАЕМ
  «Выключаем» часть сети: её вклад в остаточный поток заменяем СРЕДНИМ этого вклада
  по всем 4000 картинкам (S и Sp), отдельно для каждой позиции токена (mean-ablation).
  Средние считаются на обычной (невыключенной) сети той же сети.
  Условия (для обученной DINO):
    clean        — обычная сеть;
    attn_04-12   — внимание выключено во всех блоках 4–12   } основные условия (правило вывода)
    mlp_04-12    — MLP выключен во всех блоках 4–12          }
    attn_04 … attn_12, mlp_04 … mlp_12 — по одному подслою (уточнение: где именно; 18 условий)
  Для 3 случайных DINO (seed 0/1/2) — только clean, attn_04-12, mlp_04-12 (контроль: везде ≈ 50%).
  Номер блока: L04 = выход блока 4 (L00 = вход в блок 1), как в extract.py.

ЧТО МЕРЯЕМ
  Слой 12 (выход блока 12, до финального norm — как в extract.py). Группы токенов как в probe_v2.py:
  дальний фон (основное), CLS (дополнительно), сам близнец-вопрос (положительный контроль).
  В каждом условии пробы «левый» и «верхний» обучаются ЗАНОВО (логрег C=0,1, те же 5 фолдов по сценам,
  функции probe_v2.run_probe и probe_v2.scene_info), 2000 сцен × (S, Sp).
  Δлевый  = точность «левый»(условие)  − точность «левый»(clean),
  Δверхний = точность «верхний»(условие) − точность «верхний»(clean), по одним и тем же сценам.
  Интервалы: парный бутстрэп по сценам, 10 000 повторов, seed 0.
  Поправка на число сравнений (Бонферрони): 2 основных условия — интервал 97,5%; 18 уточняющих — 99,72%.

ПРАВИЛО ВЫВОДА (реестр 43, записано до запуска), дальний фон, L12, для каждого из 2 основных условий:
  «стирает»      — нижняя граница Δлевый > 0  И  нижняя граница (Δлевый − Δверхний) > 0;
  «не причастна» — иначе, если верхняя граница (Δлевый − Δверхний) < 3 п.п.;
  «не различает» — во всех остальных случаях;
  пометка «разрушение», если точность «верхний» после выключения < 60%.
  Итог: какие части «стирают» → А только внимание / Б только MLP / В обе / Г ни одна.
  (Уточнение до запуска, 6.11: итог определяется только тем, какие части получили «стирает»;
   ярлык второй части — «не причастна» или «не различает» — печатается рядом.)

ВСТРОЕННЫЕ ПРОВЕРКИ
  1. Наш пошаговый прогон = собственный прогон блоков и model(x) (разница ~0).
  2. «Выключение собственным значением»: средние посчитаны по ОДНОЙ картинке → выключение всех частей
     всех 12 блоков на этой же картинке обязано дать обычный прогон (разница ~0). Проверяет индексы,
     номера блоков и подстановку.
  3. Если выключить ВСЁ во всех 12 блоках, CLS на слое 12 одинаков у всех картинок (CLS не видит картинку).
  4. clean = прежние результаты: probe_v2 (дальний фон L12 «левый» 64,875%, «верхний» 83,500%),
     сохранённые активации acts_v2/L12.npy, flip_test (зеркальное отношение CLS 0,336, патчи 0,284).
  5. L00: дальний фон и CLS ровно 50,0%. Контроль «близнец» > 90%. Случайные сети ≈ 50%.
  6. Отпечатки весов случайных сетей = extract.py.

ДОПОЛНИТЕЛЬНО (не правило): зеркальное отношение d(зеркало ←→)/d(зеркало ↑↓) на L12 в каждом условии
  (как flip_test.py: 631 сцена только из кругов/квадратов/крестов, картинка S, CLS и среднее по патчам).
  Если выключение возвращает «левый», сеть должна стать менее слепой к зеркалу (отношение ближе к 1).

ЗАПУСК (на Маке, из папки binding-vit, с зарядкой; долго — ~1,5–2 ч):
  caffeinate -i python ablate_blocks.py 2>&1 | tee ablate_blocks_log.txt
  Промежуточные результаты лежат в ~/binding-vit-data/ablate/ — при обрыве та же команда продолжит с места.

ЧЕГО ЭТОТ ОПЫТ НЕ ПОКАЖЕТ
  Он показывает, НУЖНО ли вычисление части сети для потери «левого», а не КАК она его стирает
  (это Часть 4 — головы — и шаг 2 — DAS). Выключение средним — вмешательство, которого сеть не видела
  при обучении, поэтому смотрим на «верхний» и на пометку «разрушение». Одна модель (DINO ViT-B/16).
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from PIL import Image

import probe_v2 as pv
from extract import load_model, fingerprint, pick_device, run_layers, MEAN, STD

DATA = "data_v2"
BLOCKS = list(range(4, 13))                       # блоки 4..12 (номер = номер слоя на выходе блока)
QUERIES = ["left", "upper"]
QWORD = {"left": "левый", "upper": "верхний"}
GROUPS = ["far_bg", "cls", "twin"]                # twin = близнец-вопрос (left_twin / upper_twin)
GNAME = {"far_bg": "дальний фон", "cls": "CLS", "twin": "близнец-вопрос (контроль)"}
SYM_SHAPES = {"circle", "square", "cross"}        # как в flip_test.py
N_BOOT = 10000
ALPHA_MAIN = 0.05 / 2                             # 2 основных условия
ALPHA_SUB = 0.05 / 18                             # 18 уточняющих условий
MARGIN = 0.03                                     # 3 п.п.
BROKEN = 0.60
BATCH = 16                                        # как в extract.py (ради побайтового совпадения с acts_v2)
EXPECTED_FP = {0: 10638.7832, 1: 10621.1621, 2: 10653.9980}   # из extract_rand*_log.txt
REF_PROBE = {("left", "far_bg"): 0.64875, ("upper", "far_bg"): 0.83500,
             ("left", "cls"): 0.63500, ("upper", "cls"): 0.83850}   # probe_v2.csv / probe_v2_upper.csv, L12
REF_FLIP = {"cls": 0.33571, "mean": 0.28404}                          # flip_test.csv, DINO, L12
# ставка Ерназара (реестр 43, 5.11 23:55, до запуска)
BET = dict(verdict="Б", attn_max=-5.1, mlp_min=8.1, conf="90%")


def conditions(full):
    c = [("clean", set(), set()),
         ("attn_04-12", set(BLOCKS), set()),
         ("mlp_04-12", set(), set(BLOCKS))]
    if full:
        c += [(f"attn_{b:02d}", {b}, set()) for b in BLOCKS]
        c += [(f"mlp_{b:02d}", set(), {b}) for b in BLOCKS]
    return c


# ------------------------------------------------------------------
# 1. Прогон с выключением частей блоков
# ------------------------------------------------------------------
@torch.no_grad()
def forward(model, x, attn_off=(), mlp_off=(), means=None, acc=None):
    """Возвращает (L00, L12). attn_off / mlp_off — номера блоков (1..12), где вклад заменяется средним.
    acc (словарь) — если задан, сюда копятся суммы вкладов (для подсчёта средних на обычной сети)."""
    h = model.prepare_tokens(x)
    l00 = h
    for b, blk in enumerate(model.blocks, start=1):
        y, _ = blk.attn(blk.norm1(h))              # вклад внимания (как Block.forward в DINO)
        if acc is not None:
            acc["attn"][b] += y.sum(0).cpu().double()
        if b in attn_off:
            y = means["attn"][b].to(y.device, y.dtype).expand_as(y)
        h = h + y
        m = blk.mlp(blk.norm2(h))                  # вклад MLP
        if acc is not None:
            acc["mlp"][b] += m.sum(0).cpu().double()
        if b in mlp_off:
            m = means["mlp"][b].to(m.device, m.dtype).expand_as(m)
        h = h + m
    return l00, h


def new_acc():
    z = lambda: torch.zeros(197, 768, dtype=torch.float64)
    return {"attn": {b: z() for b in range(1, 13)}, "mlp": {b: z() for b in range(1, 13)}, "n": 0}


def to_means(acc):
    return {k: {b: (acc[k][b] / acc["n"]).float() for b in range(1, 13)} for k in ("attn", "mlp")}


def norm_batch(u8):
    """u8: [B,224,224,3] uint8 -> нормированный тензор, ровно как extract.load_image."""
    x = torch.from_numpy(u8).permute(0, 3, 1, 2).float() / 255.0
    return (x - MEAN) / STD


# ------------------------------------------------------------------
# 2. Признаки групп токенов на слое 12
# ------------------------------------------------------------------
def group_feats(h, rows, ginfo):
    """h: [B,197,768] (слой) -> [B, 4, 768]: дальний фон, CLS, левый близнец, верхний близнец.
    Как в probe_v2: активации сначала округляются до float16 (так они лежали в acts_v2), потом среднее."""
    a = h.float().cpu().numpy().astype(np.float16).astype(np.float32)
    out = np.zeros((len(rows), 4, 768), np.float32)
    for i, r in enumerate(rows):
        g = ginfo[r // 2]
        for j, toks in enumerate((g["far_bg"], [0], g["left_twin"], g["upper_twin"])):
            if toks:
                out[i, j] = a[i, toks].mean(0)
    return out


def cos_dist(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return 1.0 - (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)


def flip_ratio(F, rng):
    """F: [N, 3(orig, ←→, ↑↓), 2(CLS, среднее по патчам), 768] -> {группа: (отношение, низ, верх)} как flip_test."""
    res = {}
    for gi, g in enumerate(("cls", "mean")):
        dh = cos_dist(F[:, 0, gi], F[:, 1, gi])
        dv = cos_dist(F[:, 0, gi], F[:, 2, gi])
        idx = rng.integers(0, len(dh), size=(1000, len(dh)))
        b = dh[idx].mean(1) / np.maximum(dv[idx].mean(1), 1e-15)
        res[g] = (float(dh.mean() / dv.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))
    return res


# ------------------------------------------------------------------
# 3. Статистика и правило
# ------------------------------------------------------------------
def boot_paired(arrs, alpha):
    """arrs: список массивов по сценам (одни и те же сцены). Парный бутстрэп, 10 000 повторов, seed 0.
    Возвращает [(среднее, низ, верх)] для каждого массива."""
    rng = np.random.default_rng(0)
    n = len(arrs[0])
    bs = [[] for _ in arrs]
    for _ in range(N_BOOT // 1000):
        idx = rng.integers(0, n, size=(1000, n))
        for k, a in enumerate(arrs):
            bs[k].append(a[idx].mean(1))
    out = []
    for a, b in zip(arrs, bs):
        b = np.concatenate(b)
        out.append((float(a.mean()), float(np.percentile(b, 100 * alpha / 2)), float(np.percentile(b, 100 * (1 - alpha / 2)))))
    return out


def verdict(dl, dd):
    """dl, dd = (среднее, низ, верх) для Δлевый и (Δлевый − Δверхний)."""
    if dl[1] > 0 and dd[1] > 0:
        return "стирает"
    if dd[2] < MARGIN:
        return "не причастна"
    return "не различает"


def overall(v_attn, v_mlp):
    a, m = v_attn == "стирает", v_mlp == "стирает"
    return {(True, False): "А", (False, True): "Б", (True, True): "В", (False, False): "Г"}[(a, m)]


OVERALL_TEXT = {"А": "стирает внимание (обмен информацией между токенами)",
                "Б": "стирает MLP (переработка каждого токена по отдельности)",
                "В": "стирают обе части", "Г": "ни одна часть по отдельности"}


# ------------------------------------------------------------------
# 4. Одна сеть: средние, все условия, признаки (с кэшем)
# ------------------------------------------------------------------
def run_net(name, model, conds, imgs, rows, ginfo, flip_rows, device, cache, tag):
    n_img = len(rows)
    fp = round(fingerprint(model), 4)
    model.eval().to(device)
    os.makedirs(cache, exist_ok=True)
    mpath = f"{cache}/{tag}{name}_means.pt"
    t0 = time.time()

    # --- обычный прогон: средние вкладов + признаки clean + L00
    cpath = f"{cache}/{tag}{name}_clean.npz"
    if os.path.exists(mpath) and os.path.exists(cpath) and int(np.load(cpath)["fp_ok"]) == int(fp * 10000):
        means = torch.load(mpath)
        print(f"  средние вкладов и clean взяты из кэша {cache}/")
    else:
        acc = new_acc()
        F, F0 = [], []
        for s in range(0, n_img, BATCH):
            rb = rows[s:s + BATCH]
            x = norm_batch(imgs[rb]).to(device)
            l00, h = forward(model, x, acc=acc)
            acc["n"] += len(rb)
            F.append(group_feats(h, rb, ginfo))
            F0.append(group_feats(l00, rb, ginfo))
        means = to_means(acc)
        torch.save(means, mpath)
        Ff = flip_feats(model, imgs, flip_rows, device, set(), set(), means)
        np.savez(cpath, F=np.concatenate(F), F0=np.concatenate(F0), flip=Ff, fp_ok=int(fp * 10000))
        print(f"  обычный прогон + средние вкладов: {time.time() - t0:.0f} с", flush=True)
    feats = {"clean": dict(np.load(cpath))}

    for cname, a_off, m_off in conds[1:]:
        p = f"{cache}/{tag}{name}_{cname}.npz"
        if os.path.exists(p) and int(np.load(p)["fp_ok"]) == int(fp * 10000):
            feats[cname] = dict(np.load(p))
            continue
        t1 = time.time()
        F = []
        for s in range(0, n_img, BATCH):
            rb = rows[s:s + BATCH]
            x = norm_batch(imgs[rb]).to(device)
            _, h = forward(model, x, a_off, m_off, means)
            F.append(group_feats(h, rb, ginfo))
        Ff = flip_feats(model, imgs, flip_rows, device, a_off, m_off, means)
        np.savez(p, F=np.concatenate(F), flip=Ff, fp_ok=int(fp * 10000))
        feats[cname] = dict(np.load(p))
        print(f"  {cname:11s} {time.time() - t1:5.0f} с", flush=True)
    return feats, means


@torch.no_grad()
def flip_feats(model, imgs, flip_rows, device, a_off, m_off, means):
    """[N, 3, 2, 768]: картинка S, зеркало ←→, зеркало ↑↓; CLS и среднее по патчам на L12 (float32, как flip_test)."""
    out = []
    for s in range(0, len(flip_rows), BATCH):
        x = norm_batch(imgs[flip_rows[s:s + BATCH]]).to(device)
        fs = []
        for xx in (x, x.flip(3), x.flip(2)):            # ось 3 = ширина (←→), ось 2 = высота (↑↓)
            _, h = forward(model, xx, a_off, m_off, means)
            fs.append(torch.stack([h[:, 0], h[:, 1:].mean(1)], 1).float().cpu().numpy())
        out.append(np.stack(fs, 1))
    return np.concatenate(out)


def probe_all(F, labels, folds, scene_idx, groups=GROUPS):
    """F: [N_img, 4, 768]. Возвращает {(вопрос, группа): баллы по сценам} (логрег, как probe_v2)."""
    gi = {"far_bg": 0, "cls": 1}
    res = {}
    for q in QUERIES:
        yt, yr = labels[q]
        for g in groups:
            j = gi.get(g, 2 if q == "left" else 3)
            keep = np.abs(F[:, j]).sum(1) > 0          # у картинки есть токены этой группы
            per, nw = pv.run_probe(F[keep, j], yt[keep], yr[keep], scene_idx[keep], folds)
            res[(q, g)] = (per["logreg"], nw)
    return res


# ------------------------------------------------------------------
# 5. main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="только clean и 2 основных условия (без 18 уточняющих)")
    ap.add_argument("--limit", type=int, default=0, help="только для проверки: первые N сцен")
    ap.add_argument("--main-seed", type=int, default=None,
                    help="только для проверки в облаке: случайные веса seed N вместо обученной DINO")
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--cache", default=os.path.expanduser("~/binding-vit-data/ablate"))
    args = ap.parse_args()
    stem = "ablate_blocks" + (f"_test{args.limit}" if args.limit else "")
    tag = f"test{args.limit}_" if args.limit else ""
    if args.main_seed is not None:
        tag += f"mainseed{args.main_seed}_"
    pv.MODELS = ["logreg"]
    t_start = time.time()

    meta_all = [json.loads(l) for l in open(f"{DATA}/meta.jsonl")]
    # цвета и фолды — по ВСЕМ сценам, ровно как в probe_v2 (при --limit берём первые N)
    colors = sorted({o[k] for r in meta_all for o in r["objects"] for k in ("color", "color_Sp")})
    folds_all = np.random.default_rng(0).permutation(len(meta_all)) % pv.N_FOLDS
    meta = meta_all[:args.limit] if args.limit else meta_all
    if args.limit:
        print(f"⚠️ ПРОВЕРОЧНЫЙ прогон: только {len(meta)} сцен (--limit). Для настоящих чисел запускай без --limit.")
    # группы токенов и ответы — функциями probe_v2 (те же, что в прежних опытах)
    ginfo, labels = [dict() for _ in meta], {q: ([], []) for q in QUERIES}
    for q in QUERIES:
        pv.QUERY = q
        pv.GROUPS[3] = f"{q}_twin"
        for i, rec in enumerate(meta):
            groups, lab = pv.scene_info(rec, colors)
            ginfo[i]["far_bg"], ginfo[i][f"{q}_twin"] = groups["far_bg"], groups[f"{q}_twin"]
            for v in ("S", "Sp"):
                labels[q][0].append(lab[v][0])
                labels[q][1].append(lab[v][1])
    labels = {q: (np.array(a), np.array(b)) for q, (a, b) in labels.items()}
    n_sc = len(meta)
    scene_idx = np.repeat(np.arange(n_sc), 2)
    folds = folds_all[:n_sc]
    print(f"Сцен: {n_sc}, картинок (S, Sp): {2 * n_sc}. Группы и ответы — probe_v2.scene_info (проверка пикселей ✅)")
    print(f"Проба: логрег C={pv.C_REG}, {pv.N_FOLDS} фолдов по сценам (та же перестановка, что в probe_v2). Слой чтения: L12")

    # картинки в память один раз (uint8): строки 2i = S, 2i+1 = Sp
    imgs = np.zeros((2 * n_sc, 224, 224, 3), np.uint8)
    for i, rec in enumerate(meta):
        for k, v in enumerate(("S", "Sp")):
            imgs[2 * i + k] = np.asarray(Image.open(f"{DATA}/images/{rec['id']}_{v}.png").convert("RGB"))
    rows = np.arange(2 * n_sc)
    flip_rows = np.array([2 * i for i, r in enumerate(meta) if all(o["shape"] in SYM_SHAPES for o in r["objects"])])
    print(f"Для зеркальной проверки (только круг/квадрат/крест, картинка S): {len(flip_rows)} сцен")

    device = pick_device()
    full = not args.quick
    nets = [("DINO", args.main_seed)] + [(f"rand{s}", s) for s in range(args.n_random)]
    all_res, all_flip = {}, {}
    for name, seed in nets:
        model, weights, _ = load_model("dino", seed)
        fp = fingerprint(model)
        if name == "DINO" and seed is None:
            print(f"\n### {name}: {weights}   отпечаток весов {fp:.4f}   устройство: {device}")
        elif name.startswith("rand") and seed in EXPECTED_FP:
            ok = abs(fp - EXPECTED_FP[seed]) < 0.01
            print(f"\n### {name}: {weights}   отпечаток весов {fp:.4f}  "
                  f"{'✅ совпадает с extract.py' if ok else '⚠️ НЕ совпадает — проверь весь вывод'}")
        else:
            print(f"\n### {name}: {weights}   отпечаток весов {fp:.4f}  (проверочный прогон: случайные веса в роли DINO)")
        model.eval().to(device)
        if name == "DINO":
            run_checks(model, imgs, device, args)
        conds = conditions(full) if name == "DINO" else conditions(False)
        feats, _ = run_net(name, model, conds, imgs, rows, ginfo, flip_rows, device, args.cache, tag)
        # пробы
        t1 = time.time()
        res = {}
        for cname, _, _ in conds:
            grp = GROUPS if cname in ("clean", "attn_04-12", "mlp_04-12") else GROUPS[:2]
            res[cname] = probe_all(feats[cname]["F"], labels, folds, scene_idx, grp)
        res["L00"] = probe_all(feats["clean"]["F0"], labels, folds, scene_idx, GROUPS[:2])
        all_res[name] = res
        all_flip[name] = {c: flip_ratio(feats[c]["flip"], np.random.default_rng(0)) for c, _, _ in conds}
        print(f"  пробы: {time.time() - t1:.0f} с", flush=True)
        if name == "DINO" and args.main_seed is None and not args.limit:
            check_old(feats["clean"], res["clean"], all_flip[name]["clean"], meta)
        del model

    report(all_res, all_flip, nets, full, stem, args)
    print(f"\nВсего: {time.time() - t_start:.0f} с")


# ------------------------------------------------------------------
# 6. Встроенные проверки
# ------------------------------------------------------------------
@torch.no_grad()
def run_checks(model, imgs, device, args):
    x = norm_batch(imgs[:4]).to(device)
    # 1. наш прогон = блоки модели и model(x)
    _, h = forward(model, x)
    ref = run_layers(model, x)[-1]
    d1 = float((h - ref).abs().max())
    d1b = float((model.norm(h)[:, 0] - model(x)).abs().max())
    print(f"Проверка 1 (наш прогон = прогон блоков модели / model(x)): макс. разница {d1:.2e} / {d1b:.2e}  "
          f"{'✅' if max(d1, d1b) < 1e-3 else '⚠️ проверь весь вывод'}")
    # 2. выключение собственным значением: средние по одной картинке -> выключить всё во всех блоках
    x1 = x[:1]
    acc = new_acc()
    _, h_clean = forward(model, x1, acc=acc)
    acc["n"] = 1
    m1 = to_means(acc)
    _, h_self = forward(model, x1, set(range(1, 13)), set(range(1, 13)), m1)
    d2 = float((h_self - h_clean).abs().max())
    print(f"Проверка 2 (выключение «собственным значением» во всех 24 частях = обычный прогон): макс. разница {d2:.2e}  "
          f"{'✅' if d2 < 1e-3 else '⚠️ проверь весь вывод'}")
    # 3. выключить всё во всех блоках средними по 32 картинкам -> CLS одинаков у всех картинок
    xs = norm_batch(imgs[:32]).to(device)
    acc = new_acc()
    for s in range(0, 32, BATCH):
        forward(model, xs[s:s + BATCH], acc=acc)
    acc["n"] = 32
    mm = to_means(acc)
    _, hh = forward(model, xs, set(range(1, 13)), set(range(1, 13)), mm)
    d3 = float((hh[:, 0] - hh[:1, 0]).abs().max())
    print(f"Проверка 3 (всё выключено во всех блоках -> CLS на L12 одинаков у всех картинок): макс. разница {d3:.2e}  "
          f"{'✅' if d3 < 1e-3 else '⚠️ проверь весь вывод'}")


def check_old(fc, rc, flip_c, meta):
    """Проверка 4: clean = прежние результаты (только на Маке, обученная DINO, все 2000 сцен)."""
    print("\nПроверка 4 (обычная сеть = прежние результаты):")
    for (q, g), ref in REF_PROBE.items():
        acc = float(rc[(q, g)][0].mean())
        print(f"  проба «{QWORD[q]}», {GNAME[g]:11s} L12: {acc * 100:7.3f}%  (probe_v2: {ref * 100:7.3f}%)  "
              f"{'✅ точно' if abs(acc - ref) < 5e-6 else ('✅ (разница < 0,5 п.п.)' if abs(acc - ref) < 0.005 else '⚠️ проверь весь вывод')}")
    for g, ref in REF_FLIP.items():
        r = flip_c[g][0]
        print(f"  зеркальное отношение {('CLS' if g == 'cls' else 'среднее по патчам'):18s} L12: {r:.5f}  (flip_test: {ref:.5f})  "
              f"{'✅' if abs(r - ref) < 0.002 else '⚠️ проверь весь вывод'}")
    if os.path.exists("acts_v2/index.csv"):
        idx = {(r["id"], r["version"]): int(r["row"]) for r in csv.DictReader(open("acts_v2/index.csv"))}
        A = np.load("acts_v2/L12.npy", mmap_mode="r")
        ids = [meta[0]["id"], meta[1]["id"]]
        # пересчитываем токены 2 сцен (S и Sp) и сравниваем с сохранёнными float16
        print(f"  (сохранённые активации acts_v2/L12.npy сверяются по признакам групп сцен {ids[0]}, {ids[1]})")
        d = 0.0
        for i, sid in enumerate(ids):
            for k, v in enumerate(("S", "Sp")):
                tok = np.asarray(A[idx[(sid, v)]], dtype=np.float32)
                d = max(d, float(np.abs(tok[0] - fc["F"][2 * i + k, 1]).max()))
        print(f"  CLS L12 = acts_v2 (макс. разница по 4 картинкам): {d:.2e}  {'✅' if d < 1e-2 else '⚠️ проверь весь вывод'}")


# ------------------------------------------------------------------
# 7. Итог, правило, CSV, график
# ------------------------------------------------------------------
def report(all_res, all_flip, nets, full, stem, args):
    D = all_res["DINO"]
    conds = [c for c, _, _ in conditions(full)]
    rows_csv = []
    warn = sum(nw for net in all_res.values() for cond in net.values() for (_, nw) in cond.values())

    # проверка 5
    l00 = [float(all_res[n]["L00"][(q, g)][0].mean()) for n, _ in nets for q in QUERIES for g in ("far_bg", "cls")]
    print(f"\nПроверка 5а (L00: дальний фон и CLS ровно 50,0% у всех сетей): "
          f"{'✅' if all(abs(a - 0.5) < 1e-12 for a in l00) else '⚠️ проверь весь вывод'}  {sorted(set(round(a * 100, 3) for a in l00))}")
    tw = [float(all_res[n][c][(q, 'twin')][0].mean()) for n, _ in nets for c in ("clean", "attn_04-12", "mlp_04-12") for q in QUERIES]
    print(f"Проверка 5б (контроль «близнец-вопрос» > 90% в обычной сети и обоих основных условиях, все сети): "
          f"{'✅' if all(a > 0.9 for a in tw) else '⚠️ проверь весь вывод'}  (мин {min(tw) * 100:.1f}%)")
    rnd = [float(all_res[n][c][(q, g)][0].mean()) for n, _ in nets[1:] for c in ("clean", "attn_04-12", "mlp_04-12")
           for q in QUERIES for g in ("far_bg", "cls")]
    if rnd:
        print(f"Проверка 5в (случайные сети: «левый» и «верхний» ≈ 50% во всех условиях): "
              f"{'✅' if all(0.44 < a < 0.56 for a in rnd) else '⚠️ проверь весь вывод'}  ({min(rnd) * 100:.1f}–{max(rnd) * 100:.1f}%)")
    print(f"Предупреждений «обучение не сошлось» всего: {warn}  {'✅' if warn == 0 else '⚠️ проверь весь вывод'}")

    # точности
    print("\nТочность на L12 (логрег), %:  «левый» / «верхний»")
    print(f"{'условие':12s}  {'дальний фон':>17s}   {'CLS':>17s}   зеркало L12 (CLS / патчи)")
    for c in conds:
        fb = [D[c][(q, 'far_bg')][0].mean() * 100 for q in QUERIES]
        cl = [D[c][(q, 'cls')][0].mean() * 100 for q in QUERIES]
        fr = all_flip["DINO"][c]
        print(f"{c:12s}  {fb[0]:7.1f} / {fb[1]:5.1f}     {cl[0]:7.1f} / {cl[1]:5.1f}     {fr['cls'][0]:.2f} / {fr['mean'][0]:.2f}")
    for n, _ in nets[1:]:
        s = "   ".join(f"{c}: {all_res[n][c][('left', 'far_bg')][0].mean() * 100:.1f}/{all_res[n][c][('upper', 'far_bg')][0].mean() * 100:.1f}"
                       for c in ("clean", "attn_04-12", "mlp_04-12"))
        print(f"  {n} (дальний фон, левый/верхний): {s}")

    # Δ и правило
    def deltas(net, c, g, alpha):
        Lc, Uc = net["clean"][("left", g)][0], net["clean"][("upper", g)][0]
        L, U = net[c][("left", g)][0], net[c][("upper", g)][0]
        dl, du = L - Lc, U - Uc
        return boot_paired([dl, du, dl - du], alpha)

    verdicts = {}
    print("\n" + "=" * 100)
    print(f"ОСНОВНЫЕ УСЛОВИЯ — дальний фон, L12, интервал {100 * (1 - ALPHA_MAIN):.1f}% (поправка на 2 сравнения), п.п.")
    for c in ("attn_04-12", "mlp_04-12"):
        dl, du, dd = deltas(D, c, "far_bg", ALPHA_MAIN)
        v = verdict(dl, dd)
        accU = float(D[c][("upper", "far_bg")][0].mean())
        broken = accU < BROKEN
        verdicts[c] = (v, dl, du, dd, broken)
        print(f"  {c:11s} Δлевый {dl[0] * 100:+6.1f} [{dl[1] * 100:+.1f} … {dl[2] * 100:+.1f}]   "
              f"Δверхний {du[0] * 100:+6.1f} [{du[1] * 100:+.1f} … {du[2] * 100:+.1f}]   "
              f"Δлевый − Δверхний {dd[0] * 100:+6.1f} [{dd[1] * 100:+.1f} … {dd[2] * 100:+.1f}]   → «{v.upper()}»"
              + ("   ⚠️ пометка «РАЗРУШЕНИЕ» (верхний < 60%)" if broken else ""))
        rows_csv.append(["DINO", c, "far_bg", "main", *[f"{x:.5f}" for t in (dl, du, dd) for x in t], v, int(broken)])
    va, vm = verdicts["attn_04-12"][0], verdicts["mlp_04-12"][0]
    ov = overall(va, vm)
    print(f"\nВЫВОД ПО ПРАВИЛУ (реестр 43, записано до запуска): «{ov}» — {OVERALL_TEXT[ov]}")
    print(f"  внимание 4–12: «{va}»;  MLP 4–12: «{vm}»")
    print("  Оговорки: одна модель (DINO ViT-B/16); выключение средним — вмешательство вне обучения;"
          " опыт показывает, нужна ли часть для потери «левого», а не как она стирает.")

    # ставка
    dla = round(verdicts["attn_04-12"][1][0] * 100, 1)
    dlm = round(verdicts["mlp_04-12"][1][0] * 100, 1)
    print(f"\nСТАВКА ЕРНАЗАРА (реестр 43, 5.11 23:55, до запуска):")
    print(f"  43a вывод «{BET['verdict']}» -> вышло «{ov}»   {'ПОПАЛ ✅' if ov == BET['verdict'] else 'НЕ ПОПАЛ ❌'}")
    print(f"  43b Δлевый, внимание выключено: ≤ {BET['attn_max']} -> вышло {dla:+.1f}   {'ПОПАЛ ✅' if dla <= BET['attn_max'] else 'НЕ ПОПАЛ ❌'}")
    print(f"  43c Δлевый, MLP выключен: ≥ +{BET['mlp_min']} -> вышло {dlm:+.1f}   {'ПОПАЛ ✅' if dlm >= BET['mlp_min'] else 'НЕ ПОПАЛ ❌'}")
    print(f"  (уверенность в 43a: {BET['conf']})")

    # дополнительно: CLS, без правила
    print(f"\nДополнительно (НЕ правило): CLS, L12, интервал {100 * (1 - ALPHA_MAIN):.1f}%")
    for c in ("attn_04-12", "mlp_04-12"):
        dl, du, dd = deltas(D, c, "cls", ALPHA_MAIN)
        print(f"  {c:11s} Δлевый {dl[0] * 100:+6.1f} [{dl[1] * 100:+.1f} … {dl[2] * 100:+.1f}]   "
              f"Δверхний {du[0] * 100:+6.1f}   Δлевый − Δверхний {dd[0] * 100:+6.1f} [{dd[1] * 100:+.1f} … {dd[2] * 100:+.1f}]   (по правилу было бы «{verdict(dl, dd)}»)")
        rows_csv.append(["DINO", c, "cls", "extra", *[f"{x:.5f}" for t in (dl, du, dd) for x in t], verdict(dl, dd), ""])

    # уточнение
    if full:
        print(f"\nУТОЧНЕНИЕ — по одному подслою, дальний фон, L12, интервал {100 * (1 - ALPHA_SUB):.2f}% (поправка на 18), п.п.")
        for c in conds[3:]:
            dl, du, dd = deltas(D, c, "far_bg", ALPHA_SUB)
            v = verdict(dl, dd)
            accU = float(D[c][("upper", "far_bg")][0].mean())
            print(f"  {c:8s} Δлевый {dl[0] * 100:+6.1f} [{dl[1] * 100:+.1f} … {dl[2] * 100:+.1f}]   Δверхний {du[0] * 100:+6.1f}   "
                  f"Δл − Δв {dd[0] * 100:+6.1f} [{dd[1] * 100:+.1f} … {dd[2] * 100:+.1f}]   «{v}»" + ("   ⚠️ разрушение" if accU < BROKEN else ""))
            rows_csv.append(["DINO", c, "far_bg", "sub", *[f"{x:.5f}" for t in (dl, du, dd) for x in t], v, int(accU < BROKEN)])

    # случайные сети: Δ для справки
    for n, _ in nets[1:]:
        for c in ("attn_04-12", "mlp_04-12"):
            dl, du, dd = deltas(all_res[n], c, "far_bg", ALPHA_MAIN)
            rows_csv.append([n, c, "far_bg", "random", *[f"{x:.5f}" for t in (dl, du, dd) for x in t], verdict(dl, dd), ""])

    # зеркало: справка
    print("\nДополнительно (НЕ правило): зеркальное отношение на L12 [95% инт.] (1 = не слепа к зеркалу ←→)")
    for c in ("clean", "attn_04-12", "mlp_04-12"):
        fr = all_flip["DINO"][c]
        rr = [all_flip[n][c]["cls"][0] for n, _ in nets[1:]]
        print(f"  {c:11s} CLS {fr['cls'][0]:.2f} [{fr['cls'][1]:.2f}–{fr['cls'][2]:.2f}]   "
              f"патчи {fr['mean'][0]:.2f} [{fr['mean'][1]:.2f}–{fr['mean'][2]:.2f}]"
              + (f"   случайные (CLS) {min(rr):.2f}–{max(rr):.2f}" if rr else ""))

    # файлы
    with open(f"{stem}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["net", "condition", "group", "kind", "d_left", "d_left_lo", "d_left_hi", "d_upper", "d_upper_lo",
                    "d_upper_hi", "d_diff", "d_diff_lo", "d_diff_hi", "verdict", "broken"])
        w.writerows(rows_csv)
        w.writerow([])
        w.writerow(["net", "condition", "query", "group", "acc", "n_scenes", "flip_ratio_cls", "flip_ratio_mean"])
        for n, _ in nets:
            for c in all_res[n]:
                for (q, g), (sc, _) in all_res[n][c].items():
                    fr = all_flip[n].get(c, {"cls": (np.nan,), "mean": (np.nan,)})
                    w.writerow([n, c, q, g, f"{sc.mean():.5f}", len(sc), f"{fr['cls'][0]:.5f}", f"{fr['mean'][0]:.5f}"])
    np.savez(f"{stem}_scores.npz", **{f"{n}|{c}|{q}|{g}": sc for n in all_res for c in all_res[n]
                                      for (q, g), (sc, _) in all_res[n][c].items()})
    plot(D, conds, full, stem)
    print(f"\nГрафик: {stem}.png   Числа: {stem}.csv, {stem}_scores.npz")


def plot(D, conds, full, stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cs = conds[1:]
    dl = [float((D[c][("left", "far_bg")][0] - D["clean"][("left", "far_bg")][0]).mean() * 100) for c in cs]
    du = [float((D[c][("upper", "far_bg")][0] - D["clean"][("upper", "far_bg")][0]).mean() * 100) for c in cs]
    x = np.arange(len(cs))
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(cs) + 2), 4))
    ax.bar(x - 0.2, dl, 0.4, label="Δ «левый»", color="C0")
    ax.bar(x + 0.2, du, 0.4, label="Δ «верхний»", color="C1")
    ax.axhline(0, c="k", lw=0.8)
    if full:
        ax.axvline(1.5, c="gray", ls=":", lw=1)
    ax.set_xticks(x, cs, rotation=60, fontsize=8)
    ax.set_ylabel("изменение точности, п.п.")
    ax.set_title("DINO, дальний фон, L12: что меняется, если выключить часть блоков (средним)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=150)


if __name__ == "__main__":
    main()

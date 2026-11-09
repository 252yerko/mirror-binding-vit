"""
ablate_clip.py — Часть 3, шаг 1 на CLIP: сама ли сеть CLIP на поздних блоках стирает «лево/право»?
(реестр 53: дизайн, правило вывода и ставка записаны 9.11 ДО написания кода)

ВОПРОС
  На DINO (ablate_blocks.py, реестр 46) потерю «левого» делают и внимание, и MLP блоков 4–12 (вывод «В»).
  Переносится ли это на другую сеть? CLIP (OpenAI ViT-B/16) обучали без отражений; к зеркалу она слепа слабее
  (отношение на L12: CLS 0,65, патчи 0,61; у DINO 0,34 / 0,28), разрыв «верхний − левый» на дальнем фоне L12
  вдвое меньше (9,0 п.п. против 18,6).
  У CLIP «левый» на дальнем фоне растёт до L06 (72,5%) и падает к L12 (63,15%) — потеря идёт в блоках 7–12.
  «Верхний»: пик 77,8% на L07, на L12 72,15% (probe_v2_clip.csv, probe_v2_clip_upper.csv).

ЧТО ДЕЛАЕМ (как в ablate_blocks.py, другая сеть и другие блоки)
  Принцип выбора блоков тот же, что у DINO: от пика «левого» до конца (DINO: пик L03 → блоки 4–12;
  CLIP: пик L06 → блоки 7–12).
  «Выключаем» часть сети: её вклад в остаточный поток заменяем СРЕДНИМ этого вклада по всем 4000 картинкам
  (S и Sp), отдельно для каждой позиции токена (mean-ablation). Средние считаются на обычной сети.
  Условия (для обученной CLIP):
    clean        — обычная сеть;
    attn_07-12   — внимание выключено во всех блоках 7–12   } основные условия (правило вывода)
    mlp_07-12    — MLP выключен во всех блоках 7–12          }
    attn_04-12, mlp_04-12 — то же на блоках 4–12, как у DINO (СПРАВКА, не правило: прямое сравнение с DINO)
    attn_07 … attn_12, mlp_07 … mlp_12 — по одному подслою (уточнение: где именно; 12 условий)
  Для 3 случайных CLIP (seed 0/1/2) — только clean, attn_07-12, mlp_07-12 (контроль: везде ≈ 50%).
  Номер блока: L07 = выход блока 7; L00 = вход в блок 1 = после ln_pre (как в extract.py для CLIP).
  CLIP загружается как в extract.py: timm vit_base_patch16_clip_quickgelu_224.openai, нормировка CLIP.

ЧТО МЕРЯЕМ (как в ablate_blocks.py)
  Слой 12 (выход блока 12, до финального norm). Группы токенов как в probe_v2.py: дальний фон (основное),
  CLS (дополнительно), сам близнец-вопрос (положительный контроль). В каждом условии пробы «левый» и
  «верхний» обучаются ЗАНОВО (логрег C=0,1, те же 5 фолдов по сценам; функции probe_v2 через ablate_blocks).
  Δлевый = точность «левый»(условие) − точность «левый»(clean); Δверхний — так же; по одним и тем же сценам.
  Интервалы: парный бутстрэп по сценам, 10 000 повторов, seed 0.
  Поправка на число сравнений (Бонферрони): 2 основных условия — интервал 97,5%; 12 уточняющих — 99,58%.

ПРАВИЛО ВЫВОДА (реестр 53 = реестр 43, записано до кода), дальний фон, L12, для каждого из 2 основных условий:
  «стирает»      — нижняя граница Δлевый > 0  И  нижняя граница (Δлевый − Δверхний) > 0;
  «не причастна» — иначе, если верхняя граница (Δлевый − Δверхний) < 3 п.п.;
  «не различает» — во всех остальных случаях;
  пометка «разрушение», если точность «верхний» после выключения < 60%.
  Итог — только по тому, какие части получили «стирает»: А только внимание / Б только MLP / В обе / Г ни одна;
  ярлык второй части печатается рядом. Код правила — тот же, что в ablate_blocks.py (verdict, overall).

СТАВКА ЕРНАЗАРА (реестр 53, 9.11 09:45, уверенность уточнена 09:47 — до кода)
  53a вывод «В»; 53b Δлевый, внимание 7–12 выключено: Г = от +2,1 до +5,0; 53c Δлевый, MLP 7–12 выключен:
  Д = от +5,1 до +8,0; уверенность в 53a 45%. Засчёт: точечная оценка, округлённая до 0,1 п.п., границы включены.

ВСТРОЕННЫЕ ПРОВЕРКИ
  1. Наш пошаговый прогон = прогон блоков модели (extract.run_layers) и = model(x) (через norm и head CLIP).
  2. «Выключение собственным значением»: средние по ОДНОЙ картинке → выключение всех 24 частей на этой же
     картинке обязано дать обычный прогон.
  3. Всё выключено во всех 12 блоках → CLS на L12 одинаков у всех картинок.
  4. clean = прежние результаты: probe_v2_clip (дальний фон L12 «левый» 63,150%, «верхний» 72,150%;
     CLS 63,800% / 73,125%), flip_test_clip (зеркальное отношение CLS 0,65358, патчи 0,61261),
     сохранённые активации acts_v2_clip/L12.npy.
  5. L00: дальний фон и CLS ровно 50,0%. Контроль «близнец» > 90%. Случайные сети ≈ 50%.
  6. Отпечатки весов случайных сетей = extract.py.

ДОПОЛНИТЕЛЬНО (не правило): зеркальное отношение d(←→)/d(↑↓) на L12 в каждом основном и справочном условии
  (как flip_test.py: 631 сцена только из кругов/квадратов/крестов, картинка S, CLS и среднее по патчам).

ЗАПУСК (на Маке, из папки binding-vit, с зарядкой; ~1 ч):
  caffeinate -i python ablate_clip.py 2>&1 | tee ablate_clip_log.txt
  Промежуточные результаты лежат в ~/binding-vit-data/ablate_clip/ — при обрыве та же команда продолжит с места.

ЧЕГО ЭТОТ ОПЫТ НЕ ПОКАЖЕТ
  Как и на DINO: НУЖНО ли вычисление части сети для потери «левого», а не КАК она его стирает.
  Выключение средним — вмешательство вне обучения, поэтому смотрим на «верхний» и пометку «разрушение».
  Блоки у CLIP (7–12) и DINO (4–12) разные по числу; прямое сравнение — только справка attn_04-12 / mlp_04-12.
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from PIL import Image

import ablate_blocks as ab          # общие функции шага 1: пробы, бутстрэп, правило (файл не меняется)
import probe_v2 as pv
from extract import load_model, fingerprint, pick_device, run_layers, CLIP_MEAN, CLIP_STD

DATA = ab.DATA
BLOCKS = list(range(7, 13))                         # блоки 7..12 (после пика «левого» на L06)
BLOCKS_DINO = list(range(4, 13))                    # справка: те же блоки, что у DINO
QUERIES, QWORD, GROUPS, GNAME = ab.QUERIES, ab.QWORD, ab.GROUPS, ab.GNAME
ALPHA_MAIN = 0.05 / 2                               # 2 основных условия
ALPHA_SUB = 0.05 / 12                               # 12 уточняющих условий (6 блоков × внимание/MLP)
BROKEN = ab.BROKEN
BATCH = ab.BATCH
EXPECTED_FP = {0: 10638.7832, 1: 10621.1621, 2: 10653.9980}   # из extract_clip_rand*_log.txt / flip_test_clip_log.txt
REF_PROBE = {("left", "far_bg"): 0.63150, ("upper", "far_bg"): 0.72150,
             ("left", "cls"): 0.63800, ("upper", "cls"): 0.73125}   # probe_v2_clip.csv / probe_v2_clip_upper.csv, L12
REF_FLIP = {"cls": 0.65358, "mean": 0.61261}                          # flip_test_clip.csv, CLIP, L12
MAIN = ("attn_07-12", "mlp_07-12")
REF_DINO = ("attn_04-12", "mlp_04-12")
# ставка Ерназара (реестр 53, 9.11 09:45, уверенность уточнена 09:47, до кода)
BET = dict(verdict="В", attn=(2.1, 5.0), mlp=(5.1, 8.0), conf="45%")


def conditions(full):
    c = [("clean", set(), set()),
         ("attn_07-12", set(BLOCKS), set()),
         ("mlp_07-12", set(), set(BLOCKS))]
    if full:
        c += [("attn_04-12", set(BLOCKS_DINO), set()),
              ("mlp_04-12", set(), set(BLOCKS_DINO))]
        c += [(f"attn_{b:02d}", {b}, set()) for b in BLOCKS]
        c += [(f"mlp_{b:02d}", set(), {b}) for b in BLOCKS]
    return c


# ------------------------------------------------------------------
# 1. Прогон CLIP (timm) с выключением частей блоков
# ------------------------------------------------------------------
@torch.no_grad()
def forward(model, x, attn_off=(), mlp_off=(), means=None, acc=None):
    """Как ablate_blocks.forward, но для CLIP в timm: вход как extract.run_layers (ln_pre → L00),
    блок как timm Block.forward: h + ls1(attn(norm1(h))), затем h + ls2(mlp(norm2(h)))."""
    h = model.norm_pre(model.patch_drop(model._pos_embed(model.patch_embed(x))))
    l00 = h
    for b, blk in enumerate(model.blocks, start=1):
        y = blk.drop_path1(blk.ls1(blk.attn(blk.norm1(h))))    # вклад внимания
        if acc is not None:
            acc["attn"][b] += y.sum(0).cpu().double()
        if b in attn_off:
            y = means["attn"][b].to(y.device, y.dtype).expand_as(y)
        h = h + y
        m = blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(h))))      # вклад MLP
        if acc is not None:
            acc["mlp"][b] += m.sum(0).cpu().double()
        if b in mlp_off:
            m = means["mlp"][b].to(m.device, m.dtype).expand_as(m)
        h = h + m
    return l00, h


def norm_batch(u8):
    """u8: [B,224,224,3] uint8 -> нормированный тензор, ровно как extract.load_image(..., arch="clip")."""
    x = torch.from_numpy(u8).permute(0, 3, 1, 2).float() / 255.0
    return (x - CLIP_MEAN) / CLIP_STD


@torch.no_grad()
def flip_feats(model, imgs, flip_rows, device, a_off, m_off, means):
    """[N, 3, 2, 768]: картинка S, зеркало ←→, зеркало ↑↓; CLS и среднее по патчам на L12 (как flip_test)."""
    out = []
    for s in range(0, len(flip_rows), BATCH):
        x = norm_batch(imgs[flip_rows[s:s + BATCH]]).to(device)
        fs = []
        for xx in (x, x.flip(3), x.flip(2)):            # ось 3 = ширина (←→), ось 2 = высота (↑↓)
            _, h = forward(model, xx, a_off, m_off, means)
            fs.append(torch.stack([h[:, 0], h[:, 1:].mean(1)], 1).float().cpu().numpy())
        out.append(np.stack(fs, 1))
    return np.concatenate(out)


def run_net(name, model, conds, imgs, rows, ginfo, flip_rows, device, cache, tag):
    """Как ablate_blocks.run_net (средние на обычной сети, все условия, кэш), с CLIP-прогоном."""
    n_img = len(rows)
    fp = round(fingerprint(model), 4)
    model.eval().to(device)
    os.makedirs(cache, exist_ok=True)
    mpath = f"{cache}/{tag}{name}_means.pt"
    cpath = f"{cache}/{tag}{name}_clean.npz"
    t0 = time.time()
    if os.path.exists(mpath) and os.path.exists(cpath) and int(np.load(cpath)["fp_ok"]) == int(fp * 10000):
        means = torch.load(mpath)
        print(f"  средние вкладов и clean взяты из кэша {cache}/")
    else:
        acc = ab.new_acc()
        F, F0 = [], []
        for s in range(0, n_img, BATCH):
            rb = rows[s:s + BATCH]
            x = norm_batch(imgs[rb]).to(device)
            l00, h = forward(model, x, acc=acc)
            acc["n"] += len(rb)
            F.append(ab.group_feats(h, rb, ginfo))
            F0.append(ab.group_feats(l00, rb, ginfo))
        means = ab.to_means(acc)
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
            F.append(ab.group_feats(h, rb, ginfo))
        Ff = flip_feats(model, imgs, flip_rows, device, a_off, m_off, means)
        np.savez(p, F=np.concatenate(F), flip=Ff, fp_ok=int(fp * 10000))
        feats[cname] = dict(np.load(p))
        print(f"  {cname:11s} {time.time() - t1:5.0f} с", flush=True)
    return feats, means


# ------------------------------------------------------------------
# 2. main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="только clean и 2 основных условия (без справки и 12 уточняющих)")
    ap.add_argument("--limit", type=int, default=0, help="только для проверки: первые N сцен")
    ap.add_argument("--main-seed", type=int, default=None,
                    help="только для проверки в облаке: случайные веса seed N вместо обученной CLIP")
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--cache", default=os.path.expanduser("~/binding-vit-data/ablate_clip"))
    args = ap.parse_args()
    stem = "ablate_clip" + (f"_test{args.limit}" if args.limit else "")
    tag = f"test{args.limit}_" if args.limit else ""
    if args.main_seed is not None:
        tag += f"mainseed{args.main_seed}_"
    pv.MODELS = ["logreg"]
    t_start = time.time()

    meta_all = [json.loads(l) for l in open(f"{DATA}/meta.jsonl")]
    colors = sorted({o[k] for r in meta_all for o in r["objects"] for k in ("color", "color_Sp")})
    folds_all = np.random.default_rng(0).permutation(len(meta_all)) % pv.N_FOLDS
    meta = meta_all[:args.limit] if args.limit else meta_all
    if args.limit:
        print(f"⚠️ ПРОВЕРОЧНЫЙ прогон: только {len(meta)} сцен (--limit). Для настоящих чисел запускай без --limit.")
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
    print(f"Основные условия: блоки {BLOCKS[0]}–{BLOCKS[-1]} (после пика «левого» на L06); справка: блоки 4–12, как у DINO")

    imgs = np.zeros((2 * n_sc, 224, 224, 3), np.uint8)
    for i, rec in enumerate(meta):
        for k, v in enumerate(("S", "Sp")):
            imgs[2 * i + k] = np.asarray(Image.open(f"{DATA}/images/{rec['id']}_{v}.png").convert("RGB"))
    rows = np.arange(2 * n_sc)
    flip_rows = np.array([2 * i for i, r in enumerate(meta) if all(o["shape"] in ab.SYM_SHAPES for o in r["objects"])])
    print(f"Для зеркальной проверки (только круг/квадрат/крест, картинка S): {len(flip_rows)} сцен")

    device = pick_device()
    full = not args.quick
    nets = [("CLIP", args.main_seed)] + [(f"rand{s}", s) for s in range(args.n_random)]
    all_res, all_flip = {}, {}
    for name, seed in nets:
        model, weights, _ = load_model("clip", seed)
        fp = fingerprint(model)
        if name == "CLIP" and seed is None:
            print(f"\n### {name}: {weights}   отпечаток весов {fp:.4f}   устройство: {device}")
        elif name.startswith("rand") and seed in EXPECTED_FP:
            ok = abs(fp - EXPECTED_FP[seed]) < 0.01
            print(f"\n### {name}: {weights}   отпечаток весов {fp:.4f}  "
                  f"{'✅ совпадает с extract.py' if ok else '⚠️ НЕ совпадает — проверь весь вывод'}")
        else:
            print(f"\n### {name}: {weights}   отпечаток весов {fp:.4f}  (проверочный прогон: случайные веса в роли CLIP)")
        model.eval().to(device)
        if name == "CLIP":
            run_checks(model, imgs, device)
        conds = conditions(full) if name == "CLIP" else conditions(False)
        feats, _ = run_net(name, model, conds, imgs, rows, ginfo, flip_rows, device, args.cache, tag)
        t1 = time.time()
        res = {}
        for cname, _, _ in conds:
            grp = GROUPS if cname in ("clean",) + MAIN + REF_DINO else GROUPS[:2]
            res[cname] = ab.probe_all(feats[cname]["F"], labels, folds, scene_idx, grp)
        res["L00"] = ab.probe_all(feats["clean"]["F0"], labels, folds, scene_idx, GROUPS[:2])
        all_res[name] = res
        all_flip[name] = {c: ab.flip_ratio(feats[c]["flip"], np.random.default_rng(0)) for c, _, _ in conds}
        print(f"  пробы: {time.time() - t1:.0f} с", flush=True)
        if name == "CLIP" and args.main_seed is None and not args.limit:
            check_old(feats["clean"], res["clean"], all_flip[name]["clean"], meta)
        del model

    report(all_res, all_flip, nets, full, stem)
    print(f"\nВсего: {time.time() - t_start:.0f} с")


# ------------------------------------------------------------------
# 3. Встроенные проверки
# ------------------------------------------------------------------
@torch.no_grad()
def run_checks(model, imgs, device):
    x = norm_batch(imgs[:4]).to(device)
    # 1. наш прогон = блоки модели (extract.run_layers) и model(x) (head(norm(L12))[CLS])
    _, h = forward(model, x)
    ref = run_layers(model, x)[-1]
    d1 = float((h - ref).abs().max())
    d1b = float((model.forward_head(model.norm(h)) - model(x)).abs().max())
    print(f"Проверка 1 (наш прогон = прогон блоков модели / model(x)): макс. разница {d1:.2e} / {d1b:.2e}  "
          f"{'✅' if max(d1, d1b) < 1e-3 else '⚠️ проверь весь вывод'}")
    # 2. выключение собственным значением: средние по одной картинке -> выключить всё во всех блоках
    x1 = x[:1]
    acc = ab.new_acc()
    _, h_clean = forward(model, x1, acc=acc)
    acc["n"] = 1
    m1 = ab.to_means(acc)
    _, h_self = forward(model, x1, set(range(1, 13)), set(range(1, 13)), m1)
    d2 = float((h_self - h_clean).abs().max())
    print(f"Проверка 2 (выключение «собственным значением» во всех 24 частях = обычный прогон): макс. разница {d2:.2e}  "
          f"{'✅' if d2 < 1e-3 else '⚠️ проверь весь вывод'}")
    # 3. выключить всё во всех блоках средними по 32 картинкам -> CLS одинаков у всех картинок
    xs = norm_batch(imgs[:32]).to(device)
    acc = ab.new_acc()
    for s in range(0, 32, BATCH):
        forward(model, xs[s:s + BATCH], acc=acc)
    acc["n"] = 32
    mm = ab.to_means(acc)
    _, hh = forward(model, xs, set(range(1, 13)), set(range(1, 13)), mm)
    d3 = float((hh[:, 0] - hh[:1, 0]).abs().max())
    print(f"Проверка 3 (всё выключено во всех блоках -> CLS на L12 одинаков у всех картинок): макс. разница {d3:.2e}  "
          f"{'✅' if d3 < 1e-3 else '⚠️ проверь весь вывод'}")


def check_old(fc, rc, flip_c, meta, acts="acts_v2_clip"):
    """Проверка 4: clean = прежние результаты CLIP (только на Маке, обученная CLIP, все 2000 сцен)."""
    print("\nПроверка 4 (обычная сеть = прежние результаты CLIP):")
    for (q, g), ref in REF_PROBE.items():
        acc = float(rc[(q, g)][0].mean())
        print(f"  проба «{QWORD[q]}», {GNAME[g]:11s} L12: {acc * 100:7.3f}%  (probe_v2_clip: {ref * 100:7.3f}%)  "
              f"{'✅ точно' if abs(acc - ref) < 5e-6 else ('✅ (разница < 0,5 п.п.)' if abs(acc - ref) < 0.005 else '⚠️ проверь весь вывод')}")
    for g, ref in REF_FLIP.items():
        r = flip_c[g][0]
        print(f"  зеркальное отношение {('CLS' if g == 'cls' else 'среднее по патчам'):18s} L12: {r:.5f}  (flip_test_clip: {ref:.5f})  "
              f"{'✅' if abs(r - ref) < 0.002 else '⚠️ проверь весь вывод'}")
    if os.path.exists(f"{acts}/index.csv"):
        idx = {(r["id"], r["version"]): int(r["row"]) for r in csv.DictReader(open(f"{acts}/index.csv"))}
        A = np.load(f"{acts}/L12.npy", mmap_mode="r")
        ids = [meta[0]["id"], meta[1]["id"]]
        print(f"  (сохранённые активации {acts}/L12.npy сверяются по CLS сцен {ids[0]}, {ids[1]})")
        d = 0.0
        for i, sid in enumerate(ids):
            for k, v in enumerate(("S", "Sp")):
                tok = np.asarray(A[idx[(sid, v)]], dtype=np.float32)
                d = max(d, float(np.abs(tok[0] - fc["F"][2 * i + k, 1]).max()))
        print(f"  CLS L12 = {acts} (макс. разница по 4 картинкам): {d:.2e}  {'✅' if d < 1e-2 else '⚠️ проверь весь вывод'}")
    else:
        print(f"  ⚠️ не нашёл {acts}/index.csv — сверка с сохранёнными активациями пропущена")


# ------------------------------------------------------------------
# 4. Итог, правило, CSV, график
# ------------------------------------------------------------------
def deltas(net, c, g, alpha):
    Lc, Uc = net["clean"][("left", g)][0], net["clean"][("upper", g)][0]
    L, U = net[c][("left", g)][0], net[c][("upper", g)][0]
    dl, du = L - Lc, U - Uc
    return ab.boot_paired([dl, du, dl - du], alpha)


def fmt(t):
    return f"{t[0] * 100:+6.1f} [{t[1] * 100:+.1f} … {t[2] * 100:+.1f}]"


def report(all_res, all_flip, nets, full, stem):
    D = all_res["CLIP"]
    conds = [c for c, _, _ in conditions(full)]
    rows_csv = []
    warn = sum(nw for net in all_res.values() for cond in net.values() for (_, nw) in cond.values())

    # проверка 5
    l00 = [float(all_res[n]["L00"][(q, g)][0].mean()) for n, _ in nets for q in QUERIES for g in ("far_bg", "cls")]
    print(f"\nПроверка 5а (L00: дальний фон и CLS ровно 50,0% у всех сетей): "
          f"{'✅' if all(abs(a - 0.5) < 1e-12 for a in l00) else '⚠️ проверь весь вывод'}  {sorted(set(round(a * 100, 3) for a in l00))}")
    tw = [float(all_res[n][c][(q, 'twin')][0].mean()) for n, _ in nets for c in ("clean",) + MAIN for q in QUERIES]
    print(f"Проверка 5б (контроль «близнец-вопрос» > 90% в обычной сети и обоих основных условиях, все сети): "
          f"{'✅' if all(a > 0.9 for a in tw) else '⚠️ проверь весь вывод'}  (мин {min(tw) * 100:.1f}%)")
    rnd = [float(all_res[n][c][(q, g)][0].mean()) for n, _ in nets[1:] for c in ("clean",) + MAIN
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
        fr = all_flip["CLIP"][c]
        print(f"{c:12s}  {fb[0]:7.1f} / {fb[1]:5.1f}     {cl[0]:7.1f} / {cl[1]:5.1f}     {fr['cls'][0]:.2f} / {fr['mean'][0]:.2f}")
    for n, _ in nets[1:]:
        s = "   ".join(f"{c}: {all_res[n][c][('left', 'far_bg')][0].mean() * 100:.1f}/{all_res[n][c][('upper', 'far_bg')][0].mean() * 100:.1f}"
                       for c in ("clean",) + MAIN)
        print(f"  {n} (дальний фон, левый/верхний): {s}")

    # Δ и правило
    verdicts = {}
    print("\n" + "=" * 100)
    print(f"ОСНОВНЫЕ УСЛОВИЯ — дальний фон, L12, интервал {100 * (1 - ALPHA_MAIN):.1f}% (поправка на 2 сравнения), п.п.")
    for c in MAIN:
        dl, du, dd = deltas(D, c, "far_bg", ALPHA_MAIN)
        v = ab.verdict(dl, dd)
        accU = float(D[c][("upper", "far_bg")][0].mean())
        broken = accU < BROKEN
        verdicts[c] = (v, dl, du, dd, broken)
        print(f"  {c:11s} Δлевый {fmt(dl)}   Δверхний {fmt(du)}   Δлевый − Δверхний {fmt(dd)}   → «{v.upper()}»"
              + ("   ⚠️ пометка «РАЗРУШЕНИЕ» (верхний < 60%)" if broken else ""))
        rows_csv.append(["CLIP", c, "far_bg", "main", *[f"{x:.5f}" for t in (dl, du, dd) for x in t], v, int(broken)])
    va, vm = verdicts[MAIN[0]][0], verdicts[MAIN[1]][0]
    ov = ab.overall(va, vm)
    print(f"\nВЫВОД ПО ПРАВИЛУ (реестр 53, записано до кода): «{ov}» — {ab.OVERALL_TEXT[ov]}")
    print(f"  внимание 7–12: «{va}»;  MLP 7–12: «{vm}»")
    print("  Оговорки: одна модель (CLIP ViT-B/16); выключение средним — вмешательство вне обучения;"
          " опыт показывает, нужна ли часть для потери «левого», а не как она стирает.")

    # ставка
    dla = round(verdicts[MAIN[0]][1][0] * 100, 1)
    dlm = round(verdicts[MAIN[1]][1][0] * 100, 1)
    hit_a = BET["attn"][0] <= dla <= BET["attn"][1]
    hit_m = BET["mlp"][0] <= dlm <= BET["mlp"][1]
    print(f"\nСТАВКА ЕРНАЗАРА (реестр 53, 9.11 09:45, до кода):")
    print(f"  53a вывод «{BET['verdict']}» -> вышло «{ov}»   {'ПОПАЛ ✅' if ov == BET['verdict'] else 'НЕ ПОПАЛ ❌'}")
    print(f"  53b Δлевый, внимание 7–12 выключено: от +{BET['attn'][0]} до +{BET['attn'][1]} -> вышло {dla:+.1f}   {'ПОПАЛ ✅' if hit_a else 'НЕ ПОПАЛ ❌'}")
    print(f"  53c Δлевый, MLP 7–12 выключен: от +{BET['mlp'][0]} до +{BET['mlp'][1]} -> вышло {dlm:+.1f}   {'ПОПАЛ ✅' if hit_m else 'НЕ ПОПАЛ ❌'}")
    print(f"  (уверенность в 53a: {BET['conf']})")

    # дополнительно: CLS, без правила
    print(f"\nДополнительно (НЕ правило): CLS, L12, интервал {100 * (1 - ALPHA_MAIN):.1f}%")
    for c in MAIN:
        dl, du, dd = deltas(D, c, "cls", ALPHA_MAIN)
        print(f"  {c:11s} Δлевый {fmt(dl)}   Δверхний {du[0] * 100:+6.1f}   Δлевый − Δверхний {fmt(dd)}   (по правилу было бы «{ab.verdict(dl, dd)}»)")
        rows_csv.append(["CLIP", c, "cls", "extra", *[f"{x:.5f}" for t in (dl, du, dd) for x in t], ab.verdict(dl, dd), ""])

    if full:
        # справка: блоки 4–12, как у DINO
        print(f"\nСПРАВКА (НЕ правило): блоки 4–12, как у DINO — дальний фон, L12, интервал {100 * (1 - ALPHA_MAIN):.1f}%")
        for c in REF_DINO:
            dl, du, dd = deltas(D, c, "far_bg", ALPHA_MAIN)
            accU = float(D[c][("upper", "far_bg")][0].mean())
            print(f"  {c:11s} Δлевый {fmt(dl)}   Δверхний {fmt(du)}   Δл − Δв {fmt(dd)}   (по правилу было бы «{ab.verdict(dl, dd)}»)"
                  + ("   ⚠️ разрушение" if accU < BROKEN else ""))
            rows_csv.append(["CLIP", c, "far_bg", "ref_dino_blocks", *[f"{x:.5f}" for t in (dl, du, dd) for x in t],
                             ab.verdict(dl, dd), int(accU < BROKEN)])
        # уточнение
        print(f"\nУТОЧНЕНИЕ — по одному подслою, дальний фон, L12, интервал {100 * (1 - ALPHA_SUB):.2f}% (поправка на 12), п.п.")
        for c in conds[5:]:
            dl, du, dd = deltas(D, c, "far_bg", ALPHA_SUB)
            v = ab.verdict(dl, dd)
            accU = float(D[c][("upper", "far_bg")][0].mean())
            print(f"  {c:8s} Δлевый {fmt(dl)}   Δверхний {du[0] * 100:+6.1f}   Δл − Δв {fmt(dd)}   «{v}»"
                  + ("   ⚠️ разрушение" if accU < BROKEN else ""))
            rows_csv.append(["CLIP", c, "far_bg", "sub", *[f"{x:.5f}" for t in (dl, du, dd) for x in t], v, int(accU < BROKEN)])

    # случайные сети: Δ для справки
    for n, _ in nets[1:]:
        for c in MAIN:
            dl, du, dd = deltas(all_res[n], c, "far_bg", ALPHA_MAIN)
            rows_csv.append([n, c, "far_bg", "random", *[f"{x:.5f}" for t in (dl, du, dd) for x in t], ab.verdict(dl, dd), ""])

    # зеркало: справка
    print("\nДополнительно (НЕ правило): зеркальное отношение на L12 [95% инт.] (1 = не слепа к зеркалу ←→)")
    for c in ("clean",) + MAIN + (REF_DINO if full else ()):
        fr = all_flip["CLIP"][c]
        rr = [all_flip[n][c]["cls"][0] for n, _ in nets[1:] if c in all_flip[n]]
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
        ax.axvline(3.5, c="gray", ls=":", lw=1)
    ax.set_xticks(x, cs, rotation=60, fontsize=8)
    ax.set_ylabel("изменение точности, п.п.")
    ax.set_title("CLIP, дальний фон, L12: что меняется, если выключить часть блоков (средним)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=150)


if __name__ == "__main__":
    main()

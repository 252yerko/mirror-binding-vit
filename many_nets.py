"""
many_nets.py — опыт B: 24 готовые ViT. Добавлено 13.11 (реестр 139–140, ставка Ерназара — до кода).

ВОПРОС ОПЫТА:
  На 6 сетях ранги «слепота к зеркалу» и «разрыв верхний − левый» совпали (ρ = −0,943, хрупко), а на 4 сетях
  (опыт A + D, OSF npvs5) разведочно: чем слепее сеть, тем больше разрыва объясняет маскировка (R меньше).
  Здесь обе связи проверяются заранее записанными правилами на 24 сетях; заодно — влияние регистров у DINOv2.

СЕТИ (timm, вход 224×224, все — одним кодом; семейство — для правила B3):
  DINO: S/16, B/16 | DINOv2 без регистров: S/14, B/14, L/14 | DINOv2 с регистрами: S/14, B/14, L/14 |
  MAE: B/16, L/16 | CLIP OpenAI: B/16, L/14 | CLIP LAION-2B: B/16, L/14 | MetaCLIP: B/16 | DFN: B/16 |
  SigLIP: B/16 | DeiT III: S/16, B/16, L/16 | DeiT: B/16 | AugReg (ImageNet-21k): S/16, B/16, L/16.
  Случайные сети (seed 0, задан до создания модели): по одной на 7 архитектур (см. RANDOM) — для проверки 10.
  Нормировка входа — своя у каждой сети (pretrained_cfg timm). DINOv2 — img_size=224 (позиционные коды
  интерполирует timm; при обучении DINOv2 в конце дообучали на 518 — ограничение).

МЕРЫ (у каждой сети — последний слой: L12 у S/B, L24 у L):
  отношение = ср. d(←→) / ср. d(↑↓), косинусное расстояние среднего по патчам (без служебных токенов),
      631 сцена только из круга / квадрата / креста, картинка S (как flip_test.py / more_nets.py);
  разрыв = «верхний» − «левый», обычная проба (StandardScaler + логрег C = 0,1, 5 фолдов по сценам как везде),
      дальний фон, все 2000 сцен (S и Sp);
  R = как в правиле A2 опыта «A + D» (mirror_decomp.py): разрыв в несимметричной части / обычный разрыв,
      627 сцен правил; применимо, если нижняя граница обычного разрыва на этих сценах > 0.
  Дальний фон — пустые патчи, у которых зазор до любого патча близнеца ≥ 32 px: 3 патча при патче 16,
      4 патча при патче 14 (зазор 42 px). [Замечание Ерназара 22:58, принято до кода.]
  95% интервалы — бутстрэп по сценам (1000, seed 0).

ПРАВИЛА (записаны ДО запуска, реестр 139–140; 24 обученные сети; точечные значения):
  B1: ρ Спирмена(отношение, разрыв). «Связь следует», если ρ < 0 и односторонний перестановочный p < 0,05
      (10 000 перестановок, seed 0); иначе «не подтверждена».
  B2: ρ Спирмена(отношение, R) по сетям, где R применимо. «Слепее → больше маскировки», если ρ > 0 и
      односторонний перестановочный p < 0,05; иначе «не подтверждена».
  B3: B1 повторяется 12 раз, каждый раз без одного семейства. Знак ρ < 0 и p < 0,05 во всех 12 -> «устойчиво»;
      ρ < 0 во всех, но p ≥ 0,05 хотя бы раз -> «знак держится, значимость зависит от семейства»;
      ρ ≥ 0 хотя бы раз -> «неустойчиво».
  Справка (не правила): регистры — разность отношения (с регистрами − без) у DINOv2 S, B, L; средние по семействам;
      размер сети; CLS (где есть).
  Ставки Ерназара (реестр 140, 13.11 22:58): 140a B1 «связь следует»; 140b ρ в B1 от −0,70 до −0,30 (округление 0,01);
      140c B2 «не подтверждена»; 140d B3 «знак держится, значимость зависит от семейства»; 140e средняя разность
      регистров 0,05–0,15 (округление 0,01); 140f разность в S меньше, чем в B и в L; 140g самое слепое семейство
      (наименьшее среднее отношение) — DINO; 140h наименее слепое — MAE или языковое (CLIP OpenAI, CLIP LAION,
      MetaCLIP, DFN, SigLIP).

ВСТРОЕННЫЕ ПРОВЕРКИ (все должны пройти):
  1. L00: обычная и несимм. пробы (CLS, где есть, и дальний фон) — ровно 50,0%.
  2. Пиксели дальнего фона в S и Sp совпадают; группы отражений = группы по отражённой маске.
  3. Наш пошаговый прогон = forward_features самой модели (макс. разница < 1e-3).
  4. Пустая серая картинка не меняется при отражениях ни на L00, ни на последнем слое (< 1e-4).
  5. Симм. проба: на каждой паре (x, mx) баллы в сумме ровно 1.
  6. Контроль «близнец-вопрос» > 90% на последнем слое у всех сетей.
  7. Случайная сеть отличается от обученной (отпечаток весов).
  8. Сетка (14 или 16), служебные токены, число блоков — как ожидается; нормировка печатается.
  9. Прежние числа воспроизводятся: DINO B, CLIP B, DeiT III B, DINOv2 B с регистрами — отношение ±0,01,
     точности «левый» / «верхний» ±0,5 п.п. (у DINOv2 — справочная группа «≥ 4 патчей» из more_nets_dinov2.csv).
 10. Случайные сети: |разрыв| < 2 п.п.

ЗАПУСК (из папки binding-vit; три части, по одной команде; потом сводка):
  caffeinate -i python many_nets.py --part 1 2>&1 | tee many_nets_part1_log.txt
  caffeinate -i python many_nets.py --part 2 2>&1 | tee many_nets_part2_log.txt
  caffeinate -i python many_nets.py --part 3 2>&1 | tee many_nets_part3_log.txt
  python many_nets.py --summary 2>&1 | tee many_nets_summary_log.txt
  -> many_nets.csv (дописывается по сетям); кэш — ~/binding-vit-data/many_nets/.
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr

import flip_test as ft
import probe_big as pb
import mirror_decomp as md
from extract import run_layers, fingerprint, pick_device

DATA = "data_v2"
BATCH = 16
OUT_CSV = "many_nets.csv"
# имя, timm, семейство, размер, патч, часть
NETS = [
    ("DINO S/16", "vit_small_patch16_224.dino", "DINO", "S", 16, 1),
    ("DINO B/16", "vit_base_patch16_224.dino", "DINO", "B", 16, 1),
    ("DINOv2 S/14", "vit_small_patch14_dinov2.lvd142m", "DINOv2", "S", 14, 1),
    ("DINOv2 B/14", "vit_base_patch14_dinov2.lvd142m", "DINOv2", "B", 14, 1),
    ("DINOv2 L/14", "vit_large_patch14_dinov2.lvd142m", "DINOv2", "L", 14, 1),
    ("DINOv2-reg S/14", "vit_small_patch14_reg4_dinov2.lvd142m", "DINOv2-reg", "S", 14, 1),
    ("DINOv2-reg B/14", "vit_base_patch14_reg4_dinov2.lvd142m", "DINOv2-reg", "B", 14, 1),
    ("DINOv2-reg L/14", "vit_large_patch14_reg4_dinov2.lvd142m", "DINOv2-reg", "L", 14, 1),
    ("MAE B/16", "vit_base_patch16_224.mae", "MAE", "B", 16, 2),
    ("MAE L/16", "vit_large_patch16_224.mae", "MAE", "L", 16, 2),
    ("CLIP-OpenAI B/16", "vit_base_patch16_clip_quickgelu_224.openai", "CLIP-OpenAI", "B", 16, 2),
    ("CLIP-OpenAI L/14", "vit_large_patch14_clip_quickgelu_224.openai", "CLIP-OpenAI", "L", 14, 2),
    ("CLIP-LAION B/16", "vit_base_patch16_clip_224.laion2b", "CLIP-LAION", "B", 16, 2),
    ("CLIP-LAION L/14", "vit_large_patch14_clip_224.laion2b", "CLIP-LAION", "L", 14, 2),
    ("MetaCLIP B/16", "vit_base_patch16_clip_quickgelu_224.metaclip_400m", "MetaCLIP", "B", 16, 2),
    ("DFN B/16", "vit_base_patch16_clip_224.dfn2b", "DFN", "B", 16, 2),
    ("SigLIP B/16", "vit_base_patch16_siglip_224.webli", "SigLIP", "B", 16, 3),
    ("DeiT III S/16", "deit3_small_patch16_224.fb_in1k", "DeiT III", "S", 16, 3),
    ("DeiT III B/16", "deit3_base_patch16_224.fb_in1k", "DeiT III", "B", 16, 3),
    ("DeiT III L/16", "deit3_large_patch16_224.fb_in1k", "DeiT III", "L", 16, 3),
    ("DeiT B/16", "deit_base_patch16_224.fb_in1k", "DeiT", "B", 16, 3),
    ("AugReg S/16", "vit_small_patch16_224.augreg_in21k", "AugReg", "S", 16, 3),
    ("AugReg B/16", "vit_base_patch16_224.augreg_in21k", "AugReg", "B", 16, 3),
    ("AugReg L/16", "vit_large_patch16_224.augreg_in21k", "AugReg", "L", 16, 3),
]
RANDOM = {"DINO S/16": 1, "DINO B/16": 1, "DINOv2-reg B/14": 1, "CLIP-OpenAI B/16": 2, "SigLIP B/16": 3,
          "DeiT III B/16": 3, "AugReg L/16": 3}
LANG = {"CLIP-OpenAI", "CLIP-LAION", "MetaCLIP", "DFN", "SigLIP"}
# проверка 9: (файл/источник, отношение, «левый» %, «верхний» %)
OLD = {"DINO B/16": (0.284, 64.9, 83.5), "CLIP-OpenAI B/16": (0.613, 63.15, 72.15),
       "DeiT III B/16": (0.378, 57.825, 75.0), "DINOv2-reg B/14": (0.511, 63.2, 74.075)}


def load(timm_name, seed):
    import timm
    kw = dict(img_size=224) if "dinov2" in timm_name else {}
    if seed is None:
        m = timm.create_model(timm_name, pretrained=True, **kw)
        w = f"обученная {timm_name}"
    else:
        torch.manual_seed(seed)
        m = timm.create_model(timm_name, pretrained=False, **kw)
        w = f"СЛУЧАЙНАЯ {timm_name.split('.')[0]} (нулевая модель), seed {seed}"
    m.eval()
    return m, w


def setup(model, P):
    G = 224 // P
    grid = tuple(model.patch_embed.grid_size)
    n_pre = model.num_prefix_tokens
    nb = len(model.blocks)
    exp_pre = 5 if getattr(model, "num_reg_tokens", 0) == 4 else (0 if not getattr(model, "has_class_token", True) else 1)
    cfg = model.pretrained_cfg
    ok = grid == (G, G) and n_pre == exp_pre and nb in (12, 24) and tuple(cfg["input_size"][1:]) in ((224, 224), (518, 518))
    print(f"  Проверка 8: сетка {grid}; служебных токенов {n_pre}; блоков {nb}; нормировка "
          f"{tuple(round(x, 3) for x in cfg['mean'])}/{tuple(round(x, 3) for x in cfg['std'])}  {'✅' if ok else '⚠️ проверь весь вывод'}")
    mean = torch.tensor(cfg["mean"]).view(3, 1, 1)
    std = torch.tensor(cfg["std"]).view(3, 1, 1)
    return ok, G, n_pre, nb, mean, std


def img(path, mean, std):
    im = Image.open(path).convert("RGB")
    assert im.size == (224, 224)
    x = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0
    return (x - mean) / std


@torch.no_grad()
def extract(model, recs, infos, mirrors, device, n_pre, nb, mean, std, with_pm):
    """F[2 слоя (L00, последний), сцены, 2 версии, len(mirrors), 2 группы (cls, far_bg), D];
    PM[сцены, len(mirrors), D] — среднее по патчам (последний слой, только версия S); T[сцены, 2, 2, D] — близнецы."""
    D = model.embed_dim
    F = np.zeros((2, len(recs), 2, len(mirrors), 2, D), dtype=np.float32)
    PM = np.zeros((len(recs), len(mirrors), D), dtype=np.float32)
    T = np.zeros((len(recs), 2, 2, D), dtype=np.float32)
    jobs = [(si, vi, mi) for si in range(len(recs)) for vi in range(2) for mi in range(len(mirrors))]
    t0 = time.time()
    for s in range(0, len(jobs), BATCH):
        part = jobs[s:s + BATCH]
        xs = []
        for si, vi, mi in part:
            x = img(f"{DATA}/images/{recs[si]['id']}_{('S', 'Sp')[vi]}.png", mean, std)
            k = mirrors[mi]
            xs.append(x if k == "o" else (x.flip(2) if k == "h" else x.flip(1)))
        outs = run_layers(model, torch.stack(xs).to(device))
        hs = [outs[0].float().cpu().numpy(), outs[nb].float().cpu().numpy()]
        for b, (si, vi, mi) in enumerate(part):
            g = infos[si][0][mirrors[mi]]
            idx = [int(q) + n_pre for q in g["far_bg"]]
            for li, h in enumerate(hs):
                if n_pre > 0:
                    F[li, si, vi, mi, 0] = h[b, 0]
                F[li, si, vi, mi, 1] = h[b, idx].mean(0)
            if with_pm and vi == 0:
                PM[si, mi] = hs[1][b, n_pre:].mean(0)
            if mirrors[mi] == "o":
                for t in range(2):
                    T[si, vi, t] = hs[1][b, [int(q) + n_pre for q in g[f"twin{t}"]]].mean(0)
        if (s // BATCH) % 100 == 0:
            print(f"    картинок {min(s + BATCH, len(jobs))}/{len(jobs)}  ({time.time() - t0:.0f} с)", flush=True)
    return F, PM, T


def run_net(name, timm_name, family, size, P, seed, meta, args, device, colors, folds_all, ok):
    t0 = time.time()
    model, weights = load(timm_name, seed)
    tag = name if seed is None else f"{name} [случайная]"
    fp = fingerprint(model)
    print(f"\n### {tag}: {weights}   отпечаток весов {fp:.4f}", flush=True)
    ok8, G, n_pre, nb, mean, std = setup(model, P)
    ok["8"] &= ok8
    pb.FAR = 3 if P == 16 else 4
    key = (P, args.limit)
    if key not in INFO:
        INFO[key] = [md.scene_info(r, colors, P, G) for r in meta]
    infos = INFO[key]
    model.to(device)
    x2 = torch.stack([img(f"{DATA}/images/{meta[0]['id']}_{v}.png", mean, std) for v in ("S", "Sp")]).to(device)
    with torch.no_grad():
        ref = model.forward_features(x2)
        mine = model.norm(run_layers(model, x2)[nb])
    d3 = float((mine - ref).abs().max())
    ok["3"] &= d3 < 1e-3
    blank = x2[:1].clone()
    blank[:] = blank[0, :, :1, :1]
    with torch.no_grad():
        fb = [run_layers(model, b) for b in (blank, blank.flip(3), blank.flip(2))]
    z = max(float((fb[0][l] - fb[k][l]).abs().max()) for l in (0, nb) for k in (1, 2))
    ok["4"] &= z < 1e-4
    print(f"  Проверка 3: макс. разница {d3:.2e} {'✅' if d3 < 1e-3 else '⚠️'};  Проверка 4: {z:.2e} {'✅' if z < 1e-4 else '⚠️'}")

    sym_i = [i for i, r in enumerate(meta) if all(o["shape"] in ft.SYM_SHAPES for o in r["objects"])]
    def distinct(r):
        a, b = r["objects"][0]["center"], r["objects"][1]["center"]
        return a[0] != b[0] and a[1] != b[1]
    rule_in_sym = np.array([distinct(meta[i]) for i in sym_i])
    rest_i = [i for i in range(len(meta)) if i not in set(sym_i)]
    safe = name.replace("/", "_").replace(" ", "_")
    cache = os.path.join(args.out, f"{safe}{'' if seed is None else '_rand' + str(seed)}{'_test' + str(args.limit) if args.limit else ''}.npz")
    if os.path.exists(cache):
        zz = np.load(cache)
        FS, PMS, TS, FX, TX = (zz[k].astype(np.float32) for k in ("FS", "PMS", "TS", "FX", "TX"))
        print(f"  взято из {cache}")
    else:
        FS, PMS, TS = extract(model, [meta[i] for i in sym_i], [infos[i] for i in sym_i], ["o", "h", "v"], device, n_pre, nb, mean, std, True)
        FX, _, TX = extract(model, [meta[i] for i in rest_i], [infos[i] for i in rest_i], ["o"], device, n_pre, nb, mean, std, False)
        np.savez(cache, FS=FS.astype(np.float16), PMS=PMS.astype(np.float16), TS=TS.astype(np.float16),
                 FX=FX.astype(np.float16), TX=TX.astype(np.float16))
        FS, PMS, TS, FX, TX = (a.astype(np.float16).astype(np.float32) for a in (FS, PMS, TS, FX, TX))
    del model
    if device.type == "mps":
        torch.mps.empty_cache()

    rng = np.random.default_rng(0)
    # отношение (631 сцена, S, среднее по патчам, последний слой)
    dh = ft.cos_dist(PMS[:, 0], PMS[:, 1])
    dv = ft.cos_dist(PMS[:, 0], PMS[:, 2])
    ratio = ft.boot_ratio(dh, dv, np.random.default_rng(0))
    # метки
    def labels(idx):
        return {q: np.array([[infos[i][1][q][1][v] for v in ("S", "Sp")] for i in idx]) for q in ("left", "upper")}
    lab_all = labels(range(len(meta)))
    lab_rule = labels([i for i, k in zip(sym_i, rule_in_sym) if k])
    # обычная проба, все сцены, дальний фон, последний слой
    X = np.zeros((len(meta), 2, FS.shape[-1]), dtype=np.float32)
    X[sym_i], X[rest_i] = FS[1, :, :, 0, 1], FX[1, :, :, 0, 1]
    acc = {}
    for q in ("left", "upper"):
        acc[q], nw = md.probe_plain(X, lab_all[q][:, :, 0], lab_all[q][:, :, 1], folds_all)
        ok["warn"] += nw
    gap = md.boot(acc["upper"] - acc["left"], np.random.default_rng(0))
    # контроль
    TW = np.zeros((len(meta), 2, 2, FS.shape[-1]), dtype=np.float32)
    TW[sym_i], TW[rest_i] = TS, TX
    for q in ("left", "upper"):
        tq = np.array([infos[i][1][q][0] for i in range(len(meta))])
        c_, _ = md.probe_plain(TW[np.arange(len(meta)), :, tq], lab_all[q][:, :, 0], lab_all[q][:, :, 1], folds_all)
        ok["6"] &= np.nanmean(c_) > 0.9
    # R (627 сцен правил), последний слой, дальний фон; L00 — проверки 1 и 5
    folds_r = folds_all[np.array(sym_i)[rule_in_sym]]
    groups = [1] + ([0] if n_pre > 0 else [])
    Rres = {}
    for li in (0, 1):
        for gi in groups:
            r_, nw = md.probes_A(FS[li][rule_in_sym, :, :, gi], lab_rule, folds_r)
            ok["warn"] += nw
            ok["5"] &= all(r_[(q, "pair_ok")] for q in ("left", "upper"))
            if li == 0:
                ok["1"] &= all(np.allclose(r_[(q, k)], 0.5) for q in ("left", "upper") for k in ("normal", "anti"))
            Rres[(li, gi)] = r_
    r12 = Rres[(1, 1)]
    gfull = r12[("upper", "normal")] - r12[("left", "normal")]
    ganti = r12[("upper", "anti")] - r12[("left", "anti")]
    gf = md.boot(gfull, np.random.default_rng(0))
    if gf[1] > 0:
        R = md.boot_ratio(ganti, gfull, np.random.default_rng(0))
    else:
        R = (float("nan"),) * 3 + (int(rule_in_sym.sum()),)
    cls_gap = None
    if n_pre > 0:
        Xc = np.zeros_like(X)
        Xc[sym_i], Xc[rest_i] = FS[1, :, :, 0, 0], FX[1, :, :, 0, 0]
        ac = {q: md.probe_plain(Xc, lab_all[q][:, :, 0], lab_all[q][:, :, 1], folds_all)[0] for q in ("left", "upper")}
        cls_gap = md.boot(ac["upper"] - ac["left"], np.random.default_rng(0))
    res = dict(name=name, family=family, size=size, P=P, seed="" if seed is None else seed, weights=weights, fp=fp,
               ratio=ratio[0], ratio_lo=ratio[1], ratio_hi=ratio[2],
               left=float(np.nanmean(acc["left"])), upper=float(np.nanmean(acc["upper"])),
               gap=gap[0], gap_lo=gap[1], gap_hi=gap[2],
               gap_rule=gf[0], gap_rule_lo=gf[1], R=R[0], R_lo=R[1], R_hi=R[2],
               anti_left=float(np.nanmean(r12[("left", "anti")])), anti_upper=float(np.nanmean(r12[("upper", "anti")])),
               cls_gap="" if cls_gap is None else cls_gap[0], blocks=nb, n_pre=n_pre)
    print(f"  отношение {ratio[0]:.3f} [{ratio[1]:.3f}–{ratio[2]:.3f}]; «левый» {res['left'] * 100:.1f}%, «верхний» {res['upper'] * 100:.1f}%; "
          f"разрыв {gap[0] * 100:.1f} [{gap[1] * 100:.1f} … {gap[2] * 100:.1f}] п.п.; R = {R[0]:.2f} [{R[1]:.2f} … {R[2]:.2f}]   ({time.time() - t0:.0f} с)", flush=True)
    return res


INFO = {}
FIELDS = ["name", "family", "size", "P", "seed", "weights", "fp", "ratio", "ratio_lo", "ratio_hi", "left", "upper", "gap", "gap_lo",
          "gap_hi", "gap_rule", "gap_rule_lo", "R", "R_lo", "R_hi", "anti_left", "anti_upper", "cls_gap", "blocks", "n_pre"]


def run_part(args):
    meta = [json.loads(line) for line in open(f"{DATA}/meta.jsonl")]
    if args.limit:
        meta = meta[:args.limit]
    colors = sorted({ob[k] for r in meta for ob in r["objects"] for k in ("color", "color_Sp")})
    folds_all = np.random.default_rng(0).permutation(len(meta)) % md.N_FOLDS
    os.makedirs(args.out, exist_ok=True)
    device = pick_device()
    import timm
    print(f"Опыт B: 24 сети, часть {args.part}   timm {timm.__version__}, torch {torch.__version__}")
    if args.limit:
        print("⚠️ ПРОВЕРОЧНЫЙ прогон (облако): числа не для статьи.")
    ok = {k: True for k in ("1", "3", "4", "5", "6", "8")}
    ok["warn"] = 0
    rows, fps = [], {}
    sel = [n for n in NETS if n[5] == args.part and (not args.only or n[0] in args.only)]
    for name, tn, fam, size, P, _ in sel:
        if not args.random_only:
            r = run_net(name, tn, fam, size, P, 99 if args.test_random_main else None, meta, args, device, colors, folds_all, ok)
            rows.append(r)
            fps[name] = r["fp"]
        if name in RANDOM:
            rr = run_net(name, tn, fam, size, P, 0, meta, args, device, colors, folds_all, ok)
            rows.append(rr)
            if not args.random_only:
                ok.setdefault("7", True)
                ok["7"] &= abs(rr["fp"] - fps[name]) > 1e-3
            ok.setdefault("10", True)
            ok["10"] &= abs(rr["gap"]) < 0.02
    out = OUT_CSV if not args.limit else f"many_nets_test{args.limit}.csv"
    exists = os.path.exists(out)
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nПроверка 1 (L00 ровно 50,0%): {'✅' if ok['1'] else '⚠️ проверь весь вывод'}")
    print("Проверка 2 ✅ (пиксели дальнего фона S = Sp; группы отражений = по отражённой маске — иначе программа остановилась бы)")
    for k, t in (("3", "прогон = forward_features"), ("4", "пустая картинка"), ("5", "симм. проба: пары в сумме 1"),
                 ("6", "контроль > 90%"), ("8", "сетка / токены / блоки")):
        print(f"Проверка {k} ({t}): {'✅' if ok[k] else '⚠️ проверь весь вывод'}")
    if "7" in ok:
        print(f"Проверка 7 (случайные отличаются от обученных): {'✅' if ok['7'] else '⚠️ проверь весь вывод'}")
    if "10" in ok:
        print(f"Проверка 10 (случайные: |разрыв| < 2 п.п.): {'✅' if ok['10'] else '⚠️ проверь весь вывод'}")
    print(f"Предупреждений «не сошлось»: {ok['warn']}  {'✅' if ok['warn'] == 0 else '⚠️ проверь весь вывод'}")
    if not args.limit:
        for r in rows:
            if r["name"] in OLD and r["seed"] == "":
                o = OLD[r["name"]]
                good = abs(r["ratio"] - o[0]) <= 0.01 and abs(r["left"] * 100 - o[1]) <= 0.5 and abs(r["upper"] * 100 - o[2]) <= 0.5
                print(f"Проверка 9 ({r['name']}): отношение {r['ratio']:.3f} / было {o[0]:.3f}; «левый» {r['left'] * 100:.2f} / {o[1]:.2f}; "
                      f"«верхний» {r['upper'] * 100:.2f} / {o[2]:.2f}  {'✅' if good else '⚠️ проверь весь вывод'}")
    print(f"\nЧисла дописаны в {out}")


def perm_p(x, y, sign, n=10000, seed=0):
    rho = spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n):
        r = spearmanr(x, rng.permutation(y)).statistic
        cnt += (r <= rho) if sign < 0 else (r >= rho)
    return float(rho), (cnt + 1) / (n + 1)


def summary(args):
    f = OUT_CSV if not args.limit else f"many_nets_test{args.limit}.csv"
    rows = [r for r in csv.DictReader(open(f, encoding="utf-8")) if r["seed"] == ("" if not args.test_random_main else "99")]
    rows = list({r["name"]: r for r in rows}.values())   # если сеть считали дважды — берётся последняя строка
    names_ok = {n[0] for n in NETS}
    rows = [r for r in rows if r["name"] in names_ok]
    print(f"Сетей в сводке: {len(rows)} из {len(NETS)}")
    for r in sorted(rows, key=lambda r: float(r["ratio"])):
        R = float(r["R"]) if r["R"] not in ("", "nan") else float("nan")
        print(f"  {r['name']:18s} {r['family']:12s} отношение {float(r['ratio']):.3f}  разрыв {float(r['gap']) * 100:5.1f} п.п.  R {R:5.2f}")
    x = np.array([float(r["ratio"]) for r in rows])
    y = np.array([float(r["gap"]) for r in rows])
    rho1, p1 = perm_p(x, y, -1)
    v1 = "связь следует" if rho1 < 0 and p1 < 0.05 else "не подтверждена"
    okR = [i for i, r in enumerate(rows) if r["R"] not in ("", "nan") and not np.isnan(float(r["R"]))]
    if len(okR) >= 4:
        rho2, p2 = perm_p(x[okR], np.array([float(rows[i]["R"]) for i in okR]), +1)
        v2 = "слепее → больше маскировки" if rho2 > 0 and p2 < 0.05 else "не подтверждена"
    else:
        rho2, p2, v2 = float("nan"), float("nan"), "не применимо (R определено меньше чем у 4 сетей)"
    fams = sorted({r["family"] for r in rows})
    loo = []
    for fam in fams:
        keep = [i for i, r in enumerate(rows) if r["family"] != fam]
        rr, pp = perm_p(x[keep], y[keep], -1)
        loo.append((fam, rr, pp, len(keep)))
    if all(rr < 0 and pp < 0.05 for _, rr, pp, _ in loo):
        v3 = "устойчиво"
    elif all(rr < 0 for _, rr, _, _ in loo):
        v3 = "знак держится, значимость зависит от семейства"
    else:
        v3 = "неустойчиво"
    print(f"\nВЫВОД ПО ПРАВИЛАМ (реестр 139–140, записано до запуска):")
    print(f"  B1: ρ(отношение, разрыв) = {rho1:+.3f}, p = {p1:.4f} (n = {len(rows)}) -> «{v1}»")
    print(f"  B2: ρ(отношение, R) = {rho2:+.3f}, p = {p2:.4f} (n = {len(okR)}) -> «{v2}»")
    print(f"  B3: без одного семейства:")
    for fam, rr, pp, k in loo:
        print(f"      без {fam:12s}: ρ = {rr:+.3f}, p = {pp:.4f} (n = {k})")
    print(f"      -> «{v3}»")
    # справка
    by = {r["name"]: float(r["ratio"]) for r in rows}
    diffs = {s: by.get(f"DINOv2-reg {s}/14", np.nan) - by.get(f"DINOv2 {s}/14", np.nan) for s in ("S", "B", "L")}
    mdiff = float(np.nanmean(list(diffs.values())))
    print(f"\nСправка: регистры (с − без), отношение: " + ", ".join(f"{s} {d:+.3f}" for s, d in diffs.items()) + f"; среднее {mdiff:+.3f}")
    fam_mean = {fam: float(np.mean([float(r["ratio"]) for r in rows if r["family"] == fam])) for fam in fams}
    order = sorted(fam_mean, key=fam_mean.get)
    print("Справка: среднее отношение по семействам (слепее -> выше): " + "; ".join(f"{f} {fam_mean[f]:.3f}" for f in order))
    if not args.limit:
        print("\nСТАВКИ ЕРНАЗАРА (реестр 140, 13.11 22:58):")
        hit = lambda b: "ПОПАЛ ✅" if b else "НЕ ПОПАЛ ❌"
        print(f"  140a B1 «связь следует» -> «{v1}»  {hit(v1 == 'связь следует')}")
        print(f"  140b ρ от −0,70 до −0,30 -> {rho1:+.2f}  {hit(-0.70 <= round(rho1, 2) <= -0.30)}")
        print(f"  140c B2 «не подтверждена» -> «{v2}»  {hit(v2 == 'не подтверждена')}")
        print(f"  140d B3 «знак держится, значимость зависит от семейства» -> «{v3}»  {hit(v3 == 'знак держится, значимость зависит от семейства')}")
        print(f"  140e средняя разность регистров 0,05–0,15 -> {mdiff:+.2f}  {hit(0.05 <= round(mdiff, 2) <= 0.15)}")
        print(f"  140f разность в S меньше, чем в B и в L -> S {diffs['S']:+.3f}, B {diffs['B']:+.3f}, L {diffs['L']:+.3f}  "
              f"{hit(diffs['S'] < diffs['B'] and diffs['S'] < diffs['L'])}")
        print(f"  140g самое слепое семейство — DINO -> {order[0]}  {hit(order[0] == 'DINO')}")
        print(f"  140h наименее слепое — MAE или языковое -> {order[-1]}  {hit(order[-1] == 'MAE' or order[-1] in LANG)}")
    with open("many_nets_rules.csv" if not args.limit else f"many_nets_rules_test{args.limit}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rule", "rho", "p", "n", "verdict"])
        w.writerow(["B1", rho1, p1, len(rows), v1])
        w.writerow(["B2", rho2, p2, len(okR), v2])
        for fam, rr, pp, k in loo:
            w.writerow([f"B3_without_{fam}", rr, pp, k, ""])
        w.writerow(["B3", "", "", "", v3])
        w.writerow(["registers_mean_diff", mdiff, "", 3, ""])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, choices=[1, 2, 3])
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--out", default=os.path.expanduser("~/binding-vit-data/many_nets"))
    ap.add_argument("--limit", type=int, default=0, help="только для проверки в облаке")
    ap.add_argument("--test-random-main", action="store_true", help="только для проверки в облаке: вместо обученной — случайная seed 99")
    ap.add_argument("--only", nargs="*", default=None, help="только для проверки: имена сетей")
    ap.add_argument("--random-only", action="store_true", help="только для проверки")
    args = ap.parse_args()
    if args.summary:
        summary(args)
    else:
        assert args.part, "нужно --part 1|2|3 или --summary"
        run_part(args)


if __name__ == "__main__":
    main()

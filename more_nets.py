"""
more_nets.py — ещё две сети: DINOv2 ViT-B/14 с регистрами и supervised ViT-B/16 (DeiT III). Добавлено 13.11.

ВОПРОС ОПЫТА:
  Сохраняются ли на двух новых сетях слепота к зеркалу слева-направо и разрыв «верхний − левый»?
  И держится ли связь «чем слепее сеть, тем больше разрыв», если сетей шесть, а не четыре?

СЕТИ (обе, по их коду, обучались с отражением слева-направо — RandomHorizontalFlip):
  dinov2  timm vit_base_patch14_reg4_dinov2.lvd142m — DINOv2 ViT-B/14 с 4 регистрами (LVD-142M),
          самообучение (DINO + iBOT). Код аугментаций: dinov2/data/augmentations.py — RandomHorizontalFlip(p=0.5)
          на глобальных и локальных кропах. Вход 224×224 -> сетка 16×16 патчей по 14 px; позиционные коды
          timm пересчитывает с 37×37 на 16×16 (интерполяция). Служебных токенов 5: CLS + 4 регистра.
  deit3   timm deit3_base_patch16_224.fb_in1k — DeiT III ViT-B/16, обучение с учителем (классы ImageNet-1k).
          Код аугментаций: deit/augment.py (3-Augment) — RandomHorizontalFlip(). Сетка 14×14 патчей по 16 px.
  Для каждой — 3 случайные сети той же архитектуры (seed 0/1/2, seed ДО создания модели).

ЧТО МЕРЯЕМ (тем же способом, что раньше):
  1. Зеркало (как flip_test.py): 631 сцена из кругов/квадратов/крестов (картинка S);
     отношение d(зеркало ←→) / d(зеркало ↑↓), косинусное расстояние, «среднее по патчам» (без CLS и регистров)
     и CLS, все слои; 95% интервал — бутстрэп по сценам (1000, seed 0).
  2. Привязка (как probe_v2.py / probe_big.py): 2000 сцен, S и Sp; вопрос «какого цвета левый / верхний близнец»;
     средние по группам токенов: CLS, нетронутые, дальний фон, близнец-«вопрос» (контроль); все 13 слоёв;
     StandardScaler + логрег C = 0,1; 5 фолдов по сценам (те же, что в probe_v2); бутстрэп по сценам.
     Разрыв = точность «верхний» − точность «левый», парный бутстрэп по сценам (1000, seed 0).
  «Дальний фон» = пустые патчи на расстоянии ≥ 3 патчей своей сетки от обоих близнецов: у DeiT III это 48 px
  (как у DINO и CLIP), у DINOv2 — 42 px (как у MAE и I-JEPA). Справка (не правило, замечание Ерназара 13.11):
  у DINOv2 дополнительно дальний фон ≥ 4 патчей (56 px ≥ 48 px).

ПРАВИЛА (записаны ДО запуска, реестр 114–115; правило 3 считает more_nets_compare.py):
  Правило 1 (каждая сеть): «слепа к зеркалу», если ВЕРХНЯЯ граница 95% интервала отношения
    (среднее по патчам, L12) < 0,90; иначе «не показано».
  Правило 2 (каждая сеть): «разрыв есть», если НИЖНЯЯ граница 95% интервала разрыва
    (дальний фон, L12, логрег) > 0; иначе «нет».
  Правило 3 (6 сетей: DINO, CLIP, MAE, I-JEPA, DINOv2, DeiT III): ρ Спирмена между отношением (среднее по патчам,
    последний слой) и разрывом (дальний фон, последний слой). «Связь следует», если ρ ≤ −0,83 (не больше двух перестановок
    соседей; без связи — вероятность 12/720 ≈ 1,7%); иначе «не подтверждена».
    4 точки из 6 известны заранее (ρ по ним = −0,8).
  Ставки Ерназара (реестр 114, 13.11 01:03; засчёт — точечное значение, округлённое до 0,01 / 0,1, внутри диапазона):
    DINOv2 отношение 0,20–0,45; DeiT III отношение 0,35–0,65; DINOv2 разрыв 8,0–20,0; DeiT III разрыв 5,0–15,0;
    правило 3 — «не подтверждена».

ВСТРОЕННЫЕ ПРОВЕРКИ:
  1. L00: CLS / нетронутые / дальний фон в S и Sp совпадают -> точность пробы ровно 50,0%.
  2. Пиксели «нетронутых» и «дальнего фона» в S и Sp совпадают (каждая сцена).
  3. Наш пошаговый прогон = собственный прогон модели (макс. разница < 1e-3).
  4. Пустой серый фон: зеркало ничего не меняет (наибольшая разница векторов < 1e-4).
  5. Фигуры в зеркальном опыте симметричны (совпадение маски с зеркалом > 0,95 в обе стороны).
  6. Контроль «близнец-вопрос» > 90% у всех сетей на L12.
  7. Случайные сети отличаются от обученной (отпечаток весов).
  8. Нормировка входа = нормировка из pretrained_cfg timm; сетка и число служебных токенов — как ожидается.

ЗАПУСК (из папки binding-vit; по одной команде):
  caffeinate -i python more_nets.py --model dinov2 2>&1 | tee more_nets_dinov2_log.txt
  caffeinate -i python more_nets.py --model deit3 2>&1 | tee more_nets_deit3_log.txt
  python more_nets_compare.py 2>&1 | tee more_nets_compare_log.txt
  -> more_nets_<model>.csv (зеркало и проба), более ранние файлы не трогаются.
  Средние по группам кэшируются в ~/binding-vit-data/more_nets/ (повторный запуск берёт готовое).
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from PIL import Image

import flip_test as ft
import probe_big as pb
from extract import MEAN, STD, load_image, run_layers, check_forward, fingerprint, pick_device

DATA = "data_v2"
CONF = {
    "dinov2": dict(timm="vit_base_patch14_reg4_dinov2.lvd142m", short="DINOv2", P=14, G=16, n_pre=5,
                   long="обученная DINOv2 ViT-B/14 с регистрами (timm vit_base_patch14_reg4_dinov2.lvd142m)",
                   bet_ratio=(0.20, 0.45), bet_gap=(8.0, 20.0)),
    "deit3": dict(timm="deit3_base_patch16_224.fb_in1k", short="DeiT III", P=16, G=14, n_pre=1,
                  long="обученный DeiT III ViT-B/16, с учителем (timm deit3_base_patch16_224.fb_in1k)",
                  bet_ratio=(0.35, 0.65), bet_gap=(5.0, 15.0)),
}
LAYERS = list(range(13))        # ViT-B: L00 (вход) … L12 (выход 12-го блока)
QUERIES = ["left", "upper"]
QWORD = {"left": "левый", "upper": "верхний"}
NAMES = {"cls": "CLS", "unchanged": "нетронутые", "far_bg": "дальний фон", "far_bg4": "дальний фон ≥4 патчей (справка)",
         "twin": "близнец-вопрос (контроль)", "mean": "среднее по патчам"}
MIRROR_BATCH, PROBE_BATCH = 32, 16


def load_net(arch, seed):
    """seed None -> обученные веса; число -> случайные веса той же архитектуры (seed ДО создания модели)."""
    import timm
    c = CONF[arch]
    kw = dict(img_size=224) if arch == "dinov2" else {}
    if seed is None:
        m = timm.create_model(c["timm"], pretrained=True, **kw)
        w = c["long"]
    else:
        torch.manual_seed(seed)
        m = timm.create_model(c["timm"], pretrained=False, **kw)
        w = f"СЛУЧАЙНЫЕ {c['timm'].split('.')[0]} (нулевая модель), seed {seed}"
    m.eval()
    return m, w


def check_setup(model, arch):
    """Проверка 8: нормировка, сетка, служебные токены, 12 блоков."""
    c = CONF[arch]
    cfg = model.pretrained_cfg
    ok_norm = np.allclose(cfg["mean"], MEAN.flatten().tolist()) and np.allclose(cfg["std"], STD.flatten().tolist())
    grid = tuple(model.patch_embed.grid_size)
    ok = ok_norm and grid == (c["G"], c["G"]) and model.num_prefix_tokens == c["n_pre"] and len(model.blocks) == 12
    print(f"  Проверка 8: нормировка timm {tuple(round(x, 3) for x in cfg['mean'])}/{tuple(round(x, 3) for x in cfg['std'])} "
          f"= наша: {'да' if ok_norm else 'НЕТ'}; сетка {grid}; служебных токенов {model.num_prefix_tokens}; "
          f"блоков {len(model.blocks)}  {'✅' if ok else '⚠️ проверь весь вывод'}")
    return ok


# ------------------------------------------------------------------
# 1. Зеркало
# ------------------------------------------------------------------
def mirror(model, sel, imgs, blank, device):
    n_pre = model.num_prefix_tokens
    groups = ["cls", "mean"]

    @torch.no_grad()
    def feats(x):
        outs = run_layers(model, x.to(device))
        f = torch.stack([torch.stack([h[:, 0], h[:, n_pre:].mean(1)], 1) for h in outs], 1)
        return f.float().cpu().numpy()            # [B, слои, 2, D]

    fb = feats(torch.cat([blank, blank.flip(3), blank.flip(2)]))
    # сравниваем сами векторы, а не косинус: у случайной сети CLS на L00 почти нулевой (инициализация ~1e-6),
    # и косинус с защитой от деления на 0 (1e-12) даёт ложное расстояние ~1e-3 (найдено в облачном тесте 13.11)
    z = float(max(np.abs(fb[0] - fb[1]).max(), np.abs(fb[0] - fb[2]).max()))
    print(f"  Проверка 4 (пустой фон: зеркало ничего не меняет): макс. разница векторов {z:.2e}  {'✅' if z < 1e-4 else '⚠️ проверь весь вывод'}")
    F = {"orig": [], "h": [], "v": []}
    for s in range(0, len(imgs), MIRROR_BATCH):
        x = imgs[s:s + MIRROR_BATCH]
        F["orig"].append(feats(x)); F["h"].append(feats(x.flip(3))); F["v"].append(feats(x.flip(2)))
    F = {k: np.concatenate(v) for k, v in F.items()}
    rng = np.random.default_rng(0)
    out = {}
    for gi, g in enumerate(groups):
        for l in LAYERS:
            dh = ft.cos_dist(F["orig"][:, l, gi], F["h"][:, l, gi])
            dv = ft.cos_dist(F["orig"][:, l, gi], F["v"][:, l, gi])
            r, lo, hi = ft.boot_ratio(dh, dv, rng)
            out[(g, l)] = (r, lo, hi, float(dh.mean()), float(dv.mean()), float((dh < dv).mean()), len(dh))
    return out, z < 1e-4


# ------------------------------------------------------------------
# 2. Привязка
# ------------------------------------------------------------------
def groups_for(rec, colors, P, G):
    grp, ans = pb.scene_groups(rec, colors, p=P, g=G, check_pixels=True)   # проверка 2 внутри
    if P == 14:                     # справка: дальний фон ≥ 4 патчей (56 px)
        mask = np.asarray(Image.open(f"{DATA}/masks/{rec['id']}.png"))
        blk = mask.reshape(G, P, G, P).transpose(0, 2, 1, 3).reshape(G * G, P * P)
        twin = ((blk == 1) | (blk == 2)).any(1)
        empty = (blk == 0).all(1)
        ty, tx = np.where(twin)[0] // G, np.where(twin)[0] % G
        dist = np.array([np.min(np.maximum(abs(q // G - ty), abs(q % G - tx))) for q in range(G * G)])
        grp["far_bg4"] = np.where(empty & (dist >= 4))[0]
    return grp, ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(CONF), required=True)
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--out", default=os.path.expanduser("~/binding-vit-data/more_nets"))
    ap.add_argument("--limit", type=int, default=0, help="только для проверки в облаке: первые N сцен")
    ap.add_argument("--test-random-main", action="store_true",
                    help="только для проверки в облаке: вместо обученной сети — случайная seed 99")
    args = ap.parse_args()
    arch, c = args.model, CONF[args.model]
    P, G = c["P"], c["G"]
    import timm
    test = bool(args.limit or args.test_random_main)
    stem = f"more_nets_{arch}" + (f"_test{args.limit}" if args.limit else "") + ("_rmain" if args.test_random_main else "")
    main_name = c["short"]
    nets = [(main_name, 99 if args.test_random_main else None)] + [(f"{arch}_rand{s}", s) for s in range(args.n_random)]
    print(f"Опыт: ещё две сети — {c['long']} + {args.n_random} случайные сети   выход: {stem}.csv")
    print(f"timm {timm.__version__}, torch {torch.__version__}")
    if test:
        print("⚠️ ПРОВЕРОЧНЫЙ прогон (облако): числа не для статьи.")
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)

    # общие настройки модулей-помощников
    pb.LAYERS, pb.P, pb.G, pb.FAR, pb.BATCH = LAYERS, P, G, 3, PROBE_BATCH
    pb.STORE = ["cls", "unchanged", "far_bg", "twin0", "twin1"] + (["far_bg4"] if P == 14 else [])
    pb.NAMES = NAMES

    meta = [json.loads(line) for line in open(f"{DATA}/meta.jsonl")]
    if args.limit:
        meta = meta[:args.limit]
    colors = sorted({ob[k] for r in meta for ob in r["objects"] for k in ("color", "color_Sp")})
    infos = [groups_for(r, colors, P, G) for r in meta]
    nfar = [len(gr["far_bg"]) for gr, _ in infos]
    print(f"Сцен для пробы: {len(meta)}. Сетка {G}×{G} патчей по {P} px; дальний фон ≥ 3 патчей = {3 * P} px; "
          f"сцен с дальним фоном {sum(1 for x in nfar if x)} (патчей в среднем {np.mean(nfar):.1f})")
    if P == 14:
        n4 = [len(gr["far_bg4"]) for gr, _ in infos]
        print(f"  справка: дальний фон ≥ 4 патчей (56 px) — сцен {sum(1 for x in n4 if x)} (патчей в среднем {np.mean(n4):.1f})")
    print("Проверка 2 ✅ пиксели «нетронутых» и «дальнего фона» в S и Sp совпадают во всех сценах")
    folds = np.random.default_rng(0).permutation(len(meta)) % pb.N_FOLDS     # как probe_v2

    sel = [r for r in meta if all(o["shape"] in ft.SYM_SHAPES for o in r["objects"])]
    iou_h, iou_v = [], []
    for rec in sel:
        mask = np.asarray(Image.open(f"{DATA}/masks/{rec['id']}.png"))
        for k in range(len(rec["objects"])):
            ys, xs = np.where(mask == k + 1)
            obj = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1] == k + 1
            iou_h.append((obj & obj[:, ::-1]).sum() / (obj | obj[:, ::-1]).sum())
            iou_v.append((obj & obj[::-1, :]).sum() / (obj | obj[::-1, :]).sum())
    ok5 = min(np.mean(iou_h), np.mean(iou_v)) > 0.95 and abs(np.mean(iou_h) - np.mean(iou_v)) < 0.01
    print(f"Сцен для зеркала (только круг/квадрат/крест): {len(sel)}.  Проверка 5 (симметрия фигур): "
          f"{np.mean(iou_h):.4f} / {np.mean(iou_v):.4f}  {'✅' if ok5 else '⚠️ проверь весь вывод'}")
    imgs = torch.stack([load_image(f"{DATA}/images/{r['id']}_S.png", arch) for r in sel])
    blank = imgs[:1].clone()
    blank[:] = blank[0, :, :1, :1]

    device = pick_device()
    checks = {"8": True, "3": True, "4": True, "7": True}
    mir, scores, weights_of, fps, warn_total = {}, {}, {}, {}, 0
    for name, seed in nets:
        model, weights = load_net(arch, seed)
        if name == main_name and seed is not None:
            weights = "ПРОВЕРКА: " + weights
        fp = fingerprint(model)
        fps[name] = fp
        weights_of[name] = weights
        print(f"\n### {name}: {weights}   отпечаток весов {fp:.4f}", flush=True)
        checks["8"] &= check_setup(model, arch)
        model.to(device)
        xx = torch.stack([load_image(f"{DATA}/images/{meta[0]['id']}_S.png", arch),
                          load_image(f"{DATA}/images/{meta[0]['id']}_Sp.png", arch)]).to(device)
        with torch.no_grad():
            ref = model(xx)
            mine = model.forward_head(model.norm(run_layers(model, xx)[-1]))
        d3 = float((mine - ref).abs().max())
        checks["3"] &= d3 < 1e-3
        print(f"  Проверка 3 (наши слои = прогон самой модели): макс. разница {d3:.2e}  {'✅' if d3 < 1e-3 else '⚠️ проверь весь вывод'}")
        mir[name], ok4 = mirror(model, sel, imgs, blank, device)
        checks["4"] &= ok4
        r, lo, hi, *_ = mir[name][("mean", 12)]
        print(f"  зеркало, среднее по патчам, L12: {r:.3f} [{lo:.3f}–{hi:.3f}]   ({time.time() - t0:.0f} с)", flush=True)

        cache = os.path.join(args.out, f"{arch}_{'main' if seed is None else 'seed' + str(seed)}"
                                       f"{'_test' + str(args.limit) if args.limit else ''}.npz")
        if os.path.exists(cache):
            z = np.load(cache)
            assert list(z["ids"]) == [r["id"] for r in meta] and list(z["layers"]) == LAYERS \
                and list(z["store"]) == pb.STORE, f"{cache}: не тот набор — удали файл и запусти снова"
            Fm, valid = z["F"], z["valid"]
            print(f"  средние по группам взяты из {cache}")
        else:
            print(f"  снимаю средние по группам -> {cache}", flush=True)
            Fm, valid = pb.extract(model, meta, infos, device, arch, cache)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        sc, groups, nw = probe_groups(Fm, valid, infos, folds, P)
        warn_total += nw
        scores[name] = sc
        print(f"  проба готова ({time.time() - t0:.0f} с), предупреждений «не сошлось»: {nw}", flush=True)

    rand = [n for n, s in nets[1:]]
    checks["7"] = all(abs(fps[n] - fps[main_name]) > 1e-3 for n in rand) and len(set(round(fps[n], 3) for n in rand)) == len(rand)
    # ---------------- отчёт ----------------
    rows = []
    for name in mir:
        for (g, l), (r, lo, hi, mh, mv, frac, n) in mir[name].items():
            rows.append([name, weights_of[name], "mirror", g, "", f"L{l:02d}", f"{r:.5f}", f"{lo:.5f}", f"{hi:.5f}", n])
    res, gap = {}, {}
    for name in scores:
        rng = np.random.default_rng(0)
        for q in QUERIES:
            for g in scores[name][q]:
                for li, l in enumerate(LAYERS):
                    a, lo, hi, n = pb.boot(scores[name][q][g][li], rng)
                    res[(name, q, g, l)] = (a, lo, hi, n)
                    rows.append([name, weights_of[name], "probe", g, q, f"L{l:02d}", f"{a:.5f}", f"{lo:.5f}", f"{hi:.5f}", n])
        for g in scores[name]["left"]:
            if g == "twin":
                continue
            for li, l in enumerate(LAYERS):
                d = scores[name]["upper"][g][li] - scores[name]["left"][g][li]
                a, lo, hi, n = pb.boot(d, np.random.default_rng(0))
                gap[(name, g, l)] = (a, lo, hi, n)
                rows.append([name, weights_of[name], "gap_upper_minus_left", g, "", f"L{l:02d}", f"{a:.5f}", f"{lo:.5f}", f"{hi:.5f}", n])
    with open(f"{stem}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["net", "weights", "measure", "group", "query", "layer", "value", "ci_lo", "ci_hi", "n_scenes"])
        w.writerows(rows)

    groups = list(scores[main_name]["left"].keys())
    l00 = [res[(n, q, g, 0)][0] for n in scores for q in QUERIES for g in groups if g != "twin"]
    ok1 = all(abs(x - 0.5) < 1e-12 for x in l00)
    print(f"\nПроверка 1 (L00: CLS / нетронутые / дальний фон ровно 50,0%): {'✅' if ok1 else '⚠️ проверь весь вывод'}")
    okc = all(res[(n, q, "twin", 12)][0] > 0.9 for n in scores for q in QUERIES)
    print(f"Проверка 6 (контроль «близнец-вопрос» > 90% у всех сетей на L12): {'✅' if okc else '⚠️ проверь весь вывод'}")
    print(f"Проверка 7 (случайные сети отличаются от обученной и друг от друга): {'✅' if checks['7'] else '⚠️ проверь весь вывод'}  "
          f"отпечатки: " + ", ".join(f"{n} {fps[n]:.4f}" for n in fps))
    print(f"Проверки 3 / 4 / 8 у всех сетей: {'✅' if checks['3'] else '⚠️'} / {'✅' if checks['4'] else '⚠️'} / {'✅' if checks['8'] else '⚠️'}")
    print(f"Предупреждений «обучение не сошлось» всего: {warn_total}  {'✅' if warn_total == 0 else '⚠️ проверь весь вывод'}")

    print("\n==== Зеркало: d(←→) / d(↑↓); < 1 = зеркало слева-направо замечается слабее ====")
    for g in ("mean", "cls"):
        print(f"\n== {NAMES[g]} ==\nслой   {main_name} [95% инт.]     случайные (мин–макс)")
        for l in LAYERS:
            r, lo, hi, *_ = mir[main_name][(g, l)]
            rr = [mir[n][(g, l)][0] for n in rand]
            if np.isnan(r):
                print(f"L{l:02d}   —")
                continue
            print(f"L{l:02d}  {r:5.2f} [{lo:.2f}–{hi:.2f}]     {min(rr):.2f}–{max(rr):.2f}")

    for q in QUERIES:
        print(f"\n==== Проба «какого цвета {QWORD[q]} близнец», логрег (50% = мешок признаков) ====")
        for g in groups:
            print(f"\n== {NAMES[g]} ==   сцен: {res[(main_name, q, g, 12)][3]}\nслой   {main_name} [95% инт.]     случайные (мин–макс)")
            for l in LAYERS:
                a, lo, hi, _ = res[(main_name, q, g, l)]
                rr = [res[(n, q, g, l)][0] for n in rand]
                print(f"L{l:02d}  {a * 100:5.1f}% [{lo * 100:5.1f}–{hi * 100:5.1f}]   {min(rr) * 100:5.1f}–{max(rr) * 100:5.1f}%")

    print("\n==== Разрыв «верхний − левый» (п.п.), логрег; 95% инт. — парный бутстрэп по сценам ====")
    for g in [g for g in groups if g != "twin"]:
        print(f"\n== {NAMES[g]} ==\nслой   {main_name} [95% инт.]       случайные (мин … макс)")
        for l in LAYERS:
            a, lo, hi, _ = gap[(main_name, g, l)]
            rr = [gap[(n, g, l)][0] * 100 for n in rand]
            print(f"L{l:02d}  {a * 100:5.1f} [{lo * 100:5.1f} … {hi * 100:5.1f}]     {min(rr):5.1f} … {max(rr):5.1f}")

    r, lo, hi, *_ = mir[main_name][("mean", 12)]
    a, glo, ghi, _ = gap[(main_name, "far_bg", 12)]
    v1 = "слепа к зеркалу" if hi < 0.90 else "не показано"
    v2 = "разрыв есть" if glo > 0 else "нет"
    print(f"\nВЫВОД ПО ПРАВИЛАМ (реестр 114–115, записано до запуска), {main_name}:")
    print(f"  Правило 1: отношение (среднее по патчам, L12) {r:.2f} [{lo:.2f}–{hi:.2f}] -> «{v1}»   (порог: верхняя граница < 0,90)")
    print(f"  Правило 2: разрыв (дальний фон, L12) {a * 100:.1f} [{glo * 100:.1f} … {ghi * 100:.1f}] п.п. -> «{v2}»   (порог: нижняя граница > 0)")
    print(f"  Правило 3 (6 сетей) — после обоих прогонов: python more_nets_compare.py")
    if not test:
        br, bg = c["bet_ratio"], c["bet_gap"]
        h1 = br[0] <= round(r, 2) <= br[1]
        h2 = bg[0] <= round(a * 100, 1) <= bg[1]
        print(f"\nСТАВКА ЕРНАЗАРА (реестр 114, 13.11 01:03):")
        print(f"  отношение {br[0]:.2f}–{br[1]:.2f} -> вышло {r:.2f}  {'ПОПАЛ ✅' if h1 else 'НЕ ПОПАЛ ❌'}")
        print(f"  разрыв {bg[0]:.1f}–{bg[1]:.1f} -> вышло {a * 100:.1f}  {'ПОПАЛ ✅' if h2 else 'НЕ ПОПАЛ ❌'}")
    print(f"\nЧисла: {stem}.csv   ({time.time() - t0:.0f} с)")


def probe_groups(Fm, valid, infos, folds, P):
    """Как probe_big.probe_net, но с группой far_bg4 (справка, только у сетки 14 px). Только логрег."""
    n_sc = len(infos)
    groups = ["cls", "unchanged", "far_bg"] + (["far_bg4"] if P == 14 else []) + ["twin"]
    scores = {q: {g: [] for g in groups} for q in QUERIES}
    n_warn = 0
    for li, l in enumerate(LAYERS):
        for q in QUERIES:
            for g in groups:
                X, yt, yr, sc = [], [], [], []
                for si, (_, answers) in enumerate(infos):
                    k, lab = answers[q]
                    gi = pb.STORE.index(f"twin{k}" if g == "twin" else g)
                    for vi, v in enumerate(("S", "Sp")):
                        i = 2 * si + vi
                        if valid[i, gi]:
                            X.append(Fm[li, i, gi]); yt.append(lab[v][0]); yr.append(lab[v][1]); sc.append(si)
                ps, nw = pb.run_probe(np.asarray(X, dtype=np.float32), np.array(yt), np.array(yr), np.array(sc),
                                      folds, ["logreg"], n_sc)
                n_warn += nw
                scores[q][g].append(ps["logreg"])
    return scores, groups, n_warn


if __name__ == "__main__":
    main()

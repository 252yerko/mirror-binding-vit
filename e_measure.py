"""
e_measure.py — опыт E, ПИЛОТ: меры на контрольных точках обученной ViT-Tiny и на трёх случайных ViT-Tiny.
Добавлено 16.11 (реестр 241). Сеть обучает e_train.py; критерии пилота записаны в его заголовке ДО запуска.

МЕРЫ (те же функции, что в прежних опытах; код этих файлов не меняется):
  many_nets.run_net (опыт B): зеркальное отношение (631 сцена из круга / квадрата / креста, среднее по патчам, последний
      слой, ср. d(←→) / ср. d(↑↓)); обычная проба «левый» / «верхний» (дальний фон, последний слой, все 2000 сцен data_v2,
      логрег C = 0,1, 5 фолдов по сценам); разрыв «верхний − левый»; R; разрыв по CLS.
  heads_pair (опыт G): смещения голов dx и мера пары D = S_зерк − S_прям для всех пар голов одного блока
      (у ViT-Tiny 3 головы в блоке → 3 пары в блоке, 36 пар); порог T — наибольшее D у трёх случайных ViT-Tiny,
      порог |dx| — наибольший |dx| у них же; пара «найдена», если dx противоположны (интервалы без 0, |dx| выше порога)
      и D > T (правило G3, как в osf.io/8nxg7).
  Точность top-1 на проверочных фото Food-101 — для последней контрольной точки (критерий i).
Части отношения d(←→) и d(↑↓) печатаются всегда (замечание Ерназара 14:42: на фото еды, снятой сверху, сеть может стать
  менее чувствительной к обоим отражениям, и отношение подтянется к 1 при слепоте к лево-праву).
Случайные ViT-Tiny: seed 100, 101, 102 (не совпадают с начальными весами обучаемой сети, seed 0) — отношение, пробы;
  пороги T и |dx| для пар голов — по десяти случайным ViT-Tiny (seed 100–109: 360 пар; у ViT-B порог брался по 3 сетям ×
  792 пары, у ViT-Tiny пар в 22 раза меньше, и порог по трём сетям был бы слишком мягким).
  Начальные веса обучаемой сети (эпоха 0) — отдельная точка траектории (справка); проверяется, что они совпадают с
  начальными весами из e_train.py.
Исправлено 16.11 до запуска (реестр 243): имя кэша признаков включает отпечаток весов (иначе после повторного обучения
  молча взялись бы старые признаки); предупреждение, если расстояния d(←→), d(↑↓) близки к точности кэша float16;
  критерии считаются окончательными только для эпохи 40.

ЗАПУСК (из папки binding-vit, после e_train.py; ~30–60 мин):
  caffeinate -i python -u e_measure.py --cond 2 --seed 0 2>&1 | tee e_measure_c2_s0_log.txt
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import time

import numpy as np
import torch

import flip_test as ft
import heads as H
import heads_pair as hp
import many_nets as mn
import mirror_decomp as md
from extract import fingerprint, pick_device

RAND = (100, 101, 102)
RAND_THR = tuple(range(100, 110))
D_FLOOR = 1e-5                 # ниже — расстояния сравнимы с точностью кэша float16 (шум ~1e-7): отношение ненадёжно
FIELDS = ["point", "epoch", "seed", "fp", "ratio", "ratio_lo", "ratio_hi", "d_lr", "d_ud", "left", "upper", "gap", "gap_lo", "gap_hi",
          "cls_gap", "R", "n_pairs", "T", "dx_thr", "max_abs_dx", "top1"]


def tiny(seed):
    import timm
    torch.manual_seed(seed)
    return timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=101)


def from_ckpt(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = tiny(0)
    m.load_state_dict(ck["model"])
    return m, ck["epoch"]


def measure(model, name, meta, args, device, colors, folds_all, ok):
    model.eval()
    name = f"{name} fp{fingerprint(model):.4f}"                 # отпечаток весов в имени кэша: другие веса — другой кэш
    mn.load = lambda timm_name, seed: (model, name)            # run_net берёт модель отсюда (без загрузки весов timm)
    r = mn.run_net(name, "vit_tiny_patch16_224", "E", "Ti", 16, None, meta, args, device, colors, folds_all, ok)
    # числитель и знаменатель отношения — из кэша run_net (средние по патчам, последний слой, картинка S)
    safe = name.replace("/", "_").replace(" ", "_")
    z = np.load(os.path.join(args.out, f"{safe}{'_test' + str(args.limit) if args.limit else ''}.npz"))
    PMS = z["PMS"].astype(np.float16).astype(np.float32)
    r["d_lr"] = float(ft.cos_dist(PMS[:, 0], PMS[:, 1]).mean())
    r["d_ud"] = float(ft.cos_dist(PMS[:, 0], PMS[:, 2]).mean())
    assert abs(r["d_lr"] / r["d_ud"] - r["ratio"]) < 1e-6 * max(1.0, r["ratio"]), "числитель / знаменатель не дают отношение"
    low = min(r["d_lr"], r["d_ud"]) < D_FLOOR
    ok["floor"] = ok.get("floor", 0) + int(low)
    print(f"  части отношения: d(←→) {r['d_lr']:.3e}, d(↑↓) {r['d_ud']:.3e}"
          + ("  ⚠️ расстояния близки к точности кэша (float16) — отношение этой сети ненадёжно" if low else ""), flush=True)
    return r


def pair_stats(model, sym_arrs, device):
    cfg = model.pretrained_cfg
    mean = torch.tensor(cfg["mean"]).view(3, 1, 1)
    std = torch.tensor(cfg["std"]).view(3, 1, 1)
    model.to(device).eval()
    R, worst = hp.maps_pass(model, sym_arrs, device, mean, std, model.num_prefix_tokens)
    return R, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpts", default=os.path.expanduser("~/binding-vit-data/e_pilot"))
    ap.add_argument("--out", default=os.path.expanduser("~/binding-vit-data/e_pilot/measure"))
    ap.add_argument("--food", default=os.path.expanduser("~/binding-vit-data/food-101"))
    ap.add_argument("--limit", type=int, default=0, help="только для проверки в облаке: столько сцен")
    ap.add_argument("--no-top1", action="store_true", help="только для проверки в облаке")
    args = ap.parse_args()
    args.only, args.random_only, args.test_random_main = None, False, False
    import timm
    device = pick_device()
    print(f"Опыт E, пилот: меры   timm {timm.__version__}, torch {torch.__version__}, устройство {device}")
    print(f"md5 e_measure.py {hashlib.md5(open(__file__, 'rb').read()).hexdigest()}")
    if args.limit:
        print("⚠️ ПРОВЕРОЧНЫЙ прогон: числа не для статьи.")
    # ViT-Tiny: 3 головы в блоке
    H.NH = hp.NH = 3
    meta = [json.loads(line) for line in open(f"{mn.DATA}/meta.jsonl")]
    if args.limit:
        meta = meta[:args.limit]
    print(f"{mn.DATA}/meta.jsonl md5 {hashlib.md5(open(f'{mn.DATA}/meta.jsonl', 'rb').read()).hexdigest()}; сцен {len(meta)}")
    colors = sorted({ob[k] for r in meta for ob in r["objects"] for k in ("color", "color_Sp")})
    folds_all = np.random.default_rng(0).permutation(len(meta)) % md.N_FOLDS
    sym = [r for r in meta if all(o["shape"] in ft.SYM_SHAPES for o in r["objects"])]
    sym_arrs = [H.load_u8(f"{H.DATA}/images/{r['id']}_S.png") for r in sym]
    print(f"симметричных сцен {len(sym)}")
    out = os.path.join(args.out, f"c{args.cond}_s{args.seed}")
    os.makedirs(out, exist_ok=True)
    args.out = out                                               # кэш признаков run_net
    ok = {k: True for k in ("1", "3", "4", "5", "6", "8")}
    ok["warn"], ok["attn"] = 0, 0.0
    rows = []

    # 1) случайные ViT-Tiny: пороги T и |dx|; отношение и «верхний»
    T, dx_thr, rnd = -np.inf, 0.0, []
    for s in RAND_THR:
        m = tiny(s)
        Rm, w = pair_stats(m, sym_arrs, device)
        if s not in RAND:                                        # только пороги для пар голов
            ok["attn"] = max(ok["attn"], w)
            Dr = Rm["Sm"].mean(0, dtype=np.float64) - Rm["Sd"].mean(0, dtype=np.float64)
            dvals = np.array([Dr[b - 1, i, j] for b, i, j in hp.within_pairs(hp.ALL_BLOCKS)])
            T = max(T, float(dvals.max()))
            mdx = float(np.abs(Rm["dx"].mean(0)).max())
            dx_thr = max(dx_thr, mdx)
            print(f"  случайная seed {s} (только пороги): D макс {dvals.max():+.4f}; |dx| макс {mdx:.3f}", flush=True)
            continue
        r = measure(m, f"ViT-Tiny случайная seed {s}", meta, args, device, colors, folds_all, ok)
        ok["attn"] = max(ok["attn"], w)
        Dr = Rm["Sm"].mean(0, dtype=np.float64) - Rm["Sd"].mean(0, dtype=np.float64)
        dvals = np.array([Dr[b - 1, i, j] for b, i, j in hp.within_pairs(hp.ALL_BLOCKS)])
        T = max(T, float(dvals.max()))
        mdx = float(np.abs(Rm["dx"].mean(0)).max())
        dx_thr = max(dx_thr, mdx)
        rnd.append(r)
        rows.append(dict(point="random", epoch="", seed=s, fp=r["fp"], ratio=r["ratio"], ratio_lo=r["ratio_lo"], ratio_hi=r["ratio_hi"], d_lr=r["d_lr"], d_ud=r["d_ud"],
                         left=r["left"], upper=r["upper"], gap=r["gap"], gap_lo=r["gap_lo"], gap_hi=r["gap_hi"], cls_gap=r["cls_gap"],
                         R=r["R"], n_pairs="", T="", dx_thr="", max_abs_dx=mdx, top1=""))
        print(f"  случайная seed {s}: D по {len(dvals)} парам: мин {dvals.min():+.4f}, макс {dvals.max():+.4f}; |dx| макс {mdx:.3f}", flush=True)
    print(f"\nПороги по {len(RAND_THR)} случайным ViT-Tiny ({len(RAND_THR) * 36} пар): T = {T:+.4f}, |dx| = {dx_thr:.4f}")

    # 2) траектория: эпоха 0 (начальные веса обучаемой сети) и контрольные точки
    import e_train as et
    m0, m0t = tiny(args.seed), et.make_model(args.seed, 101)
    ok["init"] = all(torch.equal(a, b) for a, b in zip(m0.state_dict().values(), m0t.state_dict().values()))
    pts = [("ep00", m0, 0)]
    files = sorted(glob.glob(os.path.join(args.ckpts, f"c{args.cond}_s{args.seed}", "ep*.pt")))
    for f in files:
        m, e = from_ckpt(f)
        pts.append((os.path.basename(f)[:-3], m, e))
    assert len(pts) > 1, "нет контрольных точек — сначала e_train.py"
    last_r, last_pairs = None, None
    fps_rand = [r["fp"] for r in rnd]
    for name, m, e in pts:
        r = measure(m, f"ViT-Tiny c{args.cond} s{args.seed} {name}", meta, args, device, colors, folds_all, ok)
        Rm, w = pair_stats(m, sym_arrs, device)
        ok["attn"] = max(ok["attn"], w)
        st = hp.summarize(Rm)
        st["dx_thr"] = dx_thr
        found = hp.g3_pairs(st, T)
        mdx = float(np.abs(st["dx"]).max())
        rows.append(dict(point=name, epoch=e, seed=args.seed, fp=r["fp"], ratio=r["ratio"], ratio_lo=r["ratio_lo"], ratio_hi=r["ratio_hi"], d_lr=r["d_lr"], d_ud=r["d_ud"],
                         left=r["left"], upper=r["upper"], gap=r["gap"], gap_lo=r["gap_lo"], gap_hi=r["gap_hi"], cls_gap=r["cls_gap"],
                         R=r["R"], n_pairs=len(found), T=T, dx_thr=dx_thr, max_abs_dx=mdx, top1=""))
        pairs_txt = "; ".join(f"{b}.{i}/{b}.{j} D {st['D'][b - 1, i, j]:+.3f} (dx {st['dx'][b - 1, i]:+.2f}/{st['dx'][b - 1, j]:+.2f})"
                              for b, i, j in sorted(found, key=lambda p: -st["D"][p[0] - 1, p[1], p[2]])[:6])
        print(f"  {name}: зеркальных пар (G3) {len(found)} из 36; |dx| макс {mdx:.3f}" + (f"; {pairs_txt}" if found else ""), flush=True)
        if e > 0:
            ok.setdefault("7", True)
            ok["7"] &= all(abs(r["fp"] - f) > 1e-3 for f in fps_rand)
        last_r, last_pairs, last_m, last_e = r, len(found), m, e

    # 3) критерий (i): точность на проверочных фото последней точки
    top1 = None
    if not args.no_top1:
        cfg = last_m.pretrained_cfg
        te = et.Food(args.food, "test", et.transforms(args.cond, tuple(cfg["mean"]), tuple(cfg["std"]), False))
        dte = torch.utils.data.DataLoader(te, batch_size=256, shuffle=False, num_workers=6)
        top1 = et.evaluate(last_m.to(device), dte, device)
        rows[-1]["top1"] = top1

    # 4) критерии пилота
    c1 = top1 is not None and top1 >= 0.20
    up_r = [r["upper"] for r in rnd]
    c2 = all(last_r["upper"] - u >= 0.05 for u in up_r)
    c3 = last_r["ratio_hi"] < min(r["ratio"] for r in rnd)
    final = last_e == 40
    print(f"\nКРИТЕРИИ ПИЛОТА (записаны до запуска, реестр 241 и 243), последняя контрольная точка — эпоха {last_e}"
          + ("" if final else "  ⚠️ обучение не закончено (нужна эпоха 40): критерии предварительные") + ":")
    print(f"  (i)   top-1 на всех проверочных фото {'—' if top1 is None else f'{top1 * 100:.2f}%'} (нужно ≥ 20%): {'да' if c1 else 'нет'}")
    print(f"  (ii)  «верхний» {last_r['upper'] * 100:.2f}% против случайных {', '.join(f'{u * 100:.2f}' for u in up_r)}% "
          f"(нужно выше каждой на ≥ 5 п.п.): {'да' if c2 else 'нет'}")
    print(f"  (iii) отношение {last_r['ratio']:.3f} [{last_r['ratio_lo']:.3f}–{last_r['ratio_hi']:.3f}] против случайных "
          f"{', '.join(f'{r['ratio']:.3f}' for r in rnd)} (верхняя граница должна быть ниже наименьшей): {'да' if c3 else 'нет'}")
    verdict = "пилот пройден" if (c1 and c2 and c3) else "пилот не пройден"
    print(f"  -> «{verdict}»" + ("" if final else " (предварительно)"))
    if c1 and c2:
        print("  (i) и (ii) выполнены -> второй шаг пилота: та же сеть без отражений (e_train.py --cond 1, тот же seed),"
              " затем сравнение частей отношения между условиями 2 и 1.")
    print(f"  справка: части отношения d(←→) {last_r['d_lr']:.3e}, d(↑↓) {last_r['d_ud']:.3e}; у случайных "
          + ", ".join(f"{r['d_lr']:.3e} / {r['d_ud']:.3e}" for r in rnd))
    print(f"  справка: «левый» {last_r['left'] * 100:.2f}%, разрыв «верхний − левый» {last_r['gap'] * 100:.2f} "
          f"[{last_r['gap_lo'] * 100:.2f}; {last_r['gap_hi'] * 100:.2f}] п.п.; зеркальных пар (G3) {last_pairs} из 36")

    path = os.path.join(os.getcwd(), f"e_measure_c{args.cond}_s{args.seed}{'_test' if args.limit else ''}.csv")
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in FIELDS})
    print()
    for k, t in (("1", "L00 ровно 50,0%"), ("3", "прогон = forward_features"), ("4", "пустая картинка не меняется при отражениях"),
                 ("5", "симм. проба: пары в сумме 1"), ("6", "контроль «близнец-вопрос» > 90%"), ("8", "сетка / токены / блоки")):
        print(f"Проверка {k} ({t}): {'✅' if ok[k] else '⚠️ проверь весь вывод'}")
    print(f"Проверка 7 (обученные точки отличаются от случайных сетей): {'✅' if ok.get('7', False) else '⚠️ проверь весь вывод'}")
    print(f"Проверка 10 (эпоха 0 = начальные веса e_train.py): {'✅' if ok['init'] else '⚠️ проверь весь вывод'}")
    print(f"Проверка 11 (расстояния выше точности кэша; сетей ниже порога {D_FLOOR:.0e}: {ok.get('floor', 0)}): "
          f"{'✅' if ok.get('floor', 0) == 0 else '⚠️ отношение этих сетей ненадёжно — см. выше'}")
    print(f"Проверка 9 (строки внимания в сумме 1, отклонение {ok['attn']:.1e}): {'✅' if ok['attn'] < 1e-4 else '⚠️ проверь весь вывод'}")
    print(f"Предупреждений «не сошлось»: {ok['warn']}  {'✅' if ok['warn'] == 0 else '⚠️ проверь весь вывод'}")
    print(f"Числа: {path}")


if __name__ == "__main__":
    main()

"""
e_measure_full_c67.py — опыт E, ПОЛНЫЙ (ImageNet-100, обучение на Kaggle e_train_kaggle.py): меры на контрольных точках.
Добавлено 19.11 (реестр 285). Основа — e_measure.py пилота (a0c5289c…): те же функции мер и проверки; отличия ниже.
Исправлено 19.11 (реестр 287, перепроверка): мера эквивариантности считала float64 на устройстве — на MPS это ошибка, токены
  теперь переводятся на CPU; проверка 13 дополнена (все эпохи на месте, эпоха = имя файла, float16 на T4); печать float16 «да/нет».

ОТЛИЧИЯ ОТ ПИЛОТА:
  (1) сеть — ViT-Tiny со 100 классами (голова классов на меры не влияет: меры берутся до головы); случайные ViT-Tiny тоже
      со 100 классами (seed 100–109; отношение и пробы — seed 100–102, пороги пар голов — все десять);
  (2) оба считывателя для «левого» и «верхнего» — дальний фон И CLS, каждый со своим 95% интервалом (бутстреп по сценам);
      в пилоте по CLS печатался только разрыв (план E, раздел 3: у коллажей сторона может жить только в CLS);
  (3) НОВАЯ МЕРА (план E, раздел 3, «эквивариантность»): по 631 симметричной сцене, последний слой, патчи (196 токенов):
        E(←→) = ср. по патчам cos-расстояние( h(отражённой ←→ картинки)[p], h(картинки)[зеркальный ←→ p] ),
        B(←→) = ср. по патчам cos-расстояние( h(отражённой ←→ картинки)[p], h(картинки)[p] ),
      и то же для ↑↓. B — «насколько каждый патч изменился на своём месте» (B = 0 — сеть на этом месте отражения не видит);
      E — «насколько отражённая карта совпадает с переставленной исходной» (E = 0 — сеть хранит сторону, переставляя токены,
      т. е. эквивариантна). Печатаются E, B и q = E / B (всегда вместе с частями); q < 1 — ближе к «сторона хранится в
      переставленных токенах», q > 1 — ближе к «на своём месте ничего не меняется». Мера описательная (справка), не правило.
      Мера проверена в облаке на искусственных сетях с известным ответом (функция selftest, запуск --selftest; проверка 14):
      сеть «только свой патч, симметричное ядро, без позиций» (точно эквивариантна) → E = 0; сеть «общий признак
      картинки + позиция» (на своём месте точно не меняется) → B = 0; случайная ViT-Tiny на шуме — справка;
      токены переводятся на CPU до float64 (MPS не умеет float64; исправлено 19.11, реестр 287);
  (4) нет критериев пилота и top-1 на фото (top-1 и точность задач коллажей печатает e_train_kaggle.py на Kaggle по
      всем 5000 проверочным фото / коллажам каждую эпоху — в логе обучения);
  (5) проверка 13 (контрольные точки): условие и зерно в файле совпадают с запрошенными; список классов = CMC; эпоха в файле
      = эпохе в имени; есть все эпохи 1, 2, 3, 5, 8, 12, 20, 30, 40; все точки — float16 на T4 (платформа регистрации);
      точность, GPU и md5 каждой точки печатаются;
  (6) проверка 10: эпоха 0 = начальные веса e_train_kaggle.make_model(seed, 100).
  (7) ВЕРСИЯ ДЛЯ КОЛЛАЖЕЙ v3 (файл e_measure_full_c67.py, 19.11, реестр 305; зарегистрированный e_measure_full.py a5bd4b27… для
      условий 1–3 не меняется и импортирует неизменный e_train_kaggle.py v4): условия 6–7 коллажей v3 (две головы, выход 200): сеть строится с выходом из контрольной
      точки (n_out), проверка 10 — с make_model(seed, 200); меры читают токены и голову не используют. Условия 1–5 — без изменений.

ЗАПУСК (из папки binding-vit; контрольные точки ep*.pt — в ~/binding-vit-data/e_full/c<cond>_s<seed>/):
  caffeinate -i python -u e_measure_full.py --cond 1 --seed 0 2>&1 | tee e_measure_full_c1_s0_log.txt
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
NCLS = 100
D_FLOOR = 1e-5
FIELDS = ["point", "epoch", "seed", "fp", "ratio", "ratio_lo", "ratio_hi", "d_lr", "d_ud",
          "left", "left_lo", "left_hi", "upper", "upper_lo", "upper_hi", "gap", "gap_lo", "gap_hi",
          "cls_left", "cls_left_lo", "cls_left_hi", "cls_upper", "cls_upper_lo", "cls_upper_hi", "cls_gap", "cls_gap_lo", "cls_gap_hi",
          "R", "E_lr", "B_lr", "q_lr", "q_lr_lo", "q_lr_hi", "E_ud", "B_ud", "q_ud", "q_ud_lo", "q_ud_hi",
          "n_pairs", "T", "dx_thr", "max_abs_dx"]


def tiny(seed, n_out=NCLS):
    import timm
    torch.manual_seed(seed)
    return timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=n_out)


def from_ckpt(path, cond, seed, cmc):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = tiny(0, ck.get("n_out", NCLS))   # v2: условия 6–7 — выход 200 (две головы); пробы читают токены, голова не участвует
    m.load_state_dict(ck["model"])
    good = (ck.get("cond") == cond and ck.get("seed") == seed and list(ck.get("classes", [])) == sorted(cmc)
            and os.path.basename(path) == f"ep{ck['epoch']:02d}.pt")
    return m, ck["epoch"], good, ck.get("amp"), ck.get("gpu")


def cos_dist_t(a, b):
    return 1.0 - torch.nn.functional.cosine_similarity(a.double(), b.double(), dim=-1)


@torch.no_grad()
def equivariance(model, arrs, device, mean, std, n_pre, batch=32):
    """По сценам: E_lr, B_lr, E_ud, B_ud (средние по 196 патчам последнего слоя после norm)."""
    model.to(device).eval()
    out = {k: [] for k in ("E_lr", "B_lr", "E_ud", "B_ud")}
    for s in range(0, len(arrs), batch):
        x = H.to_x(arrs[s:s + batch], mean, std).to(device)
        n = x.shape[0]
        # [n, строка, столбец, D]; на CPU до float64 (MPS не умеет float64 — как в heads_pair.maps_pass)
        f = lambda z: model.forward_features(z)[:, n_pre:].float().cpu().reshape(n, 14, 14, -1)
        t0, th, tv = f(x), f(x.flip(3)), f(x.flip(2))
        out["E_lr"].append(cos_dist_t(th, t0.flip(2)).mean((1, 2)).cpu())
        out["B_lr"].append(cos_dist_t(th, t0).mean((1, 2)).cpu())
        out["E_ud"].append(cos_dist_t(tv, t0.flip(1)).mean((1, 2)).cpu())
        out["B_ud"].append(cos_dist_t(tv, t0).mean((1, 2)).cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def eq_summary(eq):
    r = {}
    for a in ("lr", "ud"):
        E, B = eq[f"E_{a}"], eq[f"B_{a}"]
        q = md.boot_ratio(E, B, np.random.default_rng(0))
        r[f"E_{a}"], r[f"B_{a}"], r[f"q_{a}"], r[f"q_{a}_lo"], r[f"q_{a}_hi"] = float(E.mean()), float(B.mean()), q[0], q[1], q[2]
    return r


def readers(name, meta, args, folds_all, ok):
    """«Левый» / «верхний» по дальнему фону и по CLS (последний слой, все сцены) с интервалами — из кэша run_net."""
    safe = name.replace("/", "_").replace(" ", "_")
    z = np.load(os.path.join(args.out, f"{safe}{'_test' + str(args.limit) if args.limit else ''}.npz"))
    FS, FX, PMS = (z[k].astype(np.float32) for k in ("FS", "FX", "PMS"))
    infos = mn.INFO[(16, args.limit)]
    sym_i = [i for i, r in enumerate(meta) if all(o["shape"] in ft.SYM_SHAPES for o in r["objects"])]
    rest_i = [i for i in range(len(meta)) if i not in set(sym_i)]
    lab = {q: np.array([[infos[i][1][q][1][v] for v in ("S", "Sp")] for i in range(len(meta))]) for q in ("left", "upper")}
    res = {}
    for tag, gi in (("", 1), ("cls_", 0)):
        X = np.zeros((len(meta), 2, FS.shape[-1]), dtype=np.float32)
        X[sym_i], X[rest_i] = FS[1, :, :, 0, gi], FX[1, :, :, 0, gi]
        acc = {}
        for q in ("left", "upper"):
            acc[q], nw = md.probe_plain(X, lab[q][:, :, 0], lab[q][:, :, 1], folds_all)
            ok["warn"] += nw
            b = md.boot(acc[q], np.random.default_rng(0))
            res[f"{tag}{q}"], res[f"{tag}{q}_lo"], res[f"{tag}{q}_hi"] = b[0], b[1], b[2]
        g = md.boot(acc["upper"] - acc["left"], np.random.default_rng(0))
        res[f"{tag}gap"], res[f"{tag}gap_lo"], res[f"{tag}gap_hi"] = g[0], g[1], g[2]
    res["d_lr"] = float(ft.cos_dist(PMS[:, 0], PMS[:, 1]).mean())
    res["d_ud"] = float(ft.cos_dist(PMS[:, 0], PMS[:, 2]).mean())
    return res


def measure(model, name, meta, args, device, colors, folds_all, ok, sym_arrs):
    model.eval()
    name = f"{name} fp{fingerprint(model):.4f}"
    mn.load = lambda timm_name, seed: (model, name)
    r = mn.run_net(name, "vit_tiny_patch16_224", "E", "Ti", 16, None, meta, args, device, colors, folds_all, ok)
    rr = readers(name, meta, args, folds_all, ok)
    # совпадение с run_net: «левый», «верхний», разрыв по фону и разрыв по CLS — те же числа
    same = (abs(rr["left"] - r["left"]) < 1e-9 and abs(rr["upper"] - r["upper"]) < 1e-9 and abs(rr["gap"] - r["gap"]) < 1e-9
            and abs(rr["cls_gap"] - r["cls_gap"]) < 1e-9 and abs(rr["d_lr"] / rr["d_ud"] - r["ratio"]) < 1e-6 * max(1.0, r["ratio"]))
    ok["12"] &= same
    r.update(rr)
    low = min(r["d_lr"], r["d_ud"]) < D_FLOOR
    ok["floor"] = ok.get("floor", 0) + int(low)
    cfg = model.pretrained_cfg
    mean, std = torch.tensor(cfg["mean"]).view(3, 1, 1), torch.tensor(cfg["std"]).view(3, 1, 1)
    r.update(eq_summary(equivariance(model, sym_arrs, device, mean, std, model.num_prefix_tokens)))
    print(f"  части отношения: d(←→) {r['d_lr']:.3e}, d(↑↓) {r['d_ud']:.3e}"
          + ("  ⚠️ расстояния близки к точности кэша (float16)" if low else ""), flush=True)
    print(f"  дальний фон: «левый» {r['left'] * 100:.1f} [{r['left_lo'] * 100:.1f}; {r['left_hi'] * 100:.1f}], «верхний» {r['upper'] * 100:.1f} "
          f"[{r['upper_lo'] * 100:.1f}; {r['upper_hi'] * 100:.1f}]; CLS: «левый» {r['cls_left'] * 100:.1f} [{r['cls_left_lo'] * 100:.1f}; "
          f"{r['cls_left_hi'] * 100:.1f}], «верхний» {r['cls_upper'] * 100:.1f} [{r['cls_upper_lo'] * 100:.1f}; {r['cls_upper_hi'] * 100:.1f}], "
          f"разрыв {r['cls_gap'] * 100:.1f} [{r['cls_gap_lo'] * 100:.1f}; {r['cls_gap_hi'] * 100:.1f}]", flush=True)
    print(f"  эквивариантность: ←→ E {r['E_lr']:.3e}, B {r['B_lr']:.3e}, q {r['q_lr']:.3f} [{r['q_lr_lo']:.3f}–{r['q_lr_hi']:.3f}]; "
          f"↑↓ E {r['E_ud']:.3e}, B {r['B_ud']:.3e}, q {r['q_ud']:.3f} [{r['q_ud_lo']:.3f}–{r['q_ud_hi']:.3f}]", flush=True)
    return r


def pair_stats(model, sym_arrs, device):
    cfg = model.pretrained_cfg
    mean = torch.tensor(cfg["mean"]).view(3, 1, 1)
    std = torch.tensor(cfg["std"]).view(3, 1, 1)
    model.to(device).eval()
    return hp.maps_pass(model, sym_arrs, device, mean, std, model.num_prefix_tokens)


def selftest():
    """Мера эквивариантности на искусственных сетях с известным ответом."""
    torch.manual_seed(0)
    D = 32

    class Local(torch.nn.Module):          # токен = функция только своего патча, без позиций → точно эквивариантна
        num_prefix_tokens = 1

        def __init__(self):
            super().__init__()
            self.pe = torch.nn.Conv2d(3, D, 16, 16)
            self.mlp = torch.nn.Sequential(torch.nn.Linear(D, D), torch.nn.GELU(), torch.nn.Linear(D, D))
            self.pe.weight.data = (self.pe.weight.data + self.pe.weight.data.flip(3) + self.pe.weight.data.flip(2)
                                   + self.pe.weight.data.flip(2).flip(3)) / 4      # ядро симметрично: патч и его отражение дают одно

        def forward_features(self, x):
            t = self.mlp(self.pe(x).flatten(2).transpose(1, 2))
            return torch.cat([t.mean(1, keepdim=True), t], 1)

    class Global(torch.nn.Module):         # токен = общий признак картинки (не меняется при отражении) + позиция → B = 0
        num_prefix_tokens = 1

        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3, D)
            self.pos = torch.nn.Parameter(torch.randn(196, D))

        def forward_features(self, x):
            g = self.lin(x.mean((2, 3))).unsqueeze(1)
            t = g + self.pos
            return torch.cat([g, t], 1)

    rng = np.random.default_rng(0)
    arrs = [rng.integers(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(8)]
    mean, std = torch.full((3, 1, 1), 0.5), torch.full((3, 1, 1), 0.5)
    ok = True
    for name, m, want in (("локальная (эквивариантна)", Local(), "E"), ("общая + позиция (на месте неизменна)", Global(), "B"),
                          ("случайная ViT-Tiny", tiny(100), None)):
        s = eq_summary(equivariance(m, arrs, torch.device("cpu"), mean, std, 1, batch=4))
        qt = lambda a: f"{s[f'q_{a}']:.3f}" if s[f"B_{a}"] > 1e-9 else "— (B = 0)"
        print(f"  {name}: ←→ E {s['E_lr']:.2e} B {s['B_lr']:.2e} q {qt('lr')}; ↑↓ E {s['E_ud']:.2e} B {s['B_ud']:.2e} q {qt('ud')}")
        if want == "E":
            ok &= s["E_lr"] < 1e-6 and s["E_ud"] < 1e-6 and s["B_lr"] > 1e-3
        if want == "B":
            ok &= s["B_lr"] < 1e-6 and s["B_ud"] < 1e-6 and s["E_lr"] > 1e-3
    print(f"САМОПРОВЕРКА меры эквивариантности (локальная: E = 0; общая + позиция: B = 0): {'✅' if ok else '⚠️'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", type=int, default=1, choices=(1, 2, 3, 4, 5, 6, 7))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpts", default=os.path.expanduser("~/binding-vit-data/e_full"))
    ap.add_argument("--out", default=os.path.expanduser("~/binding-vit-data/e_full/measure"))
    ap.add_argument("--limit", type=int, default=0, help="только для проверки в облаке: столько сцен")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    args.only, args.random_only, args.test_random_main = None, False, False
    import timm
    import e_train_kaggle_c67 as ek
    print(f"Опыт E (полный): меры   timm {timm.__version__}, torch {torch.__version__}")
    print(f"md5 {os.path.basename(__file__)} {hashlib.md5(open(__file__, 'rb').read()).hexdigest()}")
    ok_self = selftest()
    if args.selftest:
        return
    device = pick_device()
    print(f"устройство {device}")
    if args.limit:
        print("⚠️ ПРОВЕРОЧНЫЙ прогон: числа не для статьи.")
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
    args.out = out
    ok = {k: True for k in ("1", "3", "4", "5", "6", "8", "12")}
    ok["warn"], ok["attn"] = 0, 0.0
    rows = []

    T, dx_thr, rnd = -np.inf, 0.0, []
    for s in RAND_THR:
        m = tiny(s)
        Rm, w = pair_stats(m, sym_arrs, device)
        ok["attn"] = max(ok["attn"], w)
        Dr = Rm["Sm"].mean(0, dtype=np.float64) - Rm["Sd"].mean(0, dtype=np.float64)
        dvals = np.array([Dr[b - 1, i, j] for b, i, j in hp.within_pairs(hp.ALL_BLOCKS)])
        T = max(T, float(dvals.max()))
        mdx = float(np.abs(Rm["dx"].mean(0)).max())
        dx_thr = max(dx_thr, mdx)
        if s not in RAND:
            print(f"  случайная seed {s} (только пороги): D макс {dvals.max():+.4f}; |dx| макс {mdx:.3f}", flush=True)
            continue
        r = measure(m, f"ViT-Tiny100 случайная seed {s}", meta, args, device, colors, folds_all, ok, sym_arrs)
        rnd.append(r)
        rows.append({**{k: r.get(k, "") for k in FIELDS}, "point": "random", "epoch": "", "seed": s, "max_abs_dx": mdx})
    print(f"\nПороги по {len(RAND_THR)} случайным ViT-Tiny ({len(RAND_THR) * 36} пар): T = {T:+.4f}, |dx| = {dx_thr:.4f}")

    n_out = ek.N_OUT.get(args.cond, NCLS)
    m0, m0t = tiny(args.seed, n_out), ek.make_model(args.seed, n_out)
    ok["init"] = all(torch.equal(a, b) for a, b in zip(m0.state_dict().values(), m0t.state_dict().values()))
    pts = [("ep00", m0, 0)]
    ok["ckpt"], ok["platform"] = True, True
    for f in sorted(glob.glob(os.path.join(args.ckpts, f"c{args.cond}_s{args.seed}", "ep*.pt"))):
        m, e, good, amp, gpu = from_ckpt(f, args.cond, args.seed, ek.CMC100)
        plat = amp is True and "T4" in str(gpu)
        ok["ckpt"] &= good
        ok["platform"] &= plat
        print(f"контрольная точка {os.path.basename(f)}: эпоха {e}, условие/зерно/классы/эпоха {'✅' if good else '⚠️ НЕ СОВПАДАЮТ'}, "
              f"float16 {'да' if amp else 'нет'}, GPU {gpu}{'' if plat else ' ⚠️ не платформа регистрации (T4, float16)'}, "
              f"md5 {hashlib.md5(open(f, 'rb').read()).hexdigest()}", flush=True)
        pts.append((os.path.basename(f)[:-3], m, e))
    assert len(pts) > 1, "нет контрольных точек"
    got = sorted(e for _, _, e in pts[1:])
    ok["epochs"] = got == list(ek.CKPT_EPOCHS)
    print(f"эпохи контрольных точек: {got}" + ("  ✅ все" if ok["epochs"] else f"  ⚠️ нужны {list(ek.CKPT_EPOCHS)}"), flush=True)
    fps_rand = [r["fp"] for r in rnd]
    last = None
    for name, m, e in pts:
        r = measure(m, f"ViT-Tiny100 c{args.cond} s{args.seed} {name}", meta, args, device, colors, folds_all, ok, sym_arrs)
        Rm, w = pair_stats(m, sym_arrs, device)
        ok["attn"] = max(ok["attn"], w)
        st = hp.summarize(Rm)
        st["dx_thr"] = dx_thr
        found = hp.g3_pairs(st, T)
        mdx = float(np.abs(st["dx"]).max())
        rows.append({**{k: r.get(k, "") for k in FIELDS}, "point": name, "epoch": e, "seed": args.seed,
                     "n_pairs": len(found), "T": T, "dx_thr": dx_thr, "max_abs_dx": mdx})
        print(f"  {name}: зеркальных пар (G3) {len(found)} из 36; |dx| макс {mdx:.3f}", flush=True)
        if e > 0:
            ok.setdefault("7", True)
            ok["7"] &= all(abs(r["fp"] - f) > 1e-3 for f in fps_rand)
        last = (r, e)

    r, e = last
    print(f"\nПОСЛЕДНЯЯ ТОЧКА — эпоха {e}" + ("" if e == 40 else "  ⚠️ обучение не закончено (нужна эпоха 40)"))
    print(f"  отношение {r['ratio']:.3f} [{r['ratio_lo']:.3f}–{r['ratio_hi']:.3f}]; d(←→) {r['d_lr']:.3e}, d(↑↓) {r['d_ud']:.3e}; "
          f"у случайных: " + ", ".join(f"{x['ratio']:.3f} ({x['d_lr']:.2e} / {x['d_ud']:.2e})" for x in rnd))
    print(f"  дальний фон: «левый» {r['left'] * 100:.2f}, «верхний» {r['upper'] * 100:.2f}, разрыв {r['gap'] * 100:.2f} "
          f"[{r['gap_lo'] * 100:.2f}; {r['gap_hi'] * 100:.2f}]; CLS: «левый» {r['cls_left'] * 100:.2f}, «верхний» {r['cls_upper'] * 100:.2f}, "
          f"разрыв {r['cls_gap'] * 100:.2f}; случайные «верхний» (фон): " + ", ".join(f"{x['upper'] * 100:.2f}" for x in rnd))

    path = os.path.join(os.getcwd(), f"e_measure_full_c67_c{args.cond}_s{args.seed}{'_test' if args.limit else ''}.csv")
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        wr.writeheader()
        for row in rows:
            wr.writerow({k: row.get(k, "") for k in FIELDS})
    print()
    for k, t in (("1", "L00 ровно 50,0%"), ("3", "прогон = forward_features"), ("4", "пустая картинка не меняется при отражениях"),
                 ("5", "симм. проба: пары в сумме 1"), ("6", "контроль «близнец-вопрос» > 90%"), ("8", "сетка / токены / блоки"),
                 ("12", "считыватели из кэша = run_net")):
        print(f"Проверка {k} ({t}): {'✅' if ok[k] else '⚠️ проверь весь вывод'}")
    print(f"Проверка 7 (обученные точки отличаются от случайных сетей): {'✅' if ok.get('7', False) else '⚠️ проверь весь вывод'}")
    print(f"Проверка 10 (эпоха 0 = начальные веса e_train_kaggle.py): {'✅' if ok['init'] else '⚠️ проверь весь вывод'}")
    print(f"Проверка 11 (расстояния выше точности кэша; сетей ниже порога {D_FLOOR:.0e}: {ok.get('floor', 0)}): "
          f"{'✅' if ok.get('floor', 0) == 0 else '⚠️ отношение этих сетей ненадёжно'}")
    print(f"Проверка 9 (строки внимания в сумме 1, отклонение {ok['attn']:.1e}): {'✅' if ok['attn'] < 1e-4 else '⚠️ проверь весь вывод'}")
    ok13 = ok["ckpt"] and ok["epochs"] and ok["platform"]
    print(f"Проверка 13 (контрольные точки: условие, зерно, классы CMC, эпоха = имя; все эпохи {list(ek.CKPT_EPOCHS)}; "
          f"float16 на T4): {'✅' if ok13 else '⚠️ проверь весь вывод'}")
    print(f"Проверка 14 (самопроверка меры эквивариантности): {'✅' if ok_self else '⚠️'}")
    print(f"Предупреждений «не сошлось»: {ok['warn']}  {'✅' if ok['warn'] == 0 else '⚠️ проверь весь вывод'}")
    print(f"Числа: {path}")


if __name__ == "__main__":
    main()

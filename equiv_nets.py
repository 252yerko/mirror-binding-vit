"""
equiv_nets.py — опыт «Эквивариантность» на 24 готовых ViT опыта B (OSF xyedr) + 7 случайных.
Добавлено 19.11 (реестр 293; дизайн МОИ_ЗАМЕТКИ/ПЛАН_ОПЫТА_ЭКВ.md; ставка Ерназара — до кода, реестр 293).

ВОПРОС:
  Зеркальное отношение d(←→)/d(↑↓) опыта B считается по СРЕДНЕМУ по патчам, а среднее не меняется при перестановке
  токенов. Малое отношение может значить (а) «стирание на месте» — после отражения ←→ токен почти тот же, что у исходной
  на ТОМ ЖЕ месте; или (б) «перестановка» — токен почти тот же, что у исходной на ЗЕРКАЛЬНОМ месте (сеть переставляет
  токены вместе с картинкой; сторона хранится в том, где какой токен, а среднее её не видит).

МЕРЫ (631 сцена только из круга / квадрата / креста, картинка S — те же сцены и картинки, что у отношения в опыте B):
  токены — выход блока L (БЕЗ финальной нормировки, как у отношения опыта B: many_nets берёт run_layers()[последний]),
  только патчи (CLS и регистры исключены), сетка G × G (G = 14 при патче 16, 16 при патче 14);
  B(ось) = среднее по патчам cos-расстояние( h(отражённой)[p], h(исходной)[p] );
  E(ось) = то же, но с h(исходной)[зеркальное p];
  ρ_B = B(←→) / B(↑↓);  ρ_E = E(←→) / E(↑↓);  K = ln(ρ_E / ρ_B)  (K < 0 — асимметрия осей «перестановочная», K > 0 — «на месте»).
  Все отношения — отношения средних по сценам; 95% интервалы — бутстреп по сценам (1000, seed 0; сцены одни на все меры).
  Справка: q = E / B по каждой оси; профиль по слоям (каждый 4-й блок); те же меры после финальной нормировки;
  зеркальное отношение (повтор опыта B);
  ПРОВЕРКА УСТОЙЧИВОСТИ «с линейной картой» (справка, задана до запуска): E и B выше сравнивают ОДНИ И ТЕ ЖЕ каналы, поэтому
  не видят эквивариантность с перестановкой каналов (зеркальные «близнецы» признаков, как в CNN у Bökman & Kahl 2023;
  линейные карты отражения — Lenc & Vedaldi 2015). Поэтому то же с обученной линейной картой (гребневая регрессия со
  свободным членом, λ = 1e-3 · след(XᵀX) / D, 2 фолда по сценам, seed 0): цель — токен отражённой картинки на месте p,
  вход — токен исходной на зеркальном месте (E_lin) или на том же месте (B_lin); последний слой до нормировки; 24 места
  на сцену (случайные, seed 1000 + номер сцены, одни для всех сетей с той же сеткой); cos-расстояние предсказания и цели
  на проверочном фолде; ρ_B_lin, ρ_E_lin, K_lin — как выше. Порог для K_lin — тоже по случайным сетям.
  Изменение против плана v1 (раздел 2 «после финальной нормировки»): главные меры — до нормировки, чтобы K и Q3 относились
  к тем же токенам, что и отношение опыта B; мера после нормировки печатается справкой. Сделано до запуска (реестр 293).

ПРАВИЛА (записаны до запуска, реестр 293; Q1 исправлено после облачной проверки кода — см. ниже):
  Порог — по 7 случайным сетям (как пороги опыта B): Kmin, Kmax = наименьший и наибольший K случайных сетей.
  Сеть: «перестановка», если верхняя граница 95% интервала K < Kmin; «на месте», если нижняя > Kmax; иначе «не различается».
  Q1: «подтверждено», если «перестановка» у ≥ 20 из 24 обученных сетей (P(≥ 20 из 24 | p = 0,5) = 0,0008); иначе «не подтверждено».
      Почему не «K < 0»: в облачной проверке кода (первые 300 сцен, из них 135 симметричных; те же 7 случайных сетей, seed 0)
      у случайных сетей K = +0,02…+0,13, интервалы по сценам 6 из 7 — выше 0: набор сцен сам несимметричен по осям
      (центры фигур: x 120 ± 49, y 108 ± 52 px), и ρ_B случайных ≈ 0,93. Ноль — неверная точка отсчёта; точка отсчёта —
      случайные сети на тех же сценах. Обученные сети в облаке не считались (весов нет).
      Справка: K − (средний K случайных) у каждой сети; средний K случайных с 95% интервалом (t, 6 степеней свободы).
  Q2: ρ Спирмена(K, R) по сетям, где R определено в many_nets.csv (R — опыт B: меньше R — больше маскировки).
      «Связь есть», если ρ > 0 и односторонний перестановочный p < 0,05 (10 000 перестановок, seed 0); иначе «не показано».
  Q3 (справка, ставка не засчитывается — связь отношения с E отчасти заложена арифметикой среднего):
      ρ(отношение, ρ_E) − ρ(отношение, ρ_B) по 24 сетям, 95% интервал — бутстреп по сетям (1000, seed 0).
  Ставки Ерназара (реестр 293, 19.11 13:46): 293a Q1 «подтверждено»; 293b «перестановок» 19–23 из 24 («около 21»);
      293c у случайных K ≈ 0 — СНЯТА до запуска (K
      случайных сетей были видны в облачной проверке на 135 сценах — ставку честно засчитать нельзя); 293d Q2 «связь есть»; 293e ρ(K, R) от +0,20 до +0,60
      («≈ +0,4», округление 0,01); Q3 — справка («ρ_E лучше», не засчитывается).

ВСТРОЕННЫЕ ПРОВЕРКИ (все должны пройти):
  1. Самопроверка меры (--selftest, и в начале каждого прогона): (а) на картах токенов: «эквивариантная» карта → E = 0,
     «неизменная» → B = 0, при G = 14 и G = 16; (б) через картинки: сеть «только свой патч, симметричное ядро» при патче
     16 и 14 → E = 0 (отражение картинки ↔ отражение сетки в той же индексации, что у ViT).
  2. Наш пошаговый прогон + norm = forward_features самой модели (макс. разница < 1e-3).
  3. Пустая серая картинка: отражения не меняют токены (< 1e-4).
  4. Сетка, служебные токены, число блоков — как ожидается (проверка 8 опыта B).
  5. Отпечаток весов = отпечатку в many_nets.csv (та же сеть, что в опыте B; у случайных — та же случайная, seed 0).
  6. Отношение = many_nets.csv: по округлённым до float16 средним (как кэш опыта B) — до 2e-3; по точным — до 0,01.
  7. Нет пропусков (NaN) в мерах; E, B > 0.
  8. Случайные сети отличаются от обученных (отпечаток).

ЗАПУСК (из папки binding-vit; три части, по одной команде; потом сводка):
  caffeinate -i python -u equiv_nets.py --part 1 2>&1 | tee equiv_nets_part1_log.txt
  caffeinate -i python -u equiv_nets.py --part 2 2>&1 | tee equiv_nets_part2_log.txt
  caffeinate -i python -u equiv_nets.py --part 3 2>&1 | tee equiv_nets_part3_log.txt
  python equiv_nets.py --summary 2>&1 | tee equiv_nets_summary_log.txt
  -> equiv_nets.csv (дописывается по сетям); кэш по сценам — ~/binding-vit-data/equiv_nets/.
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from scipy.stats import spearmanr, t as tdist

import flip_test as ft
import many_nets as mn
from extract import run_layers, fingerprint, pick_device

DATA = mn.DATA
OUT_CSV = "equiv_nets.csv"
B_CSV = "many_nets.csv"
N_BOOT = 1000
FIELDS = ["name", "family", "size", "P", "seed", "fp", "ratio", "ratio_lo", "ratio_hi", "ratio16", "ratio_csv",
          "E_lr", "B_lr", "E_ud", "B_ud", "q_lr", "q_ud", "rhoB", "rhoB_lo", "rhoB_hi", "rhoE", "rhoE_lo", "rhoE_hi",
          "K", "K_lo", "K_hi", "cls", "K_post", "K_post_lo", "K_post_hi",
          "E_lr_lin", "B_lr_lin", "E_ud_lin", "B_ud_lin", "rhoE_lin", "rhoB_lin", "K_lin", "K_lin_lo", "K_lin_hi", "profile", "n_scenes", "blocks", "n_pre"]


# ---------- мера ----------
def cos_dist_t(a, b):
    return 1.0 - torch.nn.functional.cosine_similarity(a.double(), b.double(), dim=-1)


def eq_terms(t0, th, tv):
    """t*: [n, строка, столбец, D] (исходная, отражённая ←→, отражённая ↑↓). Возвращает по сценам E_lr, B_lr, E_ud, B_ud.
    ←→ меняет столбцы (ось 2), ↑↓ — строки (ось 1)."""
    return {"E_lr": cos_dist_t(th, t0.flip(2)).mean((1, 2)), "B_lr": cos_dist_t(th, t0).mean((1, 2)),
            "E_ud": cos_dist_t(tv, t0.flip(1)).mean((1, 2)), "B_ud": cos_dist_t(tv, t0).mean((1, 2))}


def k_stats(m, rng=None):
    """m: словарь массивов по сценам. ρ_B, ρ_E, K (отношения средних) с бутстреп-интервалами (одни сцены на все меры)."""
    rng = np.random.default_rng(0) if rng is None else rng
    n = len(m["E_lr"])
    idx = rng.integers(0, n, size=(N_BOOT, n))
    def f(ix):
        g = {k: v[ix].mean(-1) for k, v in m.items()}
        rB = g["B_lr"] / g["B_ud"]
        rE = g["E_lr"] / g["E_ud"]
        return rB, rE, np.log(rE / rB)
    p = f(np.arange(n))
    b = f(idx)
    ci = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return {"rhoB": (float(p[0]),) + ci(b[0]), "rhoE": (float(p[1]),) + ci(b[1]), "K": (float(p[2]),) + ci(b[2])}


N_POS = 24
LAM = 1e-3


def mirror_pos(pos, G):
    r, c = pos // G, pos % G
    return r * G + (G - 1 - c), (G - 1 - r) * G + c          # зеркальное место при ←→ и при ↑↓


def lin_terms(S0, SMh, SMv, SH, SV, seed=0):
    """S*: [сцены, места, D]. Возвращает по сценам E_lr, B_lr, E_ud, B_ud с обученной линейной картой (2 фолда по сценам)."""
    n = S0.shape[0]
    fold = np.random.default_rng(seed).permutation(n) % 2
    out = {k: np.zeros(n) for k in ("E_lr", "B_lr", "E_ud", "B_ud")}
    def fit_pred(Xtr, Ytr, Xte):
        Xtr = np.concatenate([Xtr, np.ones((len(Xtr), 1))], 1)
        Xte = np.concatenate([Xte, np.ones((len(Xte), 1))], 1)
        A = Xtr.T @ Xtr
        lam = LAM * np.trace(A) / A.shape[0]
        W = np.linalg.solve(A + lam * np.eye(A.shape[0]), Xtr.T @ Ytr)
        return Xte @ W
    def cosd(a, b):
        return 1.0 - (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)
    for key, X, Y in (("E_lr", SMh, SH), ("B_lr", S0, SH), ("E_ud", SMv, SV), ("B_ud", S0, SV)):
        for f in (0, 1):
            tr, te = fold != f, fold == f
            D = X.shape[-1]
            pred = fit_pred(X[tr].reshape(-1, D).astype(np.float64), Y[tr].reshape(-1, D).astype(np.float64),
                            X[te].reshape(-1, D).astype(np.float64))
            out[key][te] = cosd(pred, Y[te].reshape(-1, D).astype(np.float64)).reshape(int(te.sum()), -1).mean(1)
    return out


def classify(lo, hi, kmin, kmax):
    return "перестановка" if hi < kmin else ("на месте" if lo > kmax else "не различается")


# ---------- проверка 1 ----------
def selftest():
    ok = True
    torch.manual_seed(0)
    for G in (14, 16):                                              # (а) карты токенов
        t0 = torch.randn(5, G, G, 8)
        e = eq_terms(t0, t0.flip(2), t0.flip(1))                    # эквивариантная: отражённая карта = переставленная
        i = eq_terms(t0, t0, t0)                                    # неизменная на месте
        ok &= bool(e["E_lr"].max() < 1e-12 and e["E_ud"].max() < 1e-12 and e["B_lr"].min() > 1e-3)
        ok &= bool(i["B_lr"].max() < 1e-12 and i["B_ud"].max() < 1e-12 and i["E_lr"].min() > 1e-3)
    for P in (16, 14):                                              # (б) через картинки, «только свой патч»
        G = 224 // P
        conv = torch.nn.Conv2d(3, 16, P, P)
        w = conv.weight.data
        conv.weight.data = (w + w.flip(3) + w.flip(2) + w.flip(2).flip(3)) / 4
        mlp = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.GELU(), torch.nn.Linear(16, 16))
        f = lambda x: mlp(conv(x).flatten(2).transpose(1, 2)).reshape(x.shape[0], G, G, -1)   # тот же порядок, что у ViT
        x = torch.rand(3, 3, 224, 224)
        with torch.no_grad():
            s = eq_terms(f(x), f(x.flip(3)), f(x.flip(2)))
        ok &= bool(s["E_lr"].max() < 1e-9 and s["E_ud"].max() < 1e-9 and s["B_lr"].min() > 1e-4 and s["B_ud"].min() > 1e-4)
        print(f"  самопроверка, патч {P} (сетка {G}): E ←→ {float(s['E_lr'].max()):.1e}, E ↑↓ {float(s['E_ud'].max()):.1e}, "
              f"B ←→ {float(s['B_lr'].min()):.2e}, B ↑↓ {float(s['B_ud'].min()):.2e}")
    # (в) перестановка каналов: отражённая карта = переставленная исходная с переставленными каналами
    #     → E (те же каналы) велико, E_lin ≈ 0; у «на месте» с переставленными каналами → B_lin ≈ 0
    rng = np.random.default_rng(0)
    for G in (14, 16):
        D, n = 8, 60
        t0 = rng.standard_normal((n, G * G, D))
        perm = rng.permutation(D)
        mh, mv = mirror_pos(np.arange(G * G), G)
        th, tv = t0[:, mh][..., perm], t0[:, mv][..., perm]
        pos = np.stack([np.random.default_rng(1000 + i).choice(G * G, N_POS, replace=False) for i in range(n)])
        ix = np.arange(n)[:, None]
        mph, mpv = mirror_pos(pos, G)
        li = lin_terms(t0[ix, pos], t0[ix, mph], t0[ix, mpv], th[ix, pos], tv[ix, pos])
        raw = eq_terms(torch.tensor(t0).reshape(n, G, G, D), torch.tensor(th).reshape(n, G, G, D), torch.tensor(tv).reshape(n, G, G, D))
        good = li["E_lr"].max() < 1e-3 and li["E_ud"].max() < 1e-3 and li["B_lr"].min() > 0.3 and float(raw["E_lr"].min()) > 0.3
        th2, tv2 = t0[..., perm], t0[..., perm]                       # «на месте» с переставленными каналами
        li2 = lin_terms(t0[ix, pos], t0[ix, mph], t0[ix, mpv], th2[ix, pos], tv2[ix, pos])
        good &= li2["B_lr"].max() < 1e-3 and li2["B_ud"].max() < 1e-3 and li2["E_lr"].min() > 0.3
        ok &= bool(good)
        print(f"  самопроверка линейной карты, сетка {G}: перестановка каналов → E {float(raw['E_lr'].mean()):.2f}, "
              f"E_lin {li['E_lr'].max():.1e}; «на месте» → B_lin {li2['B_lr'].max():.1e}  {'✅' if good else '⚠️'}")
    print(f"Проверка 1 (самопроверка меры: эквивариантная → E = 0, неизменная → B = 0; сетки 14 и 16; линейная карта): {'✅' if ok else '⚠️'}")
    return ok


# ---------- прогон одной сети ----------
@torch.no_grad()
def passes(model, recs, device, n_pre, nb, mean, std, G, batch):
    layers = list(range(4, nb + 1, 4))
    out = {L: {k: [] for k in ("E_lr", "B_lr", "E_ud", "B_ud")} for L in layers + ["post"]}
    pm = [[], [], []]
    samp = [[] for _ in range(5)]            # S0, SMh, SMv, SH, SV
    t_start = time.time()
    for s in range(0, len(recs), batch):
        part = recs[s:s + batch]
        xs = []
        for r in part:
            x = mn.img(f"{DATA}/images/{r['id']}_S.png", mean, std)
            xs += [x, x.flip(2), x.flip(1)]                         # исходная, ←→, ↑↓ (как в many_nets: «o», «h», «v»)
        outs = run_layers(model, torch.stack(xs).to(device))
        n = len(part)
        for L in layers + ["post"]:
            h = outs[nb] if L == "post" else outs[L]
            if L == "post":
                h = model.norm(h)
            tok = h[:, n_pre:].float().cpu().reshape(n, 3, G, G, -1)
            e = eq_terms(tok[:, 0], tok[:, 1], tok[:, 2])
            for k in e:
                out[L][k].append(e[k].numpy())
            if L == nb:
                p = h[:, n_pre:].float().cpu().numpy().mean(1).reshape(n, 3, -1)
                for v in range(3):
                    pm[v].append(p[:, v])
                tk = tok.numpy().reshape(n, 3, G * G, -1)
                pos = np.stack([np.random.default_rng(1000 + s + b).choice(G * G, N_POS, replace=False) for b in range(n)])
                mph, mpv = mirror_pos(pos, G)
                ix = np.arange(n)[:, None]
                for j, (v, pp) in enumerate(((0, pos), (0, mph), (0, mpv), (1, pos), (2, pos))):
                    samp[j].append(tk[ix, v, pp])
        if (s // batch) % 20 == 0:
            print(f"    сцен {min(s + batch, len(recs))}/{len(recs)}  ({time.time() - t_start:.0f} с)", flush=True)
    res = {L: {k: np.concatenate(v) for k, v in d.items()} for L, d in out.items()}
    res["lin"] = lin_terms(*[np.concatenate(x) for x in samp])
    return res, [np.concatenate(p) for p in pm], layers


def run_net(name, timm_name, family, size, P, seed, recs, args, device, ok, bcsv):
    t0 = time.time()
    model, weights = mn.load(timm_name, seed)
    tag = name if seed is None else f"{name} [случайная]"
    fp = fingerprint(model)
    print(f"\n### {tag}: {weights}   отпечаток весов {fp:.4f}", flush=True)
    ok4, G, n_pre, nb, mean, std = mn.setup(model, P)
    ok["4"] &= ok4
    model.to(device)
    x2 = torch.stack([mn.img(f"{DATA}/images/{recs[0]['id']}_{v}.png", mean, std) for v in ("S", "Sp")]).to(device)
    ref = model.forward_features(x2)
    mine = model.norm(run_layers(model, x2)[nb])
    d2 = float((mine - ref).abs().max())
    ok["2"] &= d2 < 1e-3
    blank = x2[:1].clone()
    blank[:] = blank[0, :, :1, :1]
    fb = [run_layers(model, b) for b in (blank, blank.flip(3), blank.flip(2))]
    z = max(float((fb[0][l] - fb[k][l]).abs().max()) for l in (0, nb) for k in (1, 2))
    ok["3"] &= z < 1e-4
    print(f"  Проверка 2: макс. разница {d2:.2e} {'✅' if d2 < 1e-3 else '⚠️'};  Проверка 3: {z:.2e} {'✅' if z < 1e-4 else '⚠️'}")
    key_seed = "" if seed is None else str(seed)
    brow = bcsv.get((name, key_seed))
    if brow is not None:
        okfp = abs(float(brow["fp"]) - fp) < 1e-3
        ok["5"] &= okfp
        print(f"  Проверка 5: отпечаток {fp:.4f} / в many_nets.csv {float(brow['fp']):.4f}  {'✅' if okfp else '⚠️'}")
    elif not args.limit:
        ok["5"] = False
        print("  Проверка 5: ⚠️ сети нет в many_nets.csv")

    safe = name.replace("/", "_").replace(" ", "_")
    cache = os.path.join(args.out, f"{safe}{'' if seed is None else '_rand' + str(seed)}{'_test' + str(args.limit) if args.limit else ''}.npz")
    if os.path.exists(cache):
        zz = np.load(cache, allow_pickle=True)
        res, pm, layers = zz["res"].item(), list(zz["pm"]), list(zz["layers"])
        print(f"  взято из {cache}")
    else:
        res, pm, layers = passes(model, recs, device, n_pre, nb, mean, std, G, args.batch if size != "L" else max(1, args.batch // 2))
        np.savez(cache, res=np.array(res, dtype=object), pm=np.stack(pm), layers=np.array(layers))
    del model
    if device.type == "mps":
        torch.mps.empty_cache()

    # отношение (проверка 6)
    dh, dv = ft.cos_dist(pm[0], pm[1]), ft.cos_dist(pm[0], pm[2])
    ratio = ft.boot_ratio(dh, dv, np.random.default_rng(0))
    p16 = [p.astype(np.float16).astype(np.float32) for p in pm]
    ratio16 = float(ft.cos_dist(p16[0], p16[1]).mean() / ft.cos_dist(p16[0], p16[2]).mean())
    if brow is not None and not args.limit:
        rc = float(brow["ratio"])
        ok6 = abs(ratio16 - rc) <= 2e-3 and abs(ratio[0] - rc) <= 0.01
        ok["6"] &= ok6
        print(f"  Проверка 6: отношение {ratio[0]:.4f} (float16: {ratio16:.4f}) / many_nets.csv {rc:.4f}  {'✅' if ok6 else '⚠️'}")
    else:
        rc = float("nan")
    m = res[nb]
    ok["7"] &= all(np.isfinite(v).all() and (v > 0).all() for v in m.values())
    ks = k_stats(m)
    kp = k_stats(res["post"])
    kl = k_stats(res["lin"])
    ok["7"] &= all(np.isfinite(v).all() and (v > 0).all() for v in res["lin"].values())
    prof = {}
    for L in layers:
        prof[L] = k_stats(res[L])["K"][0]
    cls = "в сводке"          # класс — по порогу случайных сетей, в --summary
    r = dict(name=name, family=family, size=size, P=P, seed=key_seed, fp=fp, ratio=ratio[0], ratio_lo=ratio[1], ratio_hi=ratio[2],
             ratio16=ratio16, ratio_csv=rc, E_lr=float(m["E_lr"].mean()), B_lr=float(m["B_lr"].mean()),
             E_ud=float(m["E_ud"].mean()), B_ud=float(m["B_ud"].mean()),
             q_lr=float(m["E_lr"].mean() / m["B_lr"].mean()), q_ud=float(m["E_ud"].mean() / m["B_ud"].mean()),
             rhoB=ks["rhoB"][0], rhoB_lo=ks["rhoB"][1], rhoB_hi=ks["rhoB"][2], rhoE=ks["rhoE"][0], rhoE_lo=ks["rhoE"][1],
             rhoE_hi=ks["rhoE"][2], K=ks["K"][0], K_lo=ks["K"][1], K_hi=ks["K"][2], cls=cls,
             K_post=kp["K"][0], K_post_lo=kp["K"][1], K_post_hi=kp["K"][2],
             E_lr_lin=float(res["lin"]["E_lr"].mean()), B_lr_lin=float(res["lin"]["B_lr"].mean()),
             E_ud_lin=float(res["lin"]["E_ud"].mean()), B_ud_lin=float(res["lin"]["B_ud"].mean()),
             rhoE_lin=kl["rhoE"][0], rhoB_lin=kl["rhoB"][0], K_lin=kl["K"][0], K_lin_lo=kl["K"][1], K_lin_hi=kl["K"][2],
             profile=";".join(f"L{L}:{prof[L]:+.3f}" for L in layers), n_scenes=len(m["E_lr"]), blocks=nb, n_pre=n_pre)
    print(f"  ←→ E {r['E_lr']:.4f} B {r['B_lr']:.4f} (q {r['q_lr']:.3f}); ↑↓ E {r['E_ud']:.4f} B {r['B_ud']:.4f} (q {r['q_ud']:.3f})\n"
          f"  ρ_B {ks['rhoB'][0]:.3f} [{ks['rhoB'][1]:.3f}–{ks['rhoB'][2]:.3f}]; ρ_E {ks['rhoE'][0]:.3f} [{ks['rhoE'][1]:.3f}–{ks['rhoE'][2]:.3f}]; "
          f"K {ks['K'][0]:+.3f} [{ks['K'][1]:+.3f}; {ks['K'][2]:+.3f}]; после нормировки K {kp['K'][0]:+.3f} "
          f"[{kp['K'][1]:+.3f}; {kp['K'][2]:+.3f}]; с линейной картой: ←→ E {r['E_lr_lin']:.3f} B {r['B_lr_lin']:.3f}, "
          f"↑↓ E {r['E_ud_lin']:.3f} B {r['B_ud_lin']:.3f}, K_lin {kl['K'][0]:+.3f} [{kl['K'][1]:+.3f}; {kl['K'][2]:+.3f}]; по слоям {r['profile']}   ({time.time() - t0:.0f} с)", flush=True)
    return r


def read_bcsv(path):
    if not os.path.exists(path):
        return {}
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    return {(r["name"], r["seed"]): r for r in rows}       # если сеть считали дважды — последняя строка (как в many_nets)


def run_part(args):
    meta = [json.loads(line) for line in open(f"{DATA}/meta.jsonl")]
    if args.limit:
        meta = meta[:args.limit]
    recs = [r for r in meta if all(o["shape"] in ft.SYM_SHAPES for o in r["objects"])]
    os.makedirs(args.out, exist_ok=True)
    device = pick_device()
    import timm
    print(f"Опыт «Эквивариантность»: часть {args.part}, сцен {len(recs)}   timm {timm.__version__}, torch {torch.__version__}, {device}")
    if args.limit:
        print("⚠️ ПРОВЕРОЧНЫЙ прогон (облако): числа не для статьи.")
    ok = {k: True for k in ("2", "3", "4", "5", "6", "7")}
    ok["1"] = selftest()
    bcsv = read_bcsv(B_CSV)
    rows, fps = [], {}
    sel = [n for n in mn.NETS if n[5] == args.part and (not args.only or n[0] in args.only)]
    for name, tn, fam, size, P, _ in sel:
        if not args.random_only:
            r = run_net(name, tn, fam, size, P, 99 if args.test_random_main else None, recs, args, device, ok, bcsv)
            rows.append(r)
            fps[name] = r["fp"]
        if name in mn.RANDOM:
            rr = run_net(name, tn, fam, size, P, 0, recs, args, device, ok, bcsv)
            rows.append(rr)
            if name in fps:
                ok.setdefault("8", True)
                ok["8"] &= abs(rr["fp"] - fps[name]) > 1e-3
    out = OUT_CSV if not args.limit else f"equiv_nets_test{args.limit}.csv"
    exists = os.path.exists(out)
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    names = {"1": "самопроверка меры", "2": "прогон = forward_features", "3": "пустая картинка", "4": "сетка / токены / блоки",
             "5": "отпечаток = опыт B", "6": "отношение = опыт B", "7": "нет пропусков", "8": "случайные отличаются от обученных"}
    print()
    for k in sorted(ok):
        print(f"Проверка {k} ({names[k]}): {'✅' if ok[k] else '⚠️ проверь весь вывод'}")
    if args.limit:
        print("(проверки 5 и 6 в облаке не применимы: другие веса и сцены)")
    print(f"\nЧисла дописаны в {out}")


# ---------- сводка ----------
def summary(args):
    f = OUT_CSV if not args.limit else f"equiv_nets_test{args.limit}.csv"
    allrows = list({(r["name"], r["seed"]): r for r in csv.DictReader(open(f, encoding="utf-8"))}.values())
    main_seed = "" if not args.test_random_main else "99"
    names_ok = {n[0] for n in mn.NETS}
    rows = [r for r in allrows if r["seed"] == main_seed and r["name"] in names_ok]
    rnd = [r for r in allrows if r["seed"] == "0"]
    print(f"Обученных сетей в сводке: {len(rows)} из {len(mn.NETS)}; случайных: {len(rnd)} из {len(mn.RANDOM)}")
    if len(rnd) != len(mn.RANDOM) or (len(rows) != len(mn.NETS) and not args.limit):
        print("⚠️ не все сети посчитаны — правила не применяются")
        return
    kr = np.array([float(r["K"]) for r in rnd])
    kmin, kmax, k0 = float(kr.min()), float(kr.max()), float(kr.mean())
    for r in rnd:
        print(f"  [случайная] {r['name']:18s} K {float(r['K']):+.3f} [{float(r['K_lo']):+.3f}; {float(r['K_hi']):+.3f}]")
    print(f"  порог: Kmin = {kmin:+.3f}, Kmax = {kmax:+.3f}; средний K случайных {k0:+.3f}")
    for r in rows:
        r["cls"] = classify(float(r["K_lo"]), float(r["K_hi"]), kmin, kmax)
    for r in sorted(rows, key=lambda r: float(r["K"])):
        print(f"  {r['name']:18s} отношение {float(r['ratio']):.3f}  ρ_B {float(r['rhoB']):.3f}  ρ_E {float(r['rhoE']):.3f}  "
              f"K {float(r['K']):+.3f} [{float(r['K_lo']):+.3f}; {float(r['K_hi']):+.3f}]  K − K0 {float(r['K']) - k0:+.3f}  «{r['cls']}»")
    n_perm = sum(r["cls"] == "перестановка" for r in rows)
    n_place = sum(r["cls"] == "на месте" for r in rows)
    if len(kr) >= 2:
        h = tdist.ppf(0.975, len(kr) - 1) * kr.std(ddof=1) / np.sqrt(len(kr))
        rlo, rhi = float(kr.mean() - h), float(kr.mean() + h)
    else:
        rlo = rhi = float("nan")
    v1 = "подтверждено" if n_perm >= 20 else "не подтверждено"
    K = np.array([float(r["K"]) for r in rows])
    okR = [i for i, r in enumerate(rows) if r["name"] in RB and np.isfinite(RB[r["name"]])]
    if len(okR) >= 4:
        rho2, p2 = mn.perm_p(K[okR], np.array([RB[rows[i]["name"]] for i in okR]), +1)
        v2 = "связь есть" if rho2 > 0 and p2 < 0.05 else "не показано"
    else:
        rho2, p2, v2 = float("nan"), float("nan"), "не применимо"
    ratio = np.array([float(r["ratio"]) for r in rows])
    rE = np.array([float(r["rhoE"]) for r in rows])
    rB = np.array([float(r["rhoB"]) for r in rows])
    d0 = spearmanr(ratio, rE).statistic - spearmanr(ratio, rB).statistic
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(N_BOOT):
        ix = rng.integers(0, len(rows), len(rows))
        a, b = spearmanr(ratio[ix], rE[ix]).statistic, spearmanr(ratio[ix], rB[ix]).statistic
        if np.isfinite(a) and np.isfinite(b):
            bs.append(a - b)
    q3 = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) if len(bs) >= 100 else (float("nan"), float("nan"))
    klr = np.array([float(r["K_lin"]) for r in rnd])
    lmin, lmax = float(klr.min()), float(klr.max())
    n_perm_lin = sum(float(r["K_lin_hi"]) < lmin for r in rows)
    n_place_lin = sum(float(r["K_lin_lo"]) > lmax for r in rows)
    agree = sum((float(r["K_lin_hi"]) < lmin) == (r["cls"] == "перестановка") for r in rows)
    print(f"\nСправка (проверка устойчивости, линейная карта): порог по случайным K_lin [{lmin:+.3f}; {lmax:+.3f}]; "
          f"«перестановка» у {n_perm_lin}, «на месте» у {n_place_lin} из {len(rows)}; совпадает с главной мерой у {agree} из {len(rows)}")
    print(f"\nВЫВОД ПО ПРАВИЛАМ (реестр 293, записано до запуска):")
    print(f"  Q1: «перестановка» у {n_perm} из {len(rows)} («на месте» {n_place}, «не различается» {len(rows) - n_perm - n_place}); "
          f"порог Kmin {kmin:+.3f} -> «{v1}»; справка: средний K случайных {k0:+.3f} [{rlo:+.3f}; {rhi:+.3f}]")
    print(f"  Q2: ρ(K, R) = {rho2:+.3f}, p = {p2:.4f} (n = {len(okR)}) -> «{v2}»")
    print(f"  Q3 (справка): ρ(отношение, ρ_E) = {spearmanr(ratio, rE).statistic:+.3f}, ρ(отношение, ρ_B) = {spearmanr(ratio, rB).statistic:+.3f}; "
          f"разность {d0:+.3f} [{q3[0]:+.3f}; {q3[1]:+.3f}]")
    if not args.limit:
        hit = lambda b: "ПОПАЛ ✅" if b else "НЕ ПОПАЛ ❌"
        print("\nСТАВКИ ЕРНАЗАРА (реестр 293, 19.11 13:46):")
        print(f"  293a Q1 «подтверждено» -> «{v1}»  {hit(v1 == 'подтверждено')}")
        print(f"  293b «перестановок» 19–23 из 24 -> {n_perm}  {hit(19 <= n_perm <= 23)}")
        print(f"  293c у случайных K ≈ 0 — снята до запуска (справка: средний K случайных {k0:+.3f} [{rlo:+.3f}; {rhi:+.3f}])")
        print(f"  293d Q2 «связь есть» -> «{v2}»  {hit(v2 == 'связь есть')}")
        print(f"  293e ρ(K, R) от +0,20 до +0,60 -> {rho2:+.2f}  {hit(0.20 <= round(rho2, 2) <= 0.60)}")
        print(f"  Q3 «ρ_E лучше» — справка, не засчитывается: разность {d0:+.3f} [{q3[0]:+.3f}; {q3[1]:+.3f}]")
    with open("equiv_nets_rules.csv" if not args.limit else f"equiv_nets_rules_test{args.limit}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rule", "value", "lo", "hi", "n", "verdict"])
        w.writerow(["Q1_n_perm", n_perm, "", "", len(rows), v1])
        w.writerow(["Q1_random_K", k0, kmin, kmax, len(kr), "порог"])
        w.writerow(["Q2_rho", rho2, p2, "", len(okR), v2])
        w.writerow(["Q3_diff", d0, q3[0], q3[1], len(rows), "справка"])
        w.writerow(["lin_n_perm", n_perm_lin, lmin, lmax, len(rows), f"совпадает с главной у {agree}"])


RB = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, choices=[1, 2, 3])
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=os.path.expanduser("~/binding-vit-data/equiv_nets"))
    ap.add_argument("--batch", type=int, default=8, help="сцен за раз (у L — вдвое меньше)")
    ap.add_argument("--limit", type=int, default=0, help="только для проверки в облаке")
    ap.add_argument("--test-random-main", action="store_true", help="только для проверки в облаке: вместо обученной — случайная seed 99")
    ap.add_argument("--only", nargs="*", default=None, help="только для проверки: имена сетей")
    ap.add_argument("--random-only", action="store_true", help="только для проверки")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if args.summary:
        for r in read_bcsv(B_CSV).values():
            if r["seed"] == "" and r["R"] not in ("", "nan"):
                RB[r["name"]] = float(r["R"])
        summary(args)
    else:
        assert args.part, "нужно --part 1|2|3, --summary или --selftest"
        run_part(args)


if __name__ == "__main__":
    main()

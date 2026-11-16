"""
heads_pair_plus.py — опыт G+: «ломаем симметрию во всех зеркальных парах». Добавлено 16.11 (реестр 214–215, ставка Ерназара — до кода).

ВОПРОС (после опыта G, osf.io/8nxg7):
  В DINO пара голов 4.4 / 4.5 зеркальна, и выключение одной головы поднимает «левый», а обеих — нет (гашение).
  Объясняют ли такие пары слепоту к стороне шире: (H1) гасят ли зеркальные пары DeiT III, найденные зарегистрированной
  мерой G3, но выключением не проверенные; (H2) возвращает ли «левый» нарушение симметрии во ВСЕХ парах сразу
  (по одной голове из каждой пары) сильнее, чем случайные наборы голов, и даёт ли выключение обеих голов всех пар ≈ 0.

СЕТИ: DeiT III B/16 (главная), DINO B/16 (вторая пара 3.3 / 3.4). Веса — как в опыте G.
ПАРЫ (выход опыта G, heads_pair_pairs.csv: пары G3, непересекающиеся, жадно по D):
  DeiT III: 3.0/3.2, 6.6/6.11, 4.3/4.9, 5.0/5.3, 5.6/5.7, 7.7/7.9 (пара 5.3/5.6 выпадает — головы заняты); DINO: 3.3/3.4.
  «Правые» (dx > 0): 3.0, 6.6, 4.9, 5.0, 5.6, 7.7; «левые» (dx < 0): 3.2, 6.11, 4.3, 5.3, 5.7, 7.9.
ДАННЫЕ: 2000 новых сцен data_g (как G2). ВЫКЛЮЧЕНИЕ: resample, как в G2 (выход головы до общей проекции, все токены,
  заменяется её выходом на сцене-доноре; один донор на все выключенные головы), ОДНА перестановка доноров (seed 0 — та же,
  что перестановка 0 в G2); интервалы — бутстрэп по сценам (у каждой сцены свой донор). Пробы как во всех опытах:
  дальний фон, L12, логрег C = 0,1, 5 фолдов по сценам, обучаются заново. Δ = изменение точности против обычной сети.
  N(a, b) = Δлевый(a) + Δлевый(b) − Δлевый(a + b) — неаддитивность, по сценам.

ПРАВИЛА (записаны ДО запуска):
  H1 (DeiT III): пара «гасит», если нижняя граница интервала N с поправкой Бонферрони на 6 пар (99,17%, бутстрэп 10 000,
      seed 0) > 0. k = число гасящих пар из 6. Контроль: 6 случайных пар голов (по одной в блоке каждой пары, вне голов
      зеркальных пар G3, seed 0), то же правило -> k_контр. «Гашение обобщается на DeiT III», если k ≥ 2 и k > k_контр;
      иначе «не показано».
  H2а (DeiT III): для каждого из двух наборов «правые» (6 голов) и «левые» (6 голов): (i) нижняя граница 95% интервала
      Δлевый > 0; (ii) Δлевый больше, чем у каждого из 50 случайных наборов по 6 голов (то же число голов в каждом блоке,
      вне голов пар; seed 0; p = 1/51); (iii) нижняя граница 95% интервала (Δлевый − Δверхний) > 0.
      «Нарушение симметрии возвращает „левый“», если (i)–(iii) выполнены для ОБОИХ наборов; иначе «не показано».
  H2б (DeiT III): все 12 голов пар. «≈ 0», если 95% интервал Δлевый включает 0; иначе «не ≈ 0» (со знаком).
      Справка: место среди 50 случайных наборов по 12 голов; сравнение с половиной меньшего из Δлевый(правые), Δлевый(левые).
  DINO: пара 3.3 / 3.4 «гасит», если нижняя граница 95% интервала N > 0; иначе «не показано».
  Пометка «разрушение», если «верхний» < 60% в любом из основных условий.
  Ставки Ерназара (реестр 215, 16.11 02:17, до кода): 215a k от 2 до 4; 215b Δлевый H2а (среднее двух наборов,
      округление 0,1) от +3,0 до +8,0 п.п.; 215c (ii) выполнено для обоих наборов; 215d H2б «≈ 0»; 215e DINO 3.3 / 3.4 «гасит».
  Справка (не правила): «верхний» и CLS для основных условий; N по «верхнему» для каждой пары; Δлевый по одной голове;
      место наборов «правые» / «левые» среди случайных.

ВСТРОЕННЫЕ ПРОВЕРКИ (все должны пройти):
  1. Прогон с явным вниманием = run_layers (относительная разница < 1e-4).
  2. Resample всех 12 голов пар с донором «сама картинка» = обычный прогон (< 1e-4); с чужим донором — отличается
     (порог относительный: выходы голов на S и Sp различаются больше чем на 1e-6 своей величины; у случайной DeiT III
     LayerScale = 1e-6, поэтому абсолютный порог на итоговом выходе был бы неверен — урок проверки 1 в gdnvb).
  3. Правила H1, H2а, H2б на искусственных числах с известным ответом.
  4. Сетка, служебные токены, блоки; нормировка внутри внимания — пустая.
  5. Новые сцены, DeiT III: L00 — пробы ровно 50,0%; «близнец-вопрос» > 90%; обычная сеть ± 5 п.п. от data_v2 (57,83 / 75,00);
     DINO: обычный прогон из кэша опыта G = 64,00 / 81,77 (± 0,005).
  6. Пары и стороны совпадают с выходом опыта G (heads_pair_pairs.csv, heads_pair_heads.csv).
  7. Случайные наборы и контрольные пары: различны, нужное число голов в каждом блоке, без голов зеркальных пар.
  8. Предупреждений «обучение не сошлось» — 0.

ЗАПУСК (из папки binding-vit; кэш — ~/binding-vit-data/heads/, повторный запуск продолжает с места):
  caffeinate -i python heads_pair_plus.py 2>&1 | tee heads_pair_plus_log.txt      # ~4–5 ч
  -> heads_pair_plus.csv, heads_pair_plus_rules.csv
"""
import argparse
import csv
import hashlib
import os
import time

import numpy as np
import torch

import heads as H
import heads_pair as HP
import many_nets as mn
import mirror_decomp as md
from extract import fingerprint, pick_device

PAIRS_DEIT = [((3, 0), (3, 2)), ((6, 6), (6, 11)), ((4, 3), (4, 9)), ((5, 0), (5, 3)), ((5, 6), (5, 7)), ((7, 7), (7, 9))]
RIGHT = [(3, 0), (6, 6), (4, 9), (5, 0), (5, 6), (7, 7)]
LEFT = [(3, 2), (6, 11), (4, 3), (5, 3), (5, 7), (7, 9)]
PAIR_DINO = ((3, 3), (3, 4))
N_SETS = 50
V2_DEIT = (57.825, 75.00)
G2_DINO = (64.00, 81.77)
mark = lambda b: "✅" if b else "⚠️ проверь весь вывод"
hs = HP.hs
hl = lambda heads: " ".join(hs(h) for h in heads)


def boot_ci(x, alpha=0.05, n=1000, seed=0):
    """Среднее и (1 − alpha) интервал по бутстрэпу сцен (NaN — пропускаются)."""
    x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    b = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(x.mean()), float(np.percentile(b, 100 * alpha / 2)), float(np.percentile(b, 100 * (1 - alpha / 2)))


def h1_count(los):
    return int(sum(lo > 0 for lo in los))


def h1_verdict(k, k_ctrl):
    return "гашение обобщается на DeiT III" if (k >= 2 and k > k_ctrl) else "не показано"


def h2a_verdict(parts):
    """parts: {'правые': (c1, c2, c3), 'левые': (...)}"""
    return "нарушение симметрии возвращает «левый»" if all(all(v) for v in parts.values()) else "не показано"


def h2b_verdict(lo, hi, pt):
    return "≈ 0" if lo <= 0 <= hi else f"не ≈ 0 ({'+' if pt > 0 else '−'})"


def check_rules():
    ok = True
    ok &= h1_verdict(h1_count([0.01, 0.02, -0.01, 0.0, -0.5, 0.3]), h1_count([-0.1, 0.0, -0.2, -0.3, 0.01, -0.1])) == "гашение обобщается на DeiT III"
    ok &= h1_verdict(h1_count([0.01, -0.02, -0.01, 0.0, -0.5, -0.3]), 0) == "не показано"            # k = 1
    ok &= h1_verdict(2, 2) == "не показано"                                                          # k не больше контроля
    ok &= h2a_verdict({"правые": (True, True, True), "левые": (True, True, True)}).startswith("нарушение")
    ok &= h2a_verdict({"правые": (True, True, True), "левые": (True, False, True)}) == "не показано"
    ok &= h2b_verdict(-0.01, 0.02, 0.005) == "≈ 0" and h2b_verdict(0.01, 0.03, 0.02) == "не ≈ 0 (+)" and h2b_verdict(-0.03, -0.01, -0.02) == "не ≈ 0 (−)"
    rng = np.random.default_rng(1)
    pt, lo, hi = boot_ci(rng.normal(0.02, 0.1, 2000), alpha=0.05 / 6, n=2000)
    ok &= lo < pt < hi and abs(pt - 0.02) < 0.01
    print(f"Проверка 3 (правила H1, H2а, H2б на искусственных числах): {mark(ok)}")
    return ok


def block_counts(heads):
    c = {}
    for b, _ in heads:
        c[b] = c.get(b, 0) + 1
    return c


def random_sets(counts, exclude, n_sets, seed):
    rng = np.random.default_rng(seed)
    out, seen, tries = [], set(), 0
    while len(out) < n_sets and tries < 100000:
        tries += 1
        s = []
        for b, k in sorted(counts.items()):
            pool = [(b, h) for h in range(H.NH) if (b, h) not in exclude]
            s += [pool[i] for i in sorted(rng.choice(len(pool), k, replace=False))]
        key = tuple(sorted(s))
        if key not in seen:
            seen.add(key)
            out.append(list(key))
    return out


def control_pairs(pairs, exclude, seed):
    rng = np.random.default_rng(seed)
    taken, out = set(exclude), []
    for (b, _), _ in pairs:
        pool = [(b, h) for h in range(H.NH) if (b, h) not in taken]
        i, j = sorted(rng.choice(len(pool), 2, replace=False))
        out.append((pool[i], pool[j]))
        taken |= {pool[i], pool[j]}
    return out


def check_pairs_from_csv(test):
    """Проверка 6: пары и стороны = выход опыта G. Возвращает (ok, головы зеркальных пар G3 по сетям)."""
    fp, fh = "heads_pair_pairs.csv", "heads_pair_heads.csv"
    partners = {"DeiT III B/16": set(), "DINO B/16": set()}
    if not (os.path.exists(fp) and os.path.exists(fh)):
        print("Проверка 6: нет heads_pair_pairs.csv / heads_pair_heads.csv" + (" (облачный тест — пропуск)" if test else " ⚠️"))
        for p in PAIRS_DEIT:
            partners["DeiT III B/16"] |= set(p)
        partners["DINO B/16"] |= {(4, 4), (4, 5), (3, 1), (3, 3), (3, 4)}
        return test, partners
    P = [r for r in csv.DictReader(open(fp, encoding="utf-8"))]
    Hd = [r for r in csv.DictReader(open(fh, encoding="utf-8"))]
    ok = True
    sel_all = {}
    for net in partners:
        g = sorted([r for r in P if r["net"] == net and r["g3"] == "1"], key=lambda r: -float(r["d"]))
        dx = {(int(r["block"]), int(r["head"])): float(r["dx"]) for r in Hd if r["net"] == net and r["seed"] == ""}
        used, sel = set(), []
        for r in g:
            a, b = (int(r["block"]), int(r["a"])), (int(r["block"]), int(r["b"]))
            partners[net] |= {a, b}
            if a in used or b in used:
                continue
            used |= {a, b}
            sel.append((a, b))
        sel_all[net] = (sel, dx)
    sel, dx = sel_all["DeiT III B/16"]
    ok &= sel == PAIRS_DEIT
    ok &= all(dx[h] > 0 for h in RIGHT) and all(dx[h] < 0 for h in LEFT)
    ok &= sorted(RIGHT + LEFT) == sorted(h for p in PAIRS_DEIT for h in p)
    sel_d, _ = sel_all["DINO B/16"]
    ok &= sel_d == [((4, 4), (4, 5)), PAIR_DINO]
    print(f"Проверка 6: пары и стороны совпадают с выходом опыта G (DeiT III: {len(sel)} пар; DINO: {len(sel_d)}) {mark(ok)}")
    return ok, partners


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/binding-vit-data/heads"))
    ap.add_argument("--limit", type=int, default=0, help="только для проверки в облаке")
    ap.add_argument("--test-random-main", action="store_true", help="только для проверки в облаке: случайные веса seed 99")
    ap.add_argument("--n-sets", type=int, default=N_SETS, help="только для проверки в облаке")
    args = ap.parse_args()
    test = bool(args.limit or args.test_random_main)
    tagx = f"_test{args.limit}" if args.limit else ""
    device = pick_device()
    import timm
    print(f"Опыт G+ («ломаем симметрию во всех парах»)   timm {timm.__version__}, torch {torch.__version__}")
    if test:
        print("⚠️ ПРОВЕРОЧНЫЙ прогон (облако): числа не для статьи.")
    ok = {k: True for k in ("1", "2", "3", "4", "5", "6", "7")}
    ok["warn"] = 0
    ok["3"] = check_rules()
    ok["6"], partners = check_pairs_from_csv(test)
    HP.set_data(HP.DATA_NEW)
    print(f"data_g/meta.jsonl md5 {hashlib.md5(open(f'{HP.DATA_NEW}/meta.jsonl', 'rb').read()).hexdigest()}")
    meta, infos, lab, folds, _, _ = H.scenes(args)
    n = len(meta)
    os.makedirs(args.out, exist_ok=True)
    donors = HP.donors_for(n, 0)
    rows, t0 = [], time.time()

    def acc_cond(model, n_pre, nb, mean, std, key, heads, groups=(1,)):
        cache = os.path.join(args.out, f"plus_{key}_{'_'.join(hs(h) for h in heads)}{tagx}.npy")
        if os.path.exists(cache) and not args.test_random_main:
            F = np.load(cache).astype(np.float32)
        else:
            F = HP.feats_resample(model, meta, infos, device, mean, std, n_pre, nb, [(h, donors) for h in heads])
            if not args.test_random_main:
                np.save(cache, F.astype(np.float16))
        return H.probes(F, lab, folds, ok, groups=groups)

    def setup(key):
        name, timm_name = H.NETS[key]
        model, weights = mn.load(timm_name, 99 if args.test_random_main else None)
        print(f"\n### {name}: {weights}   отпечаток весов {fingerprint(model):.4f};  сцен {n}")
        ok4, _, n_pre, nb, mean, std = mn.setup(model, H.P)
        ok["4"] &= ok4 and all(type(b.attn.norm).__name__ == "Identity" for b in model.blocks)
        model.to(device)
        x2 = H.to_x([H.load_u8(f"{HP.DATA_NEW}/images/{meta[0]['id']}_{v}.png") for v in ("S", "Sp")], mean, std).to(device)
        d1 = HP.rel_check1(model, x2)
        ok["1"] &= d1 < 1e-4
        print(f"  Проверка 1: прогон с явным вниманием = run_layers, относительная разница {d1:.2e} {mark(d1 < 1e-4)}")
        return name, model, n_pre, nb, mean, std, x2

    # ======================= DeiT III =======================
    name, model, n_pre, nb, mean, std, x2 = setup("deit3")
    all12 = RIGHT + LEFT
    with torch.no_grad():
        ref = H.forward(model, x2)[nb]
        own = HP.head_outputs(model, x2, all12)
        mm = {}
        for (b, hh), o in own.items():
            mm.setdefault(b, {})[hh] = o
        alt = H.forward(model, x2, off=H.as_off(all12), means=mm)[nb]
        swp = H.forward(model, x2, off=H.as_off(all12), means={b: {hh: o.flip(0) for hh, o in v.items()} for b, v in mm.items()})[nb]
    d2, d2b = float((ref - alt).abs().max() / ref.abs().max()), float((ref - swp).abs().max() / ref.abs().max())
    dho = max(float((o - o.flip(0)).abs().max() / o.abs().max()) for o in own.values())    # выходы голов на S и Sp различаются
    good2 = d2 < 1e-4 and d2b > 0 and dho > 1e-6
    ok["2"] &= good2
    print(f"  Проверка 2: донор «сама картинка» — разница {d2:.2e}; чужой донор — {d2b:.2e} (> 0; выходы голов на S и Sp различаются "
          f"на {dho:.2e} от своей величины) {mark(good2)}")
    cache_c = os.path.join(args.out, f"plus_clean_deit3{tagx}.npz")
    if os.path.exists(cache_c) and not args.test_random_main:
        z = np.load(cache_c)
        F12, F00, T = z["F12"], z["F00"], z["T"]
    else:
        F12, F00, T = H.feats_pass(model, meta, infos, device, mean, std, n_pre, nb, with_l00=True)
        if not args.test_random_main:
            np.savez(cache_c, F12=F12, F00=F00, T=T)
    clean = H.probes(F12, lab, folds, ok, groups=(1, 0))
    l00 = H.probes(F00, lab, folds, ok)
    ok["5"] &= all(np.allclose(v, 0.5) for v in l00.values())
    for q in ("left", "upper"):
        tq = np.array([infos[i][1][q][0] for i in range(n)])
        c_, _ = md.probe_plain(T[np.arange(n), :, tq], lab[q][:, :, 0], lab[q][:, :, 1], folds)
        ok["5"] &= test or np.nanmean(c_) > 0.9
    cl, cu = np.nanmean(clean[(1, "left")]) * 100, np.nanmean(clean[(1, "upper")]) * 100
    ok["5"] &= test or (abs(cl - V2_DEIT[0]) <= 5 and abs(cu - V2_DEIT[1]) <= 5)
    print(f"  обычная сеть на новых сценах: «левый» {cl:.2f}% (data_v2: {V2_DEIT[0]:.2f}), «верхний» {cu:.2f}% (data_v2: {V2_DEIT[1]:.2f})")

    part = partners["DeiT III B/16"]
    ctrl = control_pairs(PAIRS_DEIT, part, 0)
    sets6 = random_sets(block_counts(RIGHT), part, args.n_sets, 0)
    sets12 = random_sets(block_counts(all12), part, args.n_sets, 1)
    ok["7"] &= block_counts(RIGHT) == block_counts(LEFT)
    ok["7"] &= len(sets6) == args.n_sets and len(sets12) == args.n_sets
    ok["7"] &= all(block_counts(s) == block_counts(RIGHT) and not (set(s) & part) for s in sets6)
    ok["7"] &= all(block_counts(s) == block_counts(all12) and not (set(s) & part) for s in sets12)
    ok["7"] &= all(not ({a, b} & part) and a[0] == b[0] for a, b in ctrl) and len({h for p in ctrl for h in p}) == 12
    print(f"  контрольные пары H1: {', '.join(f'{hs(a)}/{hs(b)}' for a, b in ctrl)};  Проверка 7 (наборы и контрольные пары) {mark(ok['7'])}")

    D = {}

    def run(key, label, heads, groups=(1,)):
        r = acc_cond(model if key == "deit3" else model_d, *(net_args if key == "deit3" else net_args_d), key, heads, groups)
        d = {g_q: r[g_q] - (clean if key == "deit3" else clean_d)[g_q] for g_q in r}
        D[(key, label)] = d
        pl = boot_ci(d[(1, "left")])
        pu = boot_ci(d[(1, "upper")])
        extra = f";  CLS Δлевый {np.nanmean(d[(0, 'left')]) * 100:+.2f}" if (0, "left") in d else ""
        print(f"  {label:34s} [{hl(heads)}]: Δлевый {pl[0] * 100:+.2f} [{pl[1] * 100:+.2f}; {pl[2] * 100:+.2f}], "
              f"Δверхний {pu[0] * 100:+.2f} [{pu[1] * 100:+.2f}; {pu[2] * 100:+.2f}]{extra}   ({time.time() - t0:.0f} с)", flush=True)
        rows.append(dict(net=key, cond=label, heads=hl(heads), dleft=pl[0], dleft_lo=pl[1], dleft_hi=pl[2], dupper=pu[0], dupper_lo=pu[1],
                         dupper_hi=pu[2], upper_acc=float(np.nanmean(r[(1, "upper")]))))
        return d

    net_args = (n_pre, nb, mean, std)
    print("\n  H1: пары по одной голове и вместе (resample, одна перестановка доноров)")
    for tag, plist in (("пара", PAIRS_DEIT), ("контроль", ctrl)):
        for a, b in plist:
            g = (1, 0) if tag == "пара" else (1,)
            run("deit3", f"{tag} {hs(a)}/{hs(b)}: {hs(a)}", [a], g)
            run("deit3", f"{tag} {hs(a)}/{hs(b)}: {hs(b)}", [b], g)
            run("deit3", f"{tag} {hs(a)}/{hs(b)}: обе", [a, b], g)
    print("\n  H2: наборы")
    for label, heads in (("правые (по одной из каждой пары)", RIGHT), ("левые (по одной из каждой пары)", LEFT), ("все 12 голов пар", all12)):
        run("deit3", label, heads, (1, 0))
    for i, s in enumerate(sets6):
        run("deit3", f"случайный набор 6 №{i:02d}", s)
    for i, s in enumerate(sets12):
        run("deit3", f"случайный набор 12 №{i:02d}", s)

    # --- правила DeiT III
    alpha6 = 0.05 / 6
    print("\n  H1: неаддитивность N = Δ(a) + Δ(b) − Δ(a + b), п.п. [интервал 99,17% — Бонферрони на 6]:")
    nres = {}
    for tag, plist in (("пара", PAIRS_DEIT), ("контроль", ctrl)):
        for a, b in plist:
            k0 = f"{tag} {hs(a)}/{hs(b)}"
            da, db, dj = (D[("deit3", f"{k0}: {x}")] for x in (hs(a), hs(b), "обе"))
            Nl = da[(1, "left")] + db[(1, "left")] - dj[(1, "left")]
            Nu = da[(1, "upper")] + db[(1, "upper")] - dj[(1, "upper")]
            pt, lo, hi = boot_ci(Nl, alpha=alpha6, n=10000)
            pu = boot_ci(Nu)
            nres[k0] = (pt, lo, hi)
            print(f"    {k0:22s}: N = {pt * 100:+.2f} [{lo * 100:+.2f}; {hi * 100:+.2f}]  (части: {np.nanmean(da[(1, 'left')]) * 100:+.2f} + "
                  f"{np.nanmean(db[(1, 'left')]) * 100:+.2f} − {np.nanmean(dj[(1, 'left')]) * 100:+.2f});  справка: N по «верхнему» {pu[0] * 100:+.2f} "
                  f"[{pu[1] * 100:+.2f}; {pu[2] * 100:+.2f}]  -> {'гасит' if lo > 0 else '—'}")
            rows.append(dict(net="deit3", cond=f"N {k0}", heads="", dleft=pt, dleft_lo=lo, dleft_hi=hi, dupper=pu[0], dupper_lo=pu[1], dupper_hi=pu[2], upper_acc=""))
    k = h1_count([nres[f"пара {hs(a)}/{hs(b)}"][1] for a, b in PAIRS_DEIT])
    k_ctrl = h1_count([nres[f"контроль {hs(a)}/{hs(b)}"][1] for a, b in ctrl])
    v1 = h1_verdict(k, k_ctrl)
    print(f"  H1: гасящих пар k = {k} из 6; у контрольных пар {k_ctrl} из 6 -> «{v1}»")

    r6 = np.array([np.nanmean(D[("deit3", f"случайный набор 6 №{i:02d}")][(1, "left")]) for i in range(len(sets6))])
    r12 = np.array([np.nanmean(D[("deit3", f"случайный набор 12 №{i:02d}")][(1, "left")]) for i in range(len(sets12))])
    parts, mains = {}, {}
    for side, label in (("правые", "правые (по одной из каждой пары)"), ("левые", "левые (по одной из каждой пары)")):
        d = D[("deit3", label)]
        pl = boot_ci(d[(1, "left")])
        pdd = boot_ci(d[(1, "left")] - d[(1, "upper")])
        c1, c2, c3 = pl[1] > 0, bool(pl[0] > r6.max()), pdd[1] > 0
        parts[side] = (c1, c2, c3)
        mains[side] = pl[0]
        print(f"  H2а, {side}: Δлевый {pl[0] * 100:+.2f} [{pl[1] * 100:+.2f}; {pl[2] * 100:+.2f}] — (i) {'да' if c1 else 'нет'}; "
              f"(ii) больше всех {len(r6)} случайных наборов по 6 (наибольший {r6.max() * 100:+.2f}; не меньше: {int((r6 >= pl[0]).sum())}) — {'да' if c2 else 'нет'}; "
              f"(iii) Δлевый − Δверхний {pdd[0] * 100:+.2f} [{pdd[1] * 100:+.2f}; {pdd[2] * 100:+.2f}] — {'да' if c3 else 'нет'}")
    v2a = h2a_verdict(parts)
    print(f"  H2а -> «{v2a}»;  справка: случайные наборы по 6 — среднее {r6.mean() * 100:+.2f}, σ {r6.std() * 100:.2f}")
    d12 = D[("deit3", "все 12 голов пар")]
    p12 = boot_ci(d12[(1, "left")])
    v2b = h2b_verdict(p12[1], p12[2], p12[0])
    half = 0.5 * min(mains.values())
    print(f"  H2б: все 12 голов пар Δлевый {p12[0] * 100:+.2f} [{p12[1] * 100:+.2f}; {p12[2] * 100:+.2f}] -> «{v2b}»;  справка: место среди {len(r12)} "
          f"случайных наборов по 12 — выше {int((r12 < p12[0]).sum())}, ниже {int((r12 > p12[0]).sum())} (среднее {r12.mean() * 100:+.2f}, σ {r12.std() * 100:.2f}); "
          f"половина меньшего из H2а {half * 100:+.2f} — {'меньше' if p12[0] < half else 'не меньше'}")
    destroyed = any(r["upper_acc"] != "" and r["upper_acc"] < 0.60 for r in rows if r["net"] == "deit3" and not r["cond"].startswith("случайный"))
    if destroyed:
        print("  ⚠️ пометка «разрушение»: «верхний» < 60% в одном из основных условий")
    del model

    # ======================= DINO: пара 3.3 / 3.4 =======================
    name_d, model_d, n_pre_d, nb_d, mean_d, std_d, _ = setup("dino")
    net_args_d = (n_pre_d, nb_d, mean_d, std_d)
    cdino = os.path.join(args.out, f"pair_clean{tagx}.npz")
    if os.path.exists(cdino) and not args.test_random_main:
        F12d = np.load(cdino)["F12"]
    else:
        F12d = H.feats_pass(model_d, meta, infos, device, mean_d, std_d, n_pre_d, nb_d)[0]
    clean_d = H.probes(F12d, lab, folds, ok)
    cld, cud = np.nanmean(clean_d[(1, "left")]) * 100, np.nanmean(clean_d[(1, "upper")]) * 100
    ok["5"] &= test or (abs(cld - G2_DINO[0]) < 0.005 and abs(cud - G2_DINO[1]) < 0.005)
    print(f"  обычная DINO (кэш опыта G): «левый» {cld:.2f}% (G2: {G2_DINO[0]:.2f}), «верхний» {cud:.2f}% (G2: {G2_DINO[1]:.2f})")
    a, b = PAIR_DINO
    for x, heads in ((hs(a), [a]), (hs(b), [b]), ("обе", [a, b])):
        run("dino", f"DINO {hs(a)}/{hs(b)}: {x}", heads, (1, 0))
    da, db, dj = (D[("dino", f"DINO {hs(a)}/{hs(b)}: {x}")] for x in (hs(a), hs(b), "обе"))
    pN = boot_ci(da[(1, "left")] + db[(1, "left")] - dj[(1, "left")])
    vd = "гасит" if pN[1] > 0 else "не показано"
    print(f"  DINO {hs(a)}/{hs(b)}: N = {pN[0] * 100:+.2f} [{pN[1] * 100:+.2f}; {pN[2] * 100:+.2f}] -> «{vd}»")
    rows.append(dict(net="dino", cond=f"N DINO {hs(a)}/{hs(b)}", heads="", dleft=pN[0], dleft_lo=pN[1], dleft_hi=pN[2], dupper="", dupper_lo="", dupper_hi="", upper_acc=""))

    # ======================= итог =======================
    sfx = "" if not test else "_test"
    with open(f"heads_pair_plus{sfx}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(f"heads_pair_plus_rules{sfx}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["rule", "verdict", "value", "extra"])
        w.writeheader()
        w.writerow(dict(rule="H1", verdict=v1, value=k, extra=f"контроль {k_ctrl}"))
        w.writerow(dict(rule="H2а", verdict=v2a, value=(mains["правые"] + mains["левые"]) / 2, extra=f"правые {mains['правые']:.5f}; левые {mains['левые']:.5f}; {parts}"))
        w.writerow(dict(rule="H2б", verdict=v2b, value=p12[0], extra=f"[{p12[1]:.5f}; {p12[2]:.5f}]"))
        w.writerow(dict(rule="DINO 3.3/3.4", verdict=vd, value=pN[0], extra=f"[{pN[1]:.5f}; {pN[2]:.5f}]"))
    if not test:
        hit = lambda c: "ПОПАЛ ✅" if c else "НЕ ПОПАЛ ❌"
        m2 = round((mains["правые"] + mains["левые"]) / 2 * 100, 1) + 0.0
        print("\nСТАВКИ ЕРНАЗАРА (реестр 215, 16.11 02:17, до кода):")
        print(f"  215a H1: k от 2 до 4 -> k = {k}  {hit(2 <= k <= 4)}")
        print(f"  215b H2а: Δлевый (среднее двух наборов) от +3,0 до +8,0 -> {m2:+.1f}  {hit(3.0 <= m2 <= 8.0)}")
        print(f"  215c H2а (ii) для обоих наборов -> {'да' if parts['правые'][1] and parts['левые'][1] else 'нет'}  {hit(parts['правые'][1] and parts['левые'][1])}")
        print(f"  215d H2б «≈ 0» -> «{v2b}»  {hit(v2b == '≈ 0')}")
        print(f"  215e DINO 3.3/3.4 «гасит» -> «{vd}»  {hit(vd == 'гасит')}")
    print()
    for kk, t in (("1", "прогон = run_layers"), ("2", "resample с донором «сама картинка»"), ("3", "правила на искусственных числах"),
                  ("4", "сетка, токены, блоки"), ("5", "новые сцены: L00, «близнец-вопрос», обычные сети"), ("6", "пары = выход опыта G"),
                  ("7", "случайные наборы и контрольные пары")):
        print(f"Проверка {kk} ({t}): {mark(ok[kk])}")
    print(f"Проверка 8 (предупреждений «не сошлось»: {ok['warn']}): {mark(ok['warn'] == 0)}")
    print(f"Всего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()

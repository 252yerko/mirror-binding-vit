"""
probe_robust.py — проверка устойчивости проб (реестр 101–102): зависят ли главные числа от силы
регуляризации пробы и нет ли в пробе «утечки».

Дизайн, правила и ставка Ерназара записаны в журнал ДО этого кода (реестр 102).

Что считаем (слой L12, считыватели: дальний фон и CLS; сети: обученные DINO и CLIP; кэш активаций acts_v2*):
  (а) Сетка C логистической регрессии: 0.001 / 0.01 / 0.1 (как в probe_v2) / 1 / 10.
      Для каждого C — точность «левый» и «верхний» и разрыв «верхний − левый» по сценам
      (те же 5 фолдов по сценам, что в probe_v2; парный бутстрэп по сценам, 10 000, seed 0).
      Правило А (дальний фон): «устойчив», если при всех 5 C нижняя граница 95% интервала
      разрыва > 0 у обеих сетей; иначе «зависит от C».
  (б) Контроль «случайная сторона» (C = 0.1): в каждой сцене монетка (seed 2026) решает, про какого
      близнеца спрашивать (левого или правого); для S и Sp — один и тот же близнец. Проба не знает,
      кого спрашивают, поэтому честный результат ≈ 50%.
      Правило Б (4 интервала: 2 сети × 2 считывателя; интервалы 98,75% — поправка на 4 сравнения,
      по замечанию Ерназара к дизайну): «утечка», если хотя бы у одного нижняя граница > 50%;
      иначе «чисто», если у всех интервал целиком внутри 45–55%; иначе «не различает».

Встроенные проверки:
  1. При C = 0.1 точности «левый»/«верхний» совпадают с probe_v2*.csv (логрег, L12) до 1e-5.
  2. «Мешок признаков» (одинаковые векторы у S и Sp) в контроле даёт ровно 50% в каждой сцене.
  3. Метки контроля: у сцен с монеткой 0 они совпадают с метками «левый», у сцен с монеткой 1 — переставлены.
  4. Счётчик предупреждений «обучение не сошлось» по каждому C.

Запуск на Маке (из папки binding-vit, ~15–30 мин):
  caffeinate -dims python probe_robust.py 2>&1 | tee probe_robust_log.txt
Проверочный прогон (облако, поддельные активации):
  python probe_robust.py --fake 120
"""
import argparse
import csv
import json
import os
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import ablate_blocks as ab          # boot_paired (10 000, seed 0); файл не меняется
import probe_v2 as pv               # scene_info, фолды, данные; файл не меняется

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
C_BASE = 0.1
LAYER = 12
READ = ["far_bg", "cls"]
NAME = {"far_bg": "дальний фон", "cls": "CLS"}
NETS = [("DINO", "acts_v2", "probe_v2.csv", "probe_v2_upper.csv"),
        ("CLIP", "acts_v2_clip", "probe_v2_clip.csv", "probe_v2_clip_upper.csv")]
ALPHA_A = 0.05
ALPHA_B = 0.05 / 4
COIN_SEED = 2026
# ставка Ерназара (реестр 102, 12.11 15:02) и засчёт, объявленный до кода
BET = dict(q1=(10.0, 15.0), q2="устойчив", q3="чисто")


# ------------------------------------------------------------------
# 1. Сцены и метки
# ------------------------------------------------------------------
def infos_for(meta, colors, query):
    pv.QUERY = query
    pv.GROUPS[3] = f"{query}_twin"
    return [pv.scene_info(rec, colors) for rec in meta]


def control_labels(infos_left, coin):
    """Монетка 0 — спрашиваем левого близнеца (метки как «левый»), 1 — правого (метки переставлены)."""
    out = []
    for (groups, lab), c in zip(infos_left, coin):
        if c == 0:
            out.append(lab)
        else:
            out.append({v: (lab[v][1], lab[v][0]) for v in ("S", "Sp")})
    return out


# ------------------------------------------------------------------
# 2. Признаки и проба
# ------------------------------------------------------------------
def features(acts, meta, infos_left, layer):
    rows = {(r["id"], r["version"]): int(r["row"]) for r in csv.DictReader(open(f"{acts}/index.csv"))}
    A = np.load(f"{acts}/L{layer:02d}.npy", mmap_mode="r")
    X = {g: [] for g in READ}
    keep = {g: [] for g in READ}
    for si, (rec, (groups, _)) in enumerate(zip(meta, infos_left)):
        for v in ("S", "Sp"):
            h = np.asarray(A[rows[(rec["id"], v)]], dtype=np.float32)
            for g in READ:
                if groups[g]:
                    X[g].append(h[groups[g]].mean(0))
                    keep[g].append((si, v))
    return {g: np.array(X[g]) for g in READ}, keep


def labels_for(keep, labs):
    yt = np.array([labs[si][v][0] for si, v in keep])
    yr = np.array([labs[si][v][1] for si, v in keep])
    sc = np.array([si for si, _ in keep])
    return yt, yr, sc


def run(X, yt, yr, sc, folds, C):
    """Как probe_v2.run_probe (логрег), но с заданным C. Возвращает балл каждой сцены и число предупреждений."""
    s = np.zeros(len(X))
    n_warn = 0
    for f in range(pv.N_FOLDS):
        test = folds[sc] == f
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            clf.fit(X[~test], yt[~test])
            n_warn += sum(issubclass(x.category, ConvergenceWarning) for x in w)
        d = clf.predict_proba(X[test])
        col = {c: i for i, c in enumerate(clf.classes_)}
        pt = d[np.arange(test.sum()), [col[c] for c in yt[test]]]
        pr = d[np.arange(test.sum()), [col[c] for c in yr[test]]]
        s[test] = np.where(pt > pr, 1.0, np.where(pt == pr, 0.5, 0.0))
    n_sc = sc.max() + 1
    cnt = np.bincount(sc, minlength=n_sc)
    present = cnt > 0
    per = (np.bincount(sc, weights=s, minlength=n_sc) / np.maximum(cnt, 1))[present]
    return per, np.where(present)[0], n_warn


# ------------------------------------------------------------------
# 3. Правила
# ------------------------------------------------------------------
def rule_a(lows):
    return "устойчив" if all(lo > 0 for lo in lows) else "зависит от C"


def rule_b(cis):
    if any(lo > 0.5 for _, lo, _ in cis):
        return "утечка"
    if all(lo >= 0.45 and hi <= 0.55 for _, lo, hi in cis):
        return "чисто"
    return "не различает"


def pct(t):
    return f"{100 * t[0]:5.1f} [{100 * t[1]:5.1f} … {100 * t[2]:5.1f}]"


def dpct(t):
    return f"{100 * t[0]:+5.1f} [{100 * t[1]:+5.1f} … {100 * t[2]:+5.1f}]"


# ------------------------------------------------------------------
# 4. Поддельные активации (только проверка в облаке)
# ------------------------------------------------------------------
def make_fake(meta, infos_left, colors, out, kind):
    """kind = 'bag': S и Sp одинаковы (мешок признаков); 'left': в токенах явно записан цвет левого близнеца."""
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(7)
    A = np.zeros((2 * len(meta), 197, 768), np.float16)
    with open(f"{out}/index.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "id", "version"])
        for i, (rec, (_, lab)) in enumerate(zip(meta, infos_left)):
            base = rng.normal(size=(197, 768)).astype(np.float32)
            for k, v in enumerate(("S", "Sp")):
                h = base.copy()
                if kind == "left":
                    h[:, lab[v][0]] += 3.0
                A[2 * i + k] = h
                w.writerow([2 * i + k, rec["id"], v])
    for l in range(13):
        np.save(f"{out}/L{l:02d}.npy", A if l == LAYER else A[:, :1])


# ------------------------------------------------------------------
# 5. main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", type=int, default=0, help="только для проверки в облаке: N сцен, поддельные активации")
    args = ap.parse_args()
    t0 = time.time()
    meta = [json.loads(l) for l in open(f"{pv.DATA}/meta.jsonl")]
    colors = sorted({o[k] for r in meta for o in r["objects"] for k in ("color", "color_Sp")})
    folds = np.random.default_rng(0).permutation(len(meta)) % pv.N_FOLDS     # как в probe_v2
    if args.fake:
        meta, folds = meta[:args.fake], folds[:args.fake]
        print(f"⚠️ ПРОВЕРОЧНЫЙ прогон: {len(meta)} сцен, поддельные активации. Цифры — не результат.")
    inf_l = infos_for(meta, colors, "left")
    inf_u = infos_for(meta, colors, "upper")
    for (gl, _), (gu, _) in zip(inf_l, inf_u):
        assert gl["far_bg"] == gu["far_bg"] and gl["cls"] == gu["cls"], "группы токенов зависят от вопроса"
    coin = np.random.default_rng(COIN_SEED).integers(0, 2, len(meta))
    lab_l = [lab for _, lab in inf_l]
    lab_u = [lab for _, lab in inf_u]
    lab_c = control_labels(inf_l, coin)
    ok3 = all((lab_c[i] == lab_l[i]) if coin[i] == 0 else
              all(lab_c[i][v] == (lab_l[i][v][1], lab_l[i][v][0]) for v in ("S", "Sp")) for i in range(len(meta)))
    print(f"Сцен: {len(meta)}; монетка: левый {int((coin == 0).sum())}, правый {int((coin == 1).sum())}")
    print(f"Проверка 3 (метки контроля): {'✅' if ok3 else '⚠️ проверь весь вывод'}")

    nets = NETS
    if args.fake:
        os.makedirs("fake_acts", exist_ok=True)
        make_fake(meta, inf_l, colors, "fake_acts/left", "left")
        make_fake(meta, inf_l, colors, "fake_acts/bag", "bag")
        nets = [("FAKE_left", "fake_acts/left", None, None), ("FAKE_bag", "fake_acts/bag", None, None)]

    rows_csv, res = [], {}
    for name, acts, ref_l, ref_u in nets:
        print(f"\n### {name}: {acts}/  слой L{LAYER:02d}", flush=True)
        X, keep = features(acts, meta, inf_l, LAYER)
        res[name] = {}
        for g in READ:
            yl, rl, sc = labels_for(keep[g], lab_l)
            yu, ru, _ = labels_for(keep[g], lab_u)
            yc, rc, _ = labels_for(keep[g], lab_c)
            res[name][g] = {}
            for C in C_GRID:
                t1 = time.time()
                pl, idx, wl = run(X[g], yl, rl, sc, folds, C)
                pu, idx2, wu = run(X[g], yu, ru, sc, folds, C)
                assert np.array_equal(idx, idx2)
                tl, tu, tg = ab.boot_paired([pl, pu, pu - pl], ALPHA_A)
                res[name][g][C] = dict(left=tl, upper=tu, gap=tg, warn=wl + wu, n=len(pl))
                print(f"  {NAME[g]:11s} C={C:<6g} левый {pct(tl)}  верхний {pct(tu)}  разрыв {dpct(tg)}  "
                      f"сцен {len(pl)}  несошедшихся {wl + wu}  ({time.time() - t1:.0f} с)", flush=True)
                for k, t in (("left", tl), ("upper", tu), ("gap_upper_minus_left", tg)):
                    rows_csv.append([name, g, f"{C:g}", k, f"{t[0]:.5f}", f"{t[1]:.5f}", f"{t[2]:.5f}", len(pl)])
            pc, _, wc = run(X[g], yc, rc, sc, folds, C_BASE)
            (tc,) = ab.boot_paired([pc], ALPHA_B)
            res[name][g]["ctrl"] = tc
            print(f"  {NAME[g]:11s} КОНТРОЛЬ «случайная сторона» (C=0.1, 98,75%): {pct(tc)}  несошедшихся {wc}", flush=True)
            rows_csv.append([name, g, "0.1", "control_random_side", f"{tc[0]:.5f}", f"{tc[1]:.5f}", f"{tc[2]:.5f}", len(pc)])
            # проверка 2: «мешок» — одинаковые векторы у S и Sp -> ровно 50% в контроле
            Xb = X[g].copy()
            pos = {(si, v): k for k, (si, v) in enumerate(keep[g])}
            for (si, v), k in pos.items():
                if v == "Sp" and (si, "S") in pos:
                    Xb[k] = X[g][pos[(si, "S")]]
            pb, _, _ = run(Xb, yc, rc, sc, folds, C_BASE)
            print(f"  Проверка 2 («мешок» в контроле -> ровно 50% в каждой сцене): "
                  f"{'✅' if np.all(np.abs(pb - 0.5) < 1e-12) else '⚠️ проверь весь вывод'}")
            # проверка 1: при C = 0.1 — те же числа, что в probe_v2*.csv
            if ref_l and os.path.exists(ref_l) and os.path.exists(ref_u):
                ref = {}
                for q, fn in (("left", ref_l), ("upper", ref_u)):
                    for r in csv.DictReader(open(fn)):
                        if r["probe"] == "logreg" and r["group"] == g and r["layer"] == f"L{LAYER:02d}" and r["acts"] == acts:
                            ref[q] = float(r["acc"])
                d = max(abs(res[name][g][C_BASE][q][0] - ref[q]) for q in ("left", "upper"))
                print(f"  Проверка 1 (C=0.1 = probe_v2: левый {100 * ref['left']:.3f} / верхний {100 * ref['upper']:.3f}): "
                      f"макс. разница {100 * d:.5f} п.п.  {'✅' if d < 1e-5 else '⚠️ проверь весь вывод'}")
            elif ref_l:
                print(f"  Проверка 1: ⚠️ нет {ref_l} / {ref_u} — сверить вручную")

    # ---------------- итог
    print("\n" + "=" * 78)
    print(f"РЕЗУЛЬТАТ (L{LAYER:02d}; 95% интервал для разрыва, 98,75% для контроля; бутстрэп по сценам)")
    print("=" * 78)
    names = [n for n, *_ in nets]
    lows = []
    for n in names:
        print(f"{n}: разрыв «верхний − левый», дальний фон, по C: " +
              "  ".join(f"{C:g}: {100 * res[n]['far_bg'][C]['gap'][0]:+.1f} [{100 * res[n]['far_bg'][C]['gap'][1]:+.1f}]"
                        for C in C_GRID))
        lows += [res[n]["far_bg"][C]["gap"][1] for C in C_GRID]
    va = rule_a(lows)
    print(f"Правило А (все {len(lows)} нижних границ разрыва > 0): «{va}»")
    cis = [res[n][g]["ctrl"] for n in names for g in READ]
    vb = rule_b(cis)
    print("Контроль: " + "   ".join(f"{n} {NAME[g]} {pct(res[n][g]['ctrl'])}" for n in names for g in READ))
    print(f"Правило Б: «{vb}»")
    print("Справка (НЕ правило): разрыв по CLS: " + "   ".join(
        f"{n} " + " / ".join(f"{100 * res[n]['cls'][C]['gap'][0]:+.1f}" for C in C_GRID) for n in names))

    if not args.fake:
        print("\nСтавка Ерназара (реестр 102, 12.11 15:02; засчёт объявлен до кода):")
        hits = 0
        m = min(res["DINO"]["far_bg"][C]["gap"][0] for C in C_GRID)
        x = round(100 * m, 1)
        lo, hi = BET["q1"]
        ok = lo <= x <= hi; hits += ok
        print(f"  102a наименьший разрыв DINO (дальний фон) по 5 C: [{lo}; {hi}] -> {x} {'✅' if ok else '❌'}")
        ok = va == BET["q2"]; hits += ok
        print(f"  102b правило А: «{BET['q2']}» -> «{va}» {'✅' if ok else '❌'}")
        ok = vb == BET["q3"]; hits += ok
        print(f"  102c правило Б: «{BET['q3']}» -> «{vb}» {'✅' if ok else '❌'}")
        print(f"  попаданий: {hits} из 3")

    stem = "probe_robust" + (f"_fake{args.fake}" if args.fake else "")
    with open(f"{stem}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["net", "readout", "C", "measure", "value", "ci_lo", "ci_hi", "n_scenes"])
        w.writerows(rows_csv)
    print(f"\nФайлы: {stem}.csv.  Всего: {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()

"""
heads_pair_explore.py — РАЗВЕДОЧНЫЙ анализ ПОСЛЕ запуска опыта G (osf.io/8nxg7). НЕ правило регистрации.

Вопрос: в G2 правило специфичности дало «не показано» из-за одной пары: 4.5 + контроль 4.7 = +0,77 < +0,96.
Но 4.7 по одной сама снижает «левый» (−0,98). Правило «≥ половины одиночного» не учитывало собственный эффект
контрольной головы. Здесь — мера, которая его учитывает: неаддитивность
    N(a, b) = Δлевый(a) + Δлевый(b) − Δлевый(a + b)
(по сценам; Δлевый — среднее по трём перестановкам доноров, как в G2; 95% интервал — бутстрэп по сценам, 1000, seed 0).
Гашение пары ⇔ N(4.4, 4.5) ≫ 0; для пар с контрольными головами ожидается N ≈ 0 (эффекты просто складываются).
Также: разность N(4.4, 4.5) − N(пара с контролем) с интервалом (бутстрэп по сценам, одни и те же сцены).

Сеть не запускается: признаки берутся из кэша части ablate (~/binding-vit-data/heads/pair_*.npy, pair_clean.npz),
пробы обучаются заново тем же кодом (mirror_decomp.probe_plain, те же фолды).
Проверка: Δлевый, пересчитанные здесь, совпадают с heads_pair_g2.csv (до 1e-9).

Запуск (из папки binding-vit): python heads_pair_explore.py 2>&1 | tee heads_pair_explore_log.txt   (~5–10 мин)
"""
import argparse
import csv
import os

import numpy as np

import heads as H
import heads_pair as HP
import mirror_decomp as md

PAIRS = [((4, 4), (4, 5))] + [(a, c) for a in HP.PAIR for c in HP.CTRL]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/binding-vit-data/heads"))
    ap.add_argument("--limit", type=int, default=0, help="только для проверки в облаке")
    args = ap.parse_args()
    tagx = f"_test{args.limit}" if args.limit else ""
    print("Разведочный анализ ПОСЛЕ запуска опыта G (osf.io/8nxg7) — не правило регистрации.")
    HP.set_data(HP.DATA_NEW)
    meta, infos, lab, folds, _, _ = H.scenes(args)
    ok = {"warn": 0}
    z = np.load(os.path.join(args.out, f"pair_clean{tagx}.npz"))
    clean = md.probe_plain(z["F12"][:, :, 1], lab["left"][:, :, 0], lab["left"][:, :, 1], folds)
    ok["warn"] += clean[1]
    clean = clean[0]

    def dleft(spec):
        """spec: [(голова, 'A')] -> Δлевый по сценам (среднее по 3 перестановкам)."""
        slug = "_".join(f"{HP.hs(bh)}{k}" for bh, k in spec)
        accs = []
        for p in HP.PERMS:
            F = np.load(os.path.join(args.out, f"pair_rs_{slug}_p{p}{tagx}.npy")).astype(np.float32)
            a, nw = md.probe_plain(F[:, :, 1], lab["left"][:, :, 0], lab["left"][:, :, 1], folds)
            ok["warn"] += nw
            accs.append(a)
        return np.mean(accs, axis=0) - clean

    single = {h: dleft([(h, "A")]) for h in HP.PAIR + HP.CTRL}
    joint = {(a, b): dleft([(a, "A"), (b, "A")]) for a, b in PAIRS}

    # проверка: совпадение с heads_pair_g2.csv
    good = True
    g2 = "heads_pair_g2.csv" if not args.limit else None
    if g2 and os.path.exists(g2):
        ref = {r["cond"]: float(r["far_dleft"]) for r in csv.DictReader(open(g2, encoding="utf-8")) if r["kind"] == "resample"}
        mine = {(f"одна {HP.hs(h)}" if h in HP.PAIR else f"контроль {HP.hs(h)}"): float(np.nanmean(v)) for h, v in single.items()}
        mine["пара, один донор"] = float(np.nanmean(joint[PAIRS[0]]))
        mine.update({f"{HP.hs(a)} + контроль {HP.hs(c)}": float(np.nanmean(joint[(a, c)])) for a, c in PAIRS[1:]})
        diffs = {k: abs(mine[k] - ref[k]) for k in mine}
        good = max(diffs.values()) < 1e-9
        print(f"Проверка: Δлевый совпадают с heads_pair_g2.csv, наибольшая разница {max(diffs.values()):.1e}  {'✅' if good else '⚠️'}")

    rng = lambda: np.random.default_rng(0)
    N = {pr: single[pr[0]] + single[pr[1]] - joint[pr] for pr in PAIRS}
    rows = []
    print("\nНеаддитивность N(a, b) = Δлевый(a) + Δлевый(b) − Δлевый(a + b), п.п. [95% интервал]:")
    for pr in PAIRS:
        pt, lo, hi, _ = md.boot(N[pr], rng())
        da, db, dj = (float(np.nanmean(x)) * 100 for x in (single[pr[0]], single[pr[1]], joint[pr]))
        line = f"  {HP.hs(pr[0])} + {HP.hs(pr[1])}: N = {pt * 100:+.2f} [{lo * 100:+.2f}; {hi * 100:+.2f}]   (части: {da:+.2f} + {db:+.2f} − {dj:+.2f})"
        row = dict(pair=f"{HP.hs(pr[0])}+{HP.hs(pr[1])}", N=pt, N_lo=lo, N_hi=hi, d_a=da / 100, d_b=db / 100, d_ab=dj / 100)
        if pr != PAIRS[0]:
            dd = N[PAIRS[0]] - N[pr]
            p2, l2, h2, _ = md.boot(dd, rng())
            line += f";  N(4.4 + 4.5) − N = {p2 * 100:+.2f} [{l2 * 100:+.2f}; {h2 * 100:+.2f}]"
            row.update(diff=p2, diff_lo=l2, diff_hi=h2)
        rows.append(row)
        print(line)
    ctrlN = [r for r in rows[1:]]
    print(f"\n  контрольные пары: N от {min(r['N'] for r in ctrlN) * 100:+.2f} до {max(r['N'] for r in ctrlN) * 100:+.2f}; "
          f"интервал включает 0 у {sum(r['N_lo'] <= 0 <= r['N_hi'] for r in ctrlN)} из 6; "
          f"разность с парой 4.4 + 4.5 > 0 (нижняя граница) у {sum(r['diff_lo'] > 0 for r in ctrlN)} из 6")
    print(f"Проверка (предупреждений «не сошлось»: {ok['warn']}): {'✅' if ok['warn'] == 0 else '⚠️'}")
    out = "heads_pair_explore.csv" if not args.limit else "heads_pair_explore_test.csv"
    keys = ["pair", "N", "N_lo", "N_hi", "d_a", "d_b", "d_ab", "diff", "diff_lo", "diff_hi"]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"Числа записаны в {out}")


if __name__ == "__main__":
    main()

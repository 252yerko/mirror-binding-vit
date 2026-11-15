"""
heads_stripes.py — разведочный прогон ПОСЛЕ опыта F (реестр 193). НЕ правило и не часть регистрации gdnvb.

Вопрос: в F3 (1000 сцен) головы DINO 4.4, 4.5 и 2.3 («полосы вдоль строки») по одной поднимали «левый» на
+3,65, +3,55 и +3,20 п.п., а в F2 в составе «нижних 12» голов эффекта не было (−0,12). Почему?
Условия (все 2000 сцен; выключение средним, как в heads.py; дальний фон, L12; пробы обучаются заново):
  по одной: 4.4, 4.5, 2.3;  попарно: три пары;  все три вместе;
  «остальные 9»: нижние 12 из F2 без этих трёх (тянут ли они «левый» обратно?);
  контроль шума: 6 голов тех же блоков (2 из блока 2, 4 из блока 4) с |m| < 1/197, по одной (seed 0);
  справка: нижние 12 целиком — из кэша F2.
Ставки Ерназара (20:24, до кода): 193a — все три вместе: Δлевый от +2,0 до +4,0 п.п.;
  193b — среднее Δлевый трёх голов по одной: от +1,5 до +2,5 п.п. (округление 0,1).
Проверки: обычная сеть = опыт F («левый» 64,75%, «верхний» 83,50%); нижние 12 из кэша дают Δлевый −0,12;
  предупреждений «не сошлось» — 0.
Запуск (из папки binding-vit): caffeinate -i python heads_stripes.py 2>&1 | tee heads_stripes_log.txt   (~20 мин)
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

import heads as H
import many_nets as mn
import mirror_decomp as md
from extract import pick_device

STRIPES = [(4, 4), (4, 5), (2, 3)]
BOTTOM12 = [(2, 3), (3, 0), (3, 1), (4, 4), (4, 5), (5, 1), (5, 7), (5, 11), (6, 1), (6, 2), (6, 9), (9, 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/binding-vit-data/heads"))
    ap.add_argument("--limit", type=int, default=0, help="только для проверки в облаке")
    ap.add_argument("--test-random-main", action="store_true", help="только для проверки в облаке")
    args = ap.parse_args()
    test = bool(args.limit or args.test_random_main)
    tagx = f"_test{args.limit}" if args.limit else ""
    name, timm_name = H.NETS["dino"]
    device = pick_device()
    ok = {"warn": 0}
    meta, infos, lab, folds, sym, sym_arrs = H.scenes(args)
    z = np.load(os.path.join(args.out, f"clean_dino{tagx}.npz"), allow_pickle=True)
    F12 = z["F12"]
    means = {b: torch.from_numpy(z["means"][b - 1]) for b in range(1, H.NB + 1)}
    m_pt = np.load(os.path.join(args.out, f"m_dino{tagx}.npz"))["M"].mean(0)
    model, weights = mn.load(timm_name, 99 if args.test_random_main else None)
    _, _, n_pre, nb, mean, std = mn.setup(model, H.P)
    model.to(device)
    print(f"Разведочный прогон «полосы DINO вместе»: {name} ({weights}); сцен {len(meta)}")
    clean = H.probes(F12, lab, folds, ok)
    cl, cu = clean[(1, "left")].mean(), clean[(1, "upper")].mean()
    ok_clean = test or (abs(cl * 100 - 64.75) < 0.005 and abs(cu * 100 - 83.50) < 0.005)
    print(f"  обычная сеть: «левый» {cl * 100:.2f}%, «верхний» {cu * 100:.2f}%  {'✅' if ok_clean else '⚠️ проверь весь вывод'}")

    # контроль шума: несильные головы тех же блоков
    rng = np.random.default_rng(0)
    u = 1.0 / 197
    ctrl = []
    for b, n in ((2, 2), (4, 4)):
        pool = [h for h in range(H.NH) if (test or abs(m_pt[b - 1, h]) < u) and (b, h) not in STRIPES]
        ctrl += [(b, int(h)) for h in sorted(rng.choice(pool, n, replace=False))]
    others9 = [bh for bh in BOTTOM12 if bh not in STRIPES]
    conds = ([(f"одна {b}.{h}", [(b, h)]) for b, h in STRIPES] +
             [(f"пара {a[0]}.{a[1]} + {c[0]}.{c[1]}", [a, c]) for i, a in enumerate(STRIPES) for c in STRIPES[i + 1:]] +
             [("все три", list(STRIPES)), ("остальные 9 из нижних 12", others9)] +
             [(f"контроль {b}.{h}", [(b, h)]) for b, h in ctrl])
    rows, t0 = [], time.time()

    def add(cname, heads, F):
        res = H.probes(F, lab, folds, ok)
        row = dict(cond=cname, heads=H.hs_str(heads), n_heads=len(heads))
        for gi, g in ((1, "far"), (0, "cls")):
            dl = res[(gi, "left")] - clean[(gi, "left")]
            du = res[(gi, "upper")] - clean[(gi, "upper")]
            bl, bu, bd = (md.boot(a, np.random.default_rng(0)) for a in (dl, du, dl - du))
            row.update({f"{g}_dleft": bl[0], f"{g}_dleft_lo": bl[1], f"{g}_dleft_hi": bl[2], f"{g}_dupper": bu[0], f"{g}_dupper_lo": bu[1],
                        f"{g}_dupper_hi": bu[2], f"{g}_dd": bd[0], f"{g}_dd_lo": bd[1], f"{g}_dd_hi": bd[2]})
        rows.append(row)
        print(f"  {cname:26s} [{H.hs_str(heads)}]: Δлевый {row['far_dleft'] * 100:+.2f} [{row['far_dleft_lo'] * 100:+.2f}; {row['far_dleft_hi'] * 100:+.2f}], "
              f"Δверхний {row['far_dupper'] * 100:+.2f} [{row['far_dupper_lo'] * 100:+.2f}; {row['far_dupper_hi'] * 100:+.2f}], "
              f"Δл−Δв {row['far_dd'] * 100:+.2f} [{row['far_dd_lo'] * 100:+.2f}; {row['far_dd_hi'] * 100:+.2f}];  CLS Δлевый {row['cls_dleft'] * 100:+.2f}, "
              f"CLS Δверхний {row['cls_dupper'] * 100:+.2f}   ({time.time() - t0:.0f} с)", flush=True)
        return row

    for cname, heads in conds:
        cache = os.path.join(args.out, f"stripes_{H.hs_str(heads).replace(' ', '_')}{tagx}.npy")
        if os.path.exists(cache):
            F = np.load(cache)
        else:
            F = H.feats_pass(model, meta, infos, device, mean, std, n_pre, nb, off=H.as_off(heads), means=means)[0]
            np.save(cache, F)
        add(cname, heads, F)
    ok_b12 = True
    cb = os.path.join(args.out, f"dino_bottom12{tagx}.npz")
    if os.path.exists(cb):
        r = add("нижние 12 (кэш F2)", BOTTOM12, np.load(cb, allow_pickle=True)["F"])
        ok_b12 = test or abs(r["far_dleft"] * 100 - (-0.12)) < 0.01
    by = {r["cond"]: r for r in rows}
    singles = [by[f"одна {b}.{h}"]["far_dleft"] * 100 for b, h in STRIPES]
    ctrls = [r["far_dleft"] * 100 for r in rows if r["cond"].startswith("контроль")]
    tri = by["все три"]["far_dleft"] * 100
    print(f"\n  по одной: {', '.join(f'{x:+.2f}' for x in singles)} (среднее {np.mean(singles):+.2f}, сумма {np.sum(singles):+.2f});  все три вместе: {tri:+.2f}")
    print(f"  контроль шума (6 несильных голов по одной): {', '.join(f'{x:+.2f}' for x in ctrls)} (среднее {np.mean(ctrls):+.2f}, σ {np.std(ctrls):.2f})")
    if not test:
        hit = lambda b: "ПОПАЛ ✅" if b else "НЕ ПОПАЛ ❌"
        print("\nСТАВКИ ЕРНАЗАРА (реестр 193, 15.11 20:24, до кода):")
        print(f"  193a все три вместе: Δлевый от +2,0 до +4,0 -> {round(tri, 1) + 0.0:+.1f}  {hit(2.0 <= round(tri, 1) <= 4.0)}")
        ms = round(float(np.mean(singles)), 1)
        print(f"  193b среднее по одной: от +1,5 до +2,5 -> {ms + 0.0:+.1f}  {hit(1.5 <= ms <= 2.5)}")
    print(f"\nПроверка (обычная сеть = опыт F): {'✅' if ok_clean else '⚠️ проверь весь вывод'}")
    print(f"Проверка (нижние 12 из кэша: Δлевый −0,12): {'✅' if ok_b12 else '⚠️ проверь весь вывод'}")
    print(f"Проверка (предупреждений «не сошлось»: {ok['warn']}): {'✅' if ok['warn'] == 0 else '⚠️ проверь весь вывод'}")
    out = "heads_stripes.csv" if not test else "heads_stripes_test.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Числа записаны в {out}")


if __name__ == "__main__":
    main()

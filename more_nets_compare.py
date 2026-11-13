"""
more_nets_compare.py — правило 3 опыта «ещё две сети» (реестр 114–115, записано ДО запуска).

6 сетей: DINO, CLIP (ViT-B/16), MAE, I-JEPA (ViT-H/14) — числа из прежних файлов;
DINOv2 ViT-B/14 с регистрами и DeiT III ViT-B/16 — из more_nets_dinov2.csv / more_nets_deit3.csv.
  x = зеркальное отношение d(←→)/d(↑↓), среднее по патчам, последний слой (L12 у ViT-B, L32 у ViT-H);
  y = разрыв «верхний − левый», дальний фон, последний слой, логрег (п.п.).
ρ Спирмена (x, y). «Связь следует», если ρ ≤ −0,83; иначе «не подтверждена». При 6 сетях это значит: не больше двух
перестановок соседей в порядке (ρ = −1, −0,943 или −0,886); без всякой связи такой исход случаен с вероятностью
12/720 ≈ 1,7% (точный перебор). Уточнено 13.11 до регистрации: в дизайне было «порог значимости для 6 точек» — порог не менялся. Оговорка: 4 точки из 6 известны заранее (ρ по ним −0,8).
Ставка Ерназара (реестр 114): «не подтверждена», уверенность ~75%.

Встроенная проверка: числа 4 прежних сетей, прочитанные из файлов, равны записанным в журнале (реестр 13, 25, 36–38, 41–42).

ЗАПУСК (из папки binding-vit, после обоих прогонов more_nets.py):
  python more_nets_compare.py 2>&1 | tee more_nets_compare_log.txt
"""

import csv

import numpy as np
from scipy.stats import rankdata, spearmanr

THRESH = -0.83
OLD_EXPECTED = {"DINO": (0.284, 18.6), "CLIP": (0.613, 9.0), "MAE": (0.626, 10.1), "I-JEPA": (0.403, 11.4)}


def rows(f):
    return list(csv.DictReader(open(f, encoding="utf-8")))


def mirror_ratio(f, trained_word, layer):
    for r in rows(f):
        if r["group"] == "mean" and r["layer"] == layer and trained_word in r["weights"]:
            return float(r["ratio_h_v"])
    raise KeyError(f)


def gap_v2(f_left, f_upper, layer="L12"):
    def acc(f):
        for r in rows(f):
            if r["probe"] == "logreg" and r["group"] == "far_bg" and r["layer"] == layer and "обучен" in r["weights"]:
                return float(r["acc"])
        raise KeyError(f)
    return (acc(f_upper) - acc(f_left)) * 100


def gap_big(f, layer="L32"):
    a = {}
    for r in rows(f):
        if r["probe"] == "logreg" and r["group"] == "far_bg" and r["layer"] == layer and "обучен" in r["weights"]:
            a[r["query"]] = float(r["acc"])
    return (a["upper"] - a["left"]) * 100


def new_net(f):
    x = y = None
    main = rows(f)[0]["net"]
    for r in rows(f):
        if r["net"] != main or r["layer"] != "L12":
            continue
        if r["measure"] == "mirror" and r["group"] == "mean":
            x = (float(r["value"]), float(r["ci_lo"]), float(r["ci_hi"]))
        if r["measure"] == "gap_upper_minus_left" and r["group"] == "far_bg":
            y = (float(r["value"]) * 100, float(r["ci_lo"]) * 100, float(r["ci_hi"]) * 100)
    return main, x, y


def main():
    pts = {
        "DINO": (mirror_ratio("flip_test.csv", "обученные DINO", "L12"), gap_v2("probe_v2.csv", "probe_v2_upper.csv")),
        "CLIP": (mirror_ratio("flip_test_clip.csv", "обученный CLIP", "L12"), gap_v2("probe_v2_clip.csv", "probe_v2_clip_upper.csv")),
        "MAE": (mirror_ratio("flip_test_mae.csv", "обученный MAE", "L32"), gap_big("probe_big_mae.csv")),
        "I-JEPA": (mirror_ratio("flip_test_ijepa.csv", "обученная I-JEPA", "L32"), gap_big("probe_big_ijepa.csv")),
    }
    ok = all(abs(pts[k][0] - v[0]) < 0.0015 and abs(pts[k][1] - v[1]) < 0.06 for k, v in OLD_EXPECTED.items())
    print(f"Проверка: 4 прежние сети совпадают с журналом: {'✅' if ok else '⚠️ проверь весь вывод'}")
    new = {}
    for f in ("more_nets_dinov2.csv", "more_nets_deit3.csv"):
        name, x, y = new_net(f)
        new[name] = (x, y)
        pts[name] = (x[0], y[0])
        print(f"  {name}: отношение {x[0]:.3f} [{x[1]:.3f}–{x[2]:.3f}], разрыв {y[0]:.1f} [{y[1]:.1f} … {y[2]:.1f}] п.п.")

    names = list(pts)
    xs = np.array([pts[n][0] for n in names])
    ys = np.array([pts[n][1] for n in names])
    rx = rankdata(xs)            # при равных значениях — средний ранг (так же считает spearmanr)
    ry = rankdata(-ys)
    print("\nсеть      отношение  ранг(слепее=1)   разрыв, п.п.  ранг(больше=1)")
    for i, n in enumerate(names):
        print(f"{n:9s} {xs[i]:8.3f}   {rx[i]:6.1f}          {ys[i]:8.1f}     {ry[i]:6.1f}")
    rho4 = spearmanr(xs[:4], ys[:4]).statistic
    rho = spearmanr(xs, ys).statistic
    verdict = "связь следует" if rho <= THRESH else "не подтверждена"
    print(f"\nρ Спирмена по 4 прежним сетям (известно заранее): {rho4:+.3f}")
    print(f"ρ Спирмена по 6 сетям: {rho:+.3f}   порог ≤ {THRESH}")
    print(f"ВЫВОД ПО ПРАВИЛУ 3 (реестр 114–115, записано до запуска): «{verdict}»")
    print("  Оговорки: 6 сетей различаются целью, данными, размером и сеткой патчей; ранговая связь — корреляция, не причина.")
    hit = verdict == "не подтверждена"
    print(f"\nСТАВКА ЕРНАЗАРА (реестр 114, 114e): «не подтверждена» -> вышло «{verdict}»  {'ПОПАЛ ✅' if hit else 'НЕ ПОПАЛ ❌'}")


if __name__ == "__main__":
    main()

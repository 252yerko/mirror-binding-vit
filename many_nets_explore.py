"""
many_nets_explore.py — опыт B (OSF xyedr): РАЗВЕДОЧНЫЙ разбор B2 после запуска. Не правило, не регистрировалось.
Добавлено 14.11 (реестр 150).

ЗАЧЕМ:
  R = разрыв в несимметричной части / обычный разрыв (как в правиле A2 опыта «A + D», OSF npvs5).
  В рамке «маскировка или стирание» R лежит от 0 (маскировка) до 1 (стирание). В опыте B у 7 из 24 сетей
  R > 1, у AugReg B/16 и L/16 — с нижней границей выше 1 при разрыве около 17 п.п., то есть это не шум
  деления на малый разрыв. Такое R в этой рамке не толкуется. Здесь проверяется, на чём держится B2:
    (1) B2 по всем 24 сетям — должно совпасть с many_nets_rules.csv (проверка кода);
    (2) B2 без каждого семейства по очереди (как B3 для B1);
    (3) B2 только по сетям с R ≤ 1 (порог выбран после того, как увидели данные).
  Те же ρ Спирмена и односторонний перестановочный p (10 000 перестановок, seed 0), что в many_nets.py.

ЗАПУСК:  python many_nets_explore.py   (читает many_nets.csv; < 1 мин)
"""
import csv
import sys
import numpy as np

CSV = sys.argv[1] if len(sys.argv) > 1 else "many_nets.csv"


def ranks(a):
    return np.argsort(np.argsort(a)).astype(float)


def spearman(x, y):
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def perm_p(x, y, sign=+1, n=10000):
    r = spearman(x, y)
    rng = np.random.default_rng(0)
    c = sum(sign * spearman(x, rng.permutation(y)) >= sign * r for _ in range(n))
    return r, (c + 1) / (n + 1)


rows = [r for r in csv.DictReader(open(CSV, encoding="utf-8")) if "СЛУЧ" not in r["weights"]]
x = np.array([float(r["ratio"]) for r in rows])
R = np.array([float(r["R"]) for r in rows])
fam = np.array([r["family"] for r in rows])
print(f"Обученных сетей: {len(rows)}; R определено у {int(np.isfinite(R).sum())}")
big = ["%s %.2f [%.2f; %.2f]" % (r["name"], float(r["R"]), float(r["R_lo"]), float(r["R_hi"])) for r in rows if float(r["R"]) > 1]
print("R > 1: " + ", ".join(big))

r0, p0 = perm_p(x, R)
ref = {r["rule"]: r for r in csv.DictReader(open(CSV.replace("many_nets.csv", "many_nets_rules.csv"), encoding="utf-8"))}
ok = abs(r0 - float(ref["B2"]["rho"])) < 1e-9 and abs(p0 - float(ref["B2"]["p"])) < 1e-9
print(f"\n(1) B2, все сети: ρ = {r0:+.3f}, p = {p0:.4f}, n = {len(x)}   совпадает с many_nets_rules.csv: {'✅' if ok else '⚠️ проверь весь вывод'}")

print("\n(2) B2 без одного семейства:")
for f in sorted(set(fam)):
    m = fam != f
    r, p = perm_p(x[m], R[m])
    print(f"    без {f:12s}: ρ = {r:+.3f}, p = {p:.4f} (n = {m.sum()})")

m = R <= 1
r, p = perm_p(x[m], R[m])
print(f"\n(3) B2 только по сетям с R ≤ 1 (порог после просмотра данных): ρ = {r:+.3f}, p = {p:.4f} (n = {m.sum()})")
print("\nРазведочно, после запуска; не правило регистрации xyedr.")

"""
probe_breakdown.py — ОТ ЧЕГО ЗАВИСИТ, насколько хорошо читается привязка?

Берёт ту же пробу, что probe_v2.py (логрег, C=0.1, те же 5 фолдов по сценам), но сохраняет
балл КАЖДОЙ сцены и раскладывает точность по группам сцен:
  (1) по числу объектов: 2 / 3 / 4
  (2) по разносу близнецов ВДОЛЬ ОСИ ВОПРОСА (в пикселях между центрами):
      вопрос «левый»  -> разнос по горизонтали |dx|
      вопрос «верхний» -> разнос по вертикали  |dy|
      корзины: < 16 px (меньше патча), 16–48, 48–96, ≥ 96

ПРЕДСКАЗАНИЯ (записаны ДО запуска, 5.10):
  П1. Чем больше объектов, тем хуже читается привязка (как у Campbell et al. 2024 в поведении:
      ошибки привязки растут с числом объектов).
  П2. Чем больше разнос вдоль оси вопроса, тем лучше читается (при разносе < 1 патча вопрос
      «кто левее» почти неразличим для сети, которая видит мир кусками 16×16).
  П3 (из гипотезы «зеркало»). На глубоких слоях (L08–L12) для «левого» большой разнос
      помогает СЛАБО, а для «верхнего» — сильно: горизонтальная привязка подавлена целиком,
      а не только для близко стоящих близнецов.
  Если П3 не подтвердится (для «левого» на L12 при разносе ≥ 96 px точность так же высока,
  как для «верхнего»), то спад «левого» — это проблема только близких близнецов, и гипотезу
  «зеркало» надо пересмотреть.

ГРУППЫ ТОКЕНОВ: CLS и дальний фон (там привязка читается устойчиво).
СЕТИ: обученная DINO и три случайные (rand0/1/2) — нулевая модель ДЛЯ КАЖДОЙ КОРЗИНЫ.

ВАЖНО (найдено в облачном тесте 5.10): уровень случайности В ОТДЕЛЬНОЙ КОРЗИНЕ не обязан быть 50%.
  Проба учится на всех сценах сразу; в узкой группе сцен её правило может «переворачиваться»,
  и у случайных сетей в корзине выходит, например, 44–46%. Поэтому DINO в каждой корзине сравниваем
  со случайными сетями В ТОЙ ЖЕ корзине (печатается рядом), а не с числом 50.

ВСТРОЕННЫЕ ПРОВЕРКИ:
  1. Общая точность по всем сценам должна СОВПАСТЬ с probe_v2.csv / probe_v2_upper.csv
     до 5 знаков (та же проба, те же фолды) — иначе где-то ошибка.
  2. На L00 точность в каждой корзине ровно 50.0%.

ЗАПУСК (из папки binding-vit):
  python probe_breakdown.py 2>&1 | tee probe_breakdown_log.txt
  -> таблицы, график probe_breakdown.png, числа probe_breakdown.csv (~5–10 мин)
"""

import csv
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import probe_v2 as pv

NETS = [("acts_v2", "DINO"), ("acts_v2_rand0", "rand0"), ("acts_v2_rand1", "rand1"), ("acts_v2_rand2", "rand2")]
QUERIES = ["left", "upper"]
GROUPS = ["cls", "far_bg"]
NAMES = {"cls": "CLS", "far_bg": "дальний фон"}
SEP_EDGES = [0, 16, 48, 96, 10_000]
SEP_LABELS = ["<16", "16–48", "48–96", "≥96"]
SHOW_LAYERS = [3, 5, 8, 12]
REF_CSV = {"left": "probe_v2.csv", "upper": "probe_v2_upper.csv"}


def boot(x, rng, n=1000):
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, len(x), size=(n, len(x)))
    b = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    t0 = time.time()
    pv.MODELS = ["logreg"]                       # только основная проба
    meta = [json.loads(line) for line in open(f"{pv.DATA}/meta.jsonl")]
    colors = sorted({o[k] for r in meta for o in r["objects"] for k in ("color", "color_Sp")})
    folds = np.random.default_rng(0).permutation(len(meta)) % pv.N_FOLDS      # как в probe_v2.py
    nobj = np.array([r["set_size"] for r in meta])
    dx = np.array([abs(r["objects"][0]["center"][0] - r["objects"][1]["center"][0]) for r in meta])
    dy = np.array([abs(r["objects"][0]["center"][1] - r["objects"][1]["center"][1]) for r in meta])
    sep = {"left": dx, "upper": dy}
    print(f"Сцен: {len(meta)}. По числу объектов: " + ", ".join(f"{k}: {(nobj == k).sum()}" for k in (2, 3, 4)))
    for q in QUERIES:
        b = np.digitize(sep[q], SEP_EDGES[1:-1])
        print(f"  разнос близнецов для «{pv.QWORD[q]}» ({'|dx|' if q == 'left' else '|dy|'}): " +
              ", ".join(f"{SEP_LABELS[i]} px: {(b == i).sum()}" for i in range(4)))

    scores = {}          # (net, query, group, layer) -> балл каждой сцены [2000]
    for q in QUERIES:
        pv.QUERY = q
        pv.GROUPS[3] = f"{q}_twin"
        infos = [pv.scene_info(rec, colors) for rec in meta]
        for acts, net in NETS:
            rows = {(r["id"], r["version"]): int(r["row"]) for r in csv.DictReader(open(f"{acts}/index.csv"))}
            for l in range(pv.N_LAYERS):
                A = np.load(f"{acts}/L{l:02d}.npy", mmap_mode="r")
                feats = {g: ([], [], [], []) for g in GROUPS}
                for si, (rec, (groups, labels)) in enumerate(zip(meta, infos)):
                    for v in ("S", "Sp"):
                        h = np.asarray(A[rows[(rec["id"], v)]], dtype=np.float32)
                        for g in GROUPS:
                            feats[g][0].append(h[groups[g]].mean(0))
                            feats[g][1].append(labels[v][0])
                            feats[g][2].append(labels[v][1])
                            feats[g][3].append(si)
                for g in GROUPS:
                    X, yt, yr, sc = (np.array(a) for a in feats[g])
                    per_scene, _ = pv.run_probe(X, yt, yr, sc, folds)
                    assert len(per_scene["logreg"]) == len(meta)
                    scores[(net, q, g, l)] = per_scene["logreg"]
            print(f"  готово: «{pv.QWORD[q]}», {net}  ({time.time() - t0:.0f} с)", flush=True)

    # --- проверка 1: общая точность совпадает с основными прогонами
    worst, n_cmp, n_all = 0.0, 0, 0
    for q in QUERIES:
        ref = {(r["acts"], r["group"], r["layer"]): float(r["acc"]) for r in csv.DictReader(open(REF_CSV[q]))
               if r["probe"] == "logreg"}
        for acts, net in NETS:
            for g in GROUPS:
                for l in range(pv.N_LAYERS):
                    n_all += 1
                    key = (acts, g, f"L{l:02d}")
                    if key in ref:
                        n_cmp += 1
                        worst = max(worst, abs(round(scores[(net, q, g, l)].mean(), 5) - ref[key]))
    ok1 = worst < 1e-5 and n_cmp == n_all
    print(f"\nПроверка 1 (общая точность = probe_v2.csv / probe_v2_upper.csv): сверено {n_cmp} из {n_all} чисел, "
          f"макс. расхождение {worst:.5f}  {'✅' if ok1 else '⚠️ пришли Claude весь вывод'}")

    # --- раскладка
    rng = np.random.default_rng(0)
    out, l00 = [], []
    for q in QUERIES:
        sbin = np.digitize(sep[q], SEP_EDGES[1:-1])
        for net in [n for _, n in NETS]:
            for g in GROUPS:
                for l in range(pv.N_LAYERS):
                    s = scores[(net, q, g, l)]
                    for kind, labels, mask_of in (("objects", [2, 3, 4], lambda k: nobj == k),
                                                  ("separation", list(range(4)), lambda k: sbin == k)):
                        for k in labels:
                            a, lo, hi = boot(s[mask_of(k)], rng)
                            lab = str(k) if kind == "objects" else SEP_LABELS[k]
                            out.append([net, q, g, f"L{l:02d}", kind, lab, a, lo, hi, int(mask_of(k).sum())])
                            if l == 0:
                                l00.append(a)
    ok2 = all(abs(x - 0.5) < 1e-12 for x in l00)
    print(f"Проверка 2 (L00 в каждой корзине ровно 50.0%): {'✅' if ok2 else '⚠️ пришли Claude весь вывод'}")

    def get(net, q, g, l, kind, lab):
        r = [x for x in out if x[0] == net and x[1] == q and x[2] == g and x[3] == f"L{l:02d}" and x[4] == kind and x[5] == lab][0]
        return r[6], r[7], r[8]

    RAND = [n for _, n in NETS if n != "DINO"]

    def rnd(q, g, l, kind, lab):
        v = [get(n, q, g, l, kind, lab)[0] * 100 for n in RAND]
        return f"(сл. {np.mean(v):4.1f}; {min(v):.1f}–{max(v):.1f})"

    for g in GROUPS:
        print(f"\n================ {NAMES[g]} (логрег; DINO [95% инт.], в скобках — случайные сети: среднее; мин–макс) ================")
        for q in QUERIES:
            print(f"\n-- вопрос «{pv.QWORD[q]}»: по ЧИСЛУ ОБЪЕКТОВ --")
            print("слой   " + "".join(f"{k} объекта".ljust(40) for k in (2, 3, 4)))
            for l in SHOW_LAYERS:
                cells = []
                for k in ("2", "3", "4"):
                    a, lo, hi = get("DINO", q, g, l, "objects", k)
                    cells.append(f"{a * 100:5.1f} [{lo * 100:4.1f}–{hi * 100:4.1f}] {rnd(q, g, l, 'objects', k)}".ljust(40))
                print(f"L{l:02d}   " + "".join(cells))
            print(f"-- вопрос «{pv.QWORD[q]}»: по РАЗНОСУ близнецов вдоль оси ({'|dx|' if q == 'left' else '|dy|'}, px) --")
            print("слой   " + "".join(f"{lab} px".ljust(40) for lab in SEP_LABELS))
            for l in SHOW_LAYERS:
                cells = []
                for lab in SEP_LABELS:
                    a, lo, hi = get("DINO", q, g, l, "separation", lab)
                    cells.append(f"{a * 100:5.1f} [{lo * 100:4.1f}–{hi * 100:4.1f}] {rnd(q, g, l, 'separation', lab)}".ljust(40))
                print(f"L{l:02d}   " + "".join(cells))

    print("\nИТОГ по предсказаниям (DINO, CLS; в скобках — среднее случайных сетей в той же корзине):")
    def line(q, l, kind, lab):
        a, lo, hi = get("DINO", q, "cls", l, kind, lab)
        r = np.mean([get(n, q, "cls", l, kind, lab)[0] for n in RAND])
        return f"{a * 100:.1f}% [{lo * 100:.1f}–{hi * 100:.1f}] (сл. {r * 100:.1f})"
    for q, l in (("upper", 12), ("left", 3)):
        print(f"  П1 «{pv.QWORD[q]}» L{l:02d}: 2 объекта {line(q, l, 'objects', '2')}  →  4 объекта {line(q, l, 'objects', '4')}")
    for q in QUERIES:
        print(f"  П2/П3 «{pv.QWORD[q]}» L12: разнос <16 px {line(q, 12, 'separation', '<16')}  →  ≥96 px {line(q, 12, 'separation', '≥96')}")
    print("  П3 подтверждается, если при разносе ≥96 px «левый» на L12 заметно ниже «верхнего» (интервалы не пересекаются)")

    with open("probe_breakdown.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["net", "query", "group", "layer", "split", "bin", "acc", "ci_lo", "ci_hi", "n_scenes"])
        for r in out:
            w.writerow(r[:6] + [f"{r[6]:.5f}", f"{r[7]:.5f}", f"{r[8]:.5f}", r[9]])

    # --- график: CLS, DINO; слева — по числу объектов, справа — по разносу; сплошные «верхний», пунктир «левый»
    L = np.arange(pv.N_LAYERS)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    for k, c in zip(("2", "3", "4"), ("C0", "C1", "C2")):
        for q, ls in (("upper", "-"), ("left", "--")):
            axes[0].plot(L, [get("DINO", q, "cls", l, "objects", k)[0] * 100 for l in L], ls, c=c, marker="o", ms=3,
                         label=f"{k} объекта, «{pv.QWORD[q]}»")
    for lab, c in zip(SEP_LABELS, ("C7", "C0", "C1", "C3")):
        for q, ls in (("upper", "-"), ("left", "--")):
            axes[1].plot(L, [get("DINO", q, "cls", l, "separation", lab)[0] * 100 for l in L], ls, c=c, marker="o", ms=3,
                         label=f"{lab} px, «{pv.QWORD[q]}»")
    axes[0].set_title("CLS, DINO: по числу объектов")
    axes[1].set_title("CLS, DINO: по разносу близнецов вдоль оси вопроса")
    for ax in axes:
        ax.axhline(50, ls=":", c="k", lw=1)
        ax.set_xticks(L, [f"L{l:02d}" for l in L], fontsize=7)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    axes[0].set_ylabel("точность пробы, %  (50 = мешок признаков)")
    fig.suptitle("От чего зависит, насколько читается привязка (сплошная — «верхний», пунктир — «левый»)", fontsize=10)
    fig.tight_layout()
    fig.savefig("probe_breakdown.png", dpi=150)
    print(f"\nГрафик: probe_breakdown.png   Числа: probe_breakdown.csv   ({time.time() - t0:.0f} с)")


if __name__ == "__main__":
    main()

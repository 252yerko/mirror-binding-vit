"""probe_fold_strata.py — ДОПОЛНИТЕЛЬНАЯ проверка к опыту «складывание» (6.10; НЕ была записана до запуска — это разбор после).
Ничего не обучает: берёт баллы сцен из probe_fold[_clip]_scores.npz и делит сцены общего подмножества на группы.
  (а) «ближе к середине» против «ближе к центру по прямой»: если проба читает ответ только через то, насколько объект
      близок к центру картинки вообще (в 2D), то там, где эти ответы расходятся, точность упадёт НИЖЕ 50%.
  (б) 2 объекта против 3–4: в сценах из 2 объектов нет третьей фигуры, через близость к которой можно было бы прочитать ответ.
Запуск (из binding-vit):  python probe_fold_strata.py                      (DINO)
                          python probe_fold_strata.py . acts_v2_clip probe_fold_clip   (CLIP)
В квадратных скобках — 95% бутстрэп-интервал по сценам, n — число сцен."""
import json, sys, numpy as np
D, NET, STEM = (sys.argv[1:4] if len(sys.argv) >= 4 else (".", "acts_v2", "probe_fold"))
MID = 111.5
meta = [json.loads(l) for l in open(f"{D}/data_v2/meta.jsonl")]
Z = np.load(f"{D}/{STEM}_scores.npz"); SUB = Z["sub"]
cen = np.array([[o["center"] for o in r["objects"][:2]] for r in meta], float)
nobj = np.array([r["set_size"] for r in meta])
ax = np.abs(cen[:, :, 0] - MID); ay = np.abs(cen[:, :, 1] - MID)
def pick(k):  # 0/1: какой близнец меньше по ключу (правила при равенстве не нужны — на SUB равенств нет)
    return (k[:, 1] < k[:, 0]).astype(int)
tw = {"cx": pick(ax), "cy": pick(ay), "e2": pick(ax**2 + ay**2)}
rng = np.random.default_rng(0)
def acc(s, m):
    v = s[m]; v = v[~np.isnan(v)]
    b = [rng.choice(v, len(v)).mean() for _ in range(1000)]
    return f"{v.mean()*100:5.1f} [{np.percentile(b,2.5)*100:4.1f}…{np.percentile(b,97.5)*100:4.1f}] n={len(v)}"
for g in ("cls", "far_bg"):
    print(f"=== {NET} {g}")
    for L in (3, 8, 12):
        for q in ("cx", "cy"):
            s = Z[f"{NET}|{q}|{g}|L{L:02d}"]
            same = SUB & (tw[q] == tw["e2"]); diff = SUB & (tw[q] != tw["e2"])
            print(f" L{L:02d} {q}: близнец-ответ = ближе к центру по прямой: {acc(s, same)} | НЕ ближе по прямой: {acc(s, diff)}")
        for q in ("left", "upper", "cx", "cy"):
            s = Z[f"{NET}|{q}|{g}|L{L:02d}"]
            print(f" L{L:02d} {q:5s}: 2 объекта {acc(s, SUB & (nobj == 2))} | 3–4 объекта {acc(s, SUB & (nobj > 2))}")

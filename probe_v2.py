"""
probe_v2.py — ступенька 3 «читается»: можно ли ПРОЧИТАТЬ из токенов, какой близнец какого цвета?

ВОПРОС ПРОБЕ: «какого цвета ЛЕВЫЙ близнец?»
  Близнецы — две фигуры одной формы и размера. Левый = тот, у кого центр левее
  (при равенстве — тот, что выше). Места близнецов в S и Sp одинаковые,
  а цвета переставлены, поэтому в S и Sp правильный ответ РАЗНЫЙ.

ПОЧЕМУ ЧЕСТНЫЙ УРОВЕНЬ СЛУЧАЙНОСТИ = РОВНО 50%:
  Для каждой сцены проба отвечает на две картинки: S и Sp. Набор цветов, фигуры и места
  у них одинаковые. Если токен — «мешок признаков», он в S и Sp ОДИНАКОВЫЙ, значит проба
  обязана дать одинаковый ответ, и ровно одна из двух картинок будет угадана.
  Итог — ровно 50%, при любой пробе. Всё, что выше 50%, — информация о привязке.
  Засчитываем так: из двух цветов близнецов проба должна поставить правильный выше
  (выбор только между двумя цветами близнецов, поэтому цвета других объектов не помогают).

ГДЕ ЧИТАЕМ (по каждой картинке — один вектор из 768 чисел):
  cls         — токен CLS
  нетронутые  — среднее по патчам остальных объектов (их пиксели в S и Sp одинаковые)
  дальний фон — среднее по пустым патчам на расстоянии ≥ 3 патчей от обоих близнецов
                (рядом с близнецом патч сам видит цвет — это мы исключаем)
  левый близнец — среднее по патчам самого левого близнеца. ПОЛОЖИТЕЛЬНЫЙ КОНТРОЛЬ:
                он прямо видит свой цвет, должно быть ~100%

КАК ОБУЧАЕМ:
  Основная проба: StandardScaler + логистическая регрессия (6 цветов, C = 0.1 — выбрано
  заранее, по результатам не подбирается). Проверка устойчивости: другой линейный
  классификатор — ридж (RidgeClassifier, alpha = 1000, тоже выбрано заранее). Если оба
  согласны, результат не зависит от выбора классификатора.
  5-кратная кросс-валидация ПО СЦЕНАМ: S и Sp одной сцены всегда вместе — либо в обучении,
  либо в проверке. 95% интервал — бутстрэп по сценам. Всё повторяется для обученной DINO
  и для трёх случайных сетей (нулевая модель).
  Проверено заранее (облако, 5.10): на чистом шуме обе пробы дают 50% (логрег 50,45 ± 0,53,
  ридж 49,87 ± 0,58 по 12 наборам) — метод не смещён.

ВСТРОЕННЫЕ ПРОВЕРКИ:
  1. На L00 токены CLS / нетронутых / дальнего фона в S и Sp совпадают до бита ->
     точность там обязана быть РОВНО 50.0%. Иначе — ошибка в коде.
  2. Пиксели патчей «нетронутые» и «дальний фон» в S и Sp сравниваются — должны совпадать.
  3. Считаются предупреждения «обучение не сошлось» — их число печатается.

ЗАПУСК (после extract.py для acts_v2 и acts_v2_rand0/1/2):
  python probe_v2.py 2>&1 | tee probe_v2_log.txt
  -> таблица, график probe_v2.png, числа probe_v2.csv
  ~15 мин на 4 сети (M3). Файл с числами перезаписывается после КАЖДОЙ сети — если запуск
  прервётся, готовые сети не потеряются.

ПРОВЕРКИ НАДЁЖНОСТИ (добавлено 5.10, без флагов скрипт работает ровно как раньше):
  --query upper  вопрос «какого цвета ВЕРХНИЙ близнец» (выше = меньше y; при равенстве — левее).
                 Информация та же (зная цвет левого, знаешь и цвет верхнего), но читать её надо
                 «по вертикали». Проверяем, что результат не привязан к горизонтальному вопросу.
  --probe mlp    НЕЛИНЕЙНАЯ проба: маленькая нейросеть (1 скрытый слой из 256 нейронов, alpha=0.01,
                 ранняя остановка; выбрано заранее). Отвечает на вопрос: к L12 информация о привязке
                 пропадает или только перестаёт читаться линейно? Проверено в облаке: на шуме 49,3 ± 0,5%
                 (не смещена); на линейном сигнале чуть слабее логрега (90 против 93%); нелинейный
                 сигнал находит (~100%), который логрег не видит (50%).
  --tag ИМЯ      приписка к файлам: probe_v2_ИМЯ.csv / .png (основные probe_v2.* не перезаписываются)
  Примеры:
    python probe_v2.py --query upper --tag upper 2>&1 | tee probe_v2_upper_log.txt
  CLIP (6.10): первая папка — обученный CLIP, остальные — случайные CLIP:
    python probe_v2.py --acts acts_v2_clip acts_v2_clip_rand0 acts_v2_clip_rand1 acts_v2_clip_rand2 --tag clip 2>&1 | tee probe_v2_clip_log.txt
    python probe_v2.py --acts acts_v2_clip acts_v2_clip_rand0 acts_v2_clip_rand1 acts_v2_clip_rand2 --query upper --tag clip_upper 2>&1 | tee probe_v2_clip_upper_log.txt
    python probe_v2.py --probe mlp --tag mlp 2>&1 | tee probe_v2_mlp_log.txt
"""

import argparse
import csv
import json
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = "data_v2"
P, G, N_LAYERS = 16, 14, 13
FAR = 3                       # «дальний» фон: не ближе 3 патчей к обоим близнецам
C_REG = 0.1                   # логрег: сила регуляризации, выбрана заранее
RIDGE_ALPHA = 1000.0          # ридж (проверка устойчивости): выбрано заранее
MODELS = ["logreg", "ridge"]   # можно заменить флагом --probe (первая — основная)
PROBE_NAMES = {"logreg": "логрег", "ridge": "ридж", "mlp": "MLP"}
PROBE_LONG = {"logreg": "логрег — основная проба", "ridge": "ридж — проверка устойчивости",
              "mlp": "MLP — нелинейная проба"}
QUERY = "left"                  # можно заменить флагом --query
QWORD = {"left": "левый", "upper": "верхний"}
STEM = "probe_v2"               # имя выходных файлов; с --tag становится probe_v2_ТЕГ
N_FOLDS, N_BOOT = 5, 1000
GROUPS = ["cls", "unchanged", "far_bg", "left_twin"]   # последняя группа меняется на upper_twin при --query upper
NAMES = {"cls": "CLS", "unchanged": "нетронутые", "far_bg": "дальний фон", "left_twin": "левый близнец (контроль)",
         "upper_twin": "верхний близнец (контроль)"}


# ------------------------------------------------------------------
# 1. Для каждой сцены: какие токены в какой группе и какой правильный ответ
# ------------------------------------------------------------------
def scene_info(rec, colors):
    sid = rec["id"]
    assert rec["twins"] == [0, 1], f"{sid}: близнецы не объекты 0 и 1"
    o0, o1 = rec["objects"][0], rec["objects"][1]
    # близнец-«вопрос». left: меньше x центра (center = [x, y]), при равенстве — меньше y.
    # upper: меньше y (выше на картинке), при равенстве — меньше x.
    # Ниже переменные называются left/right, но означают «близнец-вопрос» / «другой близнец».
    if QUERY == "left":
        k0, k1 = (o0["center"][0], o0["center"][1]), (o1["center"][0], o1["center"][1])
    else:
        k0, k1 = (o0["center"][1], o0["center"][0]), (o1["center"][1], o1["center"][0])
    left = 0 if k0 < k1 else 1
    right = 1 - left
    oL, oR = rec["objects"][left], rec["objects"][right]
    assert oL["color_Sp"] == oR["color"] and oR["color_Sp"] == oL["color"], f"{sid}: в Sp цвета не переставлены"

    mask = np.asarray(Image.open(f"{DATA}/masks/{sid}.png"))
    blk = mask.reshape(G, P, G, P).transpose(0, 2, 1, 3).reshape(G * G, P * P)  # [патч, пиксели]
    has_L = (blk == left + 1).any(1)        # в маске объект k помечен числом k+1
    has_R = (blk == right + 1).any(1)
    twin = has_L | has_R
    other = (blk >= 3).any(1)
    empty = (blk == 0).all(1)
    ty, tx = np.where(twin)[0] // G, np.where(twin)[0] % G
    dist = np.array([np.min(np.maximum(abs(p // G - ty), abs(p % G - tx))) for p in range(G * G)])

    groups = dict(
        cls=[0],
        unchanged=[p + 1 for p in np.where(other & ~twin)[0]],
        far_bg=[p + 1 for p in np.where(empty & (dist >= FAR))[0]],
        **{GROUPS[3]: [p + 1 for p in np.where(has_L & ~has_R)[0]]},
    )
    # проверка 2: в «нетронутых» и «дальнем фоне» пиксели S и Sp совпадают
    S = np.asarray(Image.open(f"{DATA}/images/{sid}_S.png").convert("RGB"))
    Sp = np.asarray(Image.open(f"{DATA}/images/{sid}_Sp.png").convert("RGB"))
    for k in ("unchanged", "far_bg"):
        for t in groups[k]:
            py, px = divmod(t - 1, G)
            blkS = S[py * P:(py + 1) * P, px * P:(px + 1) * P]
            blkSp = Sp[py * P:(py + 1) * P, px * P:(px + 1) * P]
            assert np.array_equal(blkS, blkSp), f"{sid}: патч {t - 1} ({k}) в S и Sp различается"
    ci = {c: i for i, c in enumerate(colors)}
    # правильный ответ (цвет левого близнеца) и «соперник» (цвет правого) для S и для Sp
    labels = {"S": (ci[oL["color"]], ci[oR["color"]]), "Sp": (ci[oL["color_Sp"]], ci[oR["color_Sp"]])}
    return groups, labels


# ------------------------------------------------------------------
# 2. Проба с кросс-валидацией по сценам
# ------------------------------------------------------------------
def make_model(name):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=C_REG, max_iter=3000))
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-2, early_stopping=True,
                                                             validation_fraction=0.1, n_iter_no_change=10,
                                                             max_iter=300, random_state=0))
    return make_pipeline(StandardScaler(), RidgeClassifier(alpha=RIDGE_ALPHA))


def run_probe(X, y_true, y_rival, scene_idx, folds):
    """X: [картинки, 768]. Для каждой пробы (логрег, ридж) возвращает балл каждой сцены
    (0, 0.5 или 1) и общее число несошедшихся обучений."""
    scores = {m: np.zeros(len(X)) for m in MODELS}
    n_warn = 0
    for f in range(N_FOLDS):
        test = folds[scene_idx] == f
        for m in MODELS:
            clf = make_model(m)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)
                clf.fit(X[~test], y_true[~test])
                n_warn += sum(issubclass(x.category, ConvergenceWarning) for x in w)
            # логрег даёт вероятности, ридж — «баллы» классов; сравниваем одинаково: правильный цвет выше соперника?
            d = clf.predict_proba(X[test]) if m in ("logreg", "mlp") else clf.decision_function(X[test])
            col = {c: i for i, c in enumerate(clf.classes_)}
            pt = d[np.arange(test.sum()), [col[c] for c in y_true[test]]]
            pr = d[np.arange(test.sum()), [col[c] for c in y_rival[test]]]
            scores[m][test] = np.where(pt > pr, 1.0, np.where(pt == pr, 0.5, 0.0))
    # балл сцены = среднее по её двум картинкам (S и Sp)
    n_sc = scene_idx.max() + 1
    cnt = np.bincount(scene_idx, minlength=n_sc)
    present = cnt > 0
    per_scene = {m: (np.bincount(scene_idx, weights=scores[m], minlength=n_sc) / np.maximum(cnt, 1))[present] for m in MODELS}
    return per_scene, n_warn


def boot_ci(per_scene, rng):
    idx = rng.integers(0, len(per_scene), size=(N_BOOT, len(per_scene)))
    b = per_scene[idx].mean(1)
    return float(per_scene.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def save_csv(acts_list, out):
    with open(f"{STEM}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["acts", "weights", "probe", "group", "layer", "acc", "ci_lo", "ci_hi", "n_scenes"])
        for a, (wt, r, _) in zip(acts_list, out):
            for m in MODELS:
                for g in GROUPS:
                    for l in range(N_LAYERS):
                        acc, lo, hi, n = r[m][g][l]
                        w.writerow([a, wt, m, g, f"L{l:02d}", f"{acc:.5f}", f"{lo:.5f}", f"{hi:.5f}", n])


def analyse(acts, meta, infos, folds):
    rows = {(r["id"], r["version"]): int(r["row"]) for r in csv.DictReader(open(f"{acts}/index.csv"))}
    weights = json.load(open(f"{acts}/info.json")).get("weights", "обученные DINO")
    print(f"\n### {acts}/   веса: {weights}")
    rng = np.random.default_rng(0)
    res = {m: {g: [] for g in GROUPS} for m in MODELS}
    warn_total, t0 = 0, time.time()
    for l in range(N_LAYERS):
        A = np.load(f"{acts}/L{l:02d}.npy", mmap_mode="r")
        feats = {g: ([], [], [], []) for g in GROUPS}   # X, y_true, y_rival, scene
        for si, (rec, (groups, labels)) in enumerate(zip(meta, infos)):
            for v in ("S", "Sp"):
                h = np.asarray(A[rows[(rec["id"], v)]], dtype=np.float32)
                for g in GROUPS:
                    if groups[g]:
                        feats[g][0].append(h[groups[g]].mean(0))
                        feats[g][1].append(labels[v][0])
                        feats[g][2].append(labels[v][1])
                        feats[g][3].append(si)
        line = []
        for g in GROUPS:
            X, yt, yr, sc = (np.array(a) for a in feats[g])
            per_scene, nw = run_probe(X, yt, yr, sc, folds)
            warn_total += nw
            for m in MODELS:
                acc, lo, hi = boot_ci(per_scene[m], rng)
                res[m][g].append((acc, lo, hi, len(per_scene[m])))
            extra = "".join(f" ({PROBE_NAMES[m]} {res[m][g][-1][0] * 100:5.1f})" for m in MODELS[1:])
            line.append(f"{NAMES[g].split(' ')[0]} {res[MODELS[0]][g][-1][0] * 100:5.1f}%{extra}")
        print(f"  L{l:02d}  " + "   ".join(line) + f"   ({time.time() - t0:.0f} с)", flush=True)
    print(f"  предупреждений «обучение не сошлось»: {warn_total}")
    return weights, res, warn_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", nargs="+", default=["acts_v2", "acts_v2_rand0", "acts_v2_rand1", "acts_v2_rand2"],
                    help="первая папка — обученная DINO, остальные — случайные сети")
    ap.add_argument("--query", choices=["left", "upper"], default="left", help="какой близнец спрашиваем")
    ap.add_argument("--probe", choices=["linear", "mlp"], default="linear",
                    help="linear = логрег + ридж (как раньше); mlp = нелинейная проба")
    ap.add_argument("--tag", default="", help="приписка к именам выходных файлов")
    args = ap.parse_args()
    global QUERY, MODELS, STEM
    QUERY = args.query
    MODELS = ["mlp"] if args.probe == "mlp" else ["logreg", "ridge"]
    STEM = "probe_v2" + (f"_{args.tag}" if args.tag else "")
    GROUPS[3] = f"{QUERY}_twin"
    print(f"Вопрос: «какого цвета {QWORD[QUERY]} близнец»   проба: {', '.join(PROBE_LONG[m] for m in MODELS)}   "
          f"выход: {STEM}.csv, {STEM}.png")

    meta = [json.loads(line) for line in open(f"{DATA}/meta.jsonl")]
    colors = sorted({o[k] for r in meta for o in r["objects"] for k in ("color", "color_Sp")})
    infos = [scene_info(rec, colors) for rec in meta]
    n_g = {g: sum(1 for gr, _ in infos if gr[g]) for g in GROUPS}
    print(f"Сцен: {len(meta)}. Цвета ({len(colors)}): {', '.join(colors)}")
    print("Сцен с непустой группой: " + ", ".join(f"{NAMES[g]} {n_g[g]}" for g in GROUPS))
    print(f"Проверка пикселей ✅ в «нетронутых» и «дальнем фоне» S и Sp совпадают во всех сценах")
    desc = {"logreg": f"логистическая регрессия C={C_REG}", "ridge": f"ридж alpha={RIDGE_ALPHA:g}",
            "mlp": "MLP 256 нейронов, alpha=0.01, ранняя остановка"}
    print(f"Проба: StandardScaler + {' / '.join(desc[m] for m in MODELS)}, {N_FOLDS} фолдов по сценам. Уровень случайности = 50%")

    folds = np.random.default_rng(0).permutation(len(meta)) % N_FOLDS
    out = []
    for a in args.acts:
        out.append(analyse(a, meta, infos, folds))
        save_csv(args.acts[:len(out)], out)          # сохраняем после каждой сети
        print(f"  (сохранено в {STEM}.csv: {len(out)} из {len(args.acts)} сетей)", flush=True)
    total_warn = sum(w for _, _, w in out)
    full = out
    out = [(wt, r[MODELS[0]], w) for wt, r, w in full]         # основная проба — первая в MODELS
    (w_d, dino, _), rand = out[0], out[1:]
    # подпись первой сети берём из её info.json (чтобы надпись совпадала с тем, что реально сравнили)
    SHORT = {"обученные DINO": ("DINO", "обученная DINO"), "обученный CLIP (OpenAI)": ("CLIP", "обученный CLIP (OpenAI)")}
    ref, ref_long = SHORT.get(w_d, (f"[{w_d}]", w_d))
    print(f"\nСравниваем: {ref_long} ({args.acts[0]}) против {len(rand)} случайных сетей ({', '.join(args.acts[1:]) or '—'})")

    # --- проверка 1: L00 ровно 50% там, где токены S и Sp совпадают (обе пробы, все сети)
    l00 = [r[m][g][0][0] for _, r, _ in full for m in MODELS for g in ("cls", "unchanged", "far_bg")]
    ok = all(abs(x - 0.5) < 1e-12 for x in l00)
    print(f"\nПроверка L00 (CLS, нетронутые, дальний фон должны быть ровно 50.0%): "
          f"{'✅' if ok else '⚠️ проверь весь вывод'}  {sorted(set(round(x * 100, 3) for x in l00))}")

    # --- таблица
    print(f"\nТочность пробы «какого цвета {QWORD[QUERY]} близнец», {PROBE_LONG[MODELS[0]]} (50% = мешок признаков)")
    for g in GROUPS:
        print(f"\n== {NAMES[g]} ==   сцен: {dino[g][0][3]}")
        print(f"слой   {ref}   [95% инт.]       случайные: среднее (мин–макс)")
        for l in range(N_LAYERS):
            a, lo, hi, _ = dino[g][l]
            rr = [r[g][l][0] for _, r, _ in rand]
            rs = f"{np.mean(rr) * 100:5.1f}% ({min(rr) * 100:.1f}–{max(rr) * 100:.1f})" if rr else "—"
            print(f"L{l:02d}  {a * 100:5.1f}%  [{lo * 100:5.1f}–{hi * 100:5.1f}]     {rs}")

    # --- форма кривой у первой сети: где пик и насколько падает к L12
    print(f"\nФорма кривой ({ref}, {PROBE_NAMES[MODELS[0]]}): пик → L12")
    for g in GROUPS[:3]:
        accs = [x[0] for x in dino[g]]
        lp = int(np.argmax(accs))
        print(f"  {NAMES[g]:12s} пик L{lp:02d} {accs[lp] * 100:5.1f}% [{dino[g][lp][1] * 100:.1f}–{dino[g][lp][2] * 100:.1f}]   "
              f"L12 {accs[-1] * 100:5.1f}% [{dino[g][-1][1] * 100:.1f}–{dino[g][-1][2] * 100:.1f}]   падение {100 * (accs[lp] - accs[-1]):4.1f} п.п.")

    print(f"\nПредупреждений «обучение не сошлось» всего: {total_warn}  {'✅' if total_warn == 0 else '⚠️ проверь весь вывод'}")

    if rand:
        for m in MODELS:
            title, d, rr = PROBE_LONG[m], full[0][1][m], [r[m] for _, r, _ in full[1:]]
            print(f"\nИТОГ на L12 ({title}):")
            for g in GROUPS:
                a, lo, hi, _ = d[g][-1]
                rmax = max(r[g][-1][0] for r in rr)
                if g == GROUPS[3]:     # положительный контроль: сам видит свой цвет, ~100% ожидается у ВСЕХ сетей
                    ok_c = a > 0.9 and all(r[g][-1][0] > 0.9 for r in rr)
                    print(f"  {NAMES[g]:26s} {ref} {a * 100:5.1f}%   случайные мин {min(r[g][-1][0] for r in rr) * 100:5.1f}%   "
                          f"контроль (у всех сетей > 90%): {'✅' if ok_c else '⚠️ проверь весь вывод'}")
                    continue
                print(f"  {NAMES[g]:26s} {ref} {a * 100:5.1f}% [{lo * 100:.1f}–{hi * 100:.1f}]   случайные макс {rmax * 100:5.1f}%   "
                      f"выше 50%: {'да ✅' if lo > 0.5 else 'нет'}   выше всех случайных: {'да ✅' if lo > rmax else 'нет'}")
        print(f"  (\"выше\" = нижняя граница 95% интервала {ref} больше этого числа)")

    # --- график
    L = np.arange(N_LAYERS)
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.3), sharey=True)
    for ax, g in zip(axes, GROUPS):
        ax.plot(L, [x[0] * 100 for x in dino[g]], "o-", c="C3", label=ref_long)
        ax.fill_between(L, [x[1] * 100 for x in dino[g]], [x[2] * 100 for x in dino[g]], color="C3", alpha=0.15)
        for i, (_, r, _) in enumerate(rand):
            ax.plot(L, [x[0] * 100 for x in r[g]], "o--", c="gray", alpha=0.6, ms=3,
                    label="случайные веса" if i == 0 else None)
        ax.axhline(50, ls=":", c="k", lw=1)
        ax.set_title(NAMES[g])
        ax.set_xticks(L, [f"L{l:02d}" for l in L], fontsize=7)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("точность, %  (50 = мешок признаков)")
    fig.suptitle(f"Проба «какого цвета {QWORD[QUERY]} близнец» ({PROBE_NAMES[MODELS[0]]}): {ref_long}" + (f" против {len(rand)} случайных сетей" if rand else ""), fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{STEM}.png", dpi=150)
    print(f"\nГрафик: {STEM}.png   Числа: {STEM}.csv")


if __name__ == "__main__":
    main()

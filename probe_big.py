"""
probe_big.py — проба привязки «левый» / «верхний» на MAE и I-JEPA (ViT-H/14). Добавлено 22.10.

ВОПРОС ОПЫТА:
  У DINO и CLIP горизонтальная привязка («какого цвета ЛЕВЫЙ близнец») слабела настолько,
  насколько сеть слепа к зеркалу слева-направо, а вертикальная («ВЕРХНИЙ близнец») — нет.
  Мера: разрыв «верхний − левый» (п.п.) на дальнем фоне, последний слой:
  DINO 18,6 при слепоте 0,28; CLIP 9,0 при 0,61 (реестр 13, 23, 25).
  Держится ли это на MAE (слепота 0,63) и I-JEPA (0,40, слепее)? Если да — разрыв у I-JEPA больше.

ЧТО ДЕЛАЕТ (та же проба, что probe_v2.py, — меняется только сеть и сетка патчей):
  Сцены data_v2 (2000), картинки S и Sp. ViT-H/14 режет 224×224 на 16×16 = 256 патчей по 14 px
  (у DINO/CLIP было 14×14 патчей по 16 px). Группы токенов — по маске пикселей, как в probe_v2:
    cls         — токен CLS (есть только у MAE; у I-JEPA CLS нет)
    нетронутые  — патчи остальных объектов (пиксели в S и Sp одинаковые)
    дальний фон — пустые патчи на расстоянии ≥ 3 патчей от обоих близнецов
                  (3 патча = 42 px; у DINO было 48 px — сетка у сети своя)
    близнец-«вопрос» — патчи самого левого (или верхнего) близнеца: ПОЛОЖИТЕЛЬНЫЙ КОНТРОЛЬ, ~100%
  Проба: StandardScaler + логрег C=0,1 (основная) и ридж alpha=1000 (проверка устойчивости,
  только у обученной сети), 5 фолдов по сценам (те же, что в probe_v2), бутстрэп по сценам.
  Уровень «мешка признаков» — ровно 50% (почему — см. probe_v2.py).

ЭКОНОМИЯ (полные активации ViT-H — ~90 ГБ на сеть):
  Сохраняем не все токены, а только СРЕДНЕЕ по каждой группе, и только на 9 слоях
  L00, L04, L08, …, L32 (L32 — последний, по нему правило вывода). ~0,5 ГБ на сеть,
  пишется в ~/binding-vit-data/probe_big/ (вне iCloud). Повторный запуск берёт готовое оттуда.

ПРАВИЛО ВЫВОДА (записано ДО запуска, реестр 39, 22.10; считает probe_big_compare.py):
  Разрыв = точность «верхний» − точность «левый», дальний фон, L32, логрег (п.п.).
  Δ = разрыв(I-JEPA) − разрыв(MAE), 95% интервал — парный бутстрэп по сценам (2000, seed 0).
  «Следует» — весь интервал Δ > 0; «противоречит» — весь < 0; «не различает» — задевает 0.
  Ставка Ерназара (реестр 39): MAE 8,8–9,0; I-JEPA 16,0–18,0 (его «примерно 17»); вывод «следует».

ВСТРОЕННЫЕ ПРОВЕРКИ:
  1. На L00 токены CLS / нетронутых / дальнего фона в S и Sp совпадают до бита -> точность РОВНО 50,0%.
  2. Пиксели патчей «нетронутые» и «дальний фон» в S и Sp совпадают (сверяется для каждой сцены).
  3. Прогон: наши слои = собственный прогон модели (как в flip_test.py).
  4. Отпечатки случайных сетей = те же, что в flip_test_mae/ijepa_log.txt (seed 0/1/2).
  5. Контроль (близнец-«вопрос») > 90% у всех сетей на L32.
  6. Число предупреждений «обучение не сошлось».

ЗАПУСК (из папки binding-vit; по одной команде; каждая ~1–1,5 ч):
  python probe_big.py --model mae 2>&1 | tee probe_big_mae_log.txt
  python probe_big.py --model ijepa 2>&1 | tee probe_big_ijepa_log.txt
  python probe_big_compare.py 2>&1 | tee probe_big_compare_log.txt
  -> probe_big_<model>.csv / .png / _scores.npz; файлы probe_v2*, flip_test* не трогаются.
"""

import argparse
import csv
import json
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import flip_test as ft
from extract import load_image, run_layers, check_forward, fingerprint, pick_device

DATA = "data_v2"
P, G = 14, 16                  # ViT-H/14: патч 14 px, сетка 16×16
FAR = 3                        # «дальний» фон: не ближе 3 патчей к обоим близнецам (как в probe_v2)
LAYERS = list(range(0, 33, 4))  # L00, L04, …, L32
C_REG, RIDGE_ALPHA = 0.1, 1000.0   # выбраны заранее (как в probe_v2)
N_FOLDS, N_BOOT = 5, 1000
BATCH = 16
QUERIES = ["left", "upper"]
QWORD = {"left": "левый", "upper": "верхний"}
STORE = ["cls", "unchanged", "far_bg", "twin0", "twin1"]   # что сохраняем: средние по группам
NAMES = {"cls": "CLS", "unchanged": "нетронутые", "far_bg": "дальний фон", "twin": "близнец-вопрос (контроль)"}
EXPECTED_FP = {0: 15517.9609, 1: 15495.1045, 2: 15534.0615}   # flip_test_mae/ijepa_log.txt (у обеих архитектур одинаковы)
LONG = {"mae": "обученный MAE (ViT-H/14)", "ijepa": "обученная I-JEPA (ViT-H/14, целевой энкодер)"}
SHORT = {"mae": "MAE", "ijepa": "I-JEPA"}


# ------------------------------------------------------------------
# 1. Группы патчей и ответы для каждой сцены (логика probe_v2.scene_info, сетка — параметр)
# ------------------------------------------------------------------
def scene_groups(rec, colors, p=P, g=G, check_pixels=True):
    """Возвращает: патчи (номера 0..g*g-1) групп unchanged / far_bg / twin0 / twin1
    и ответы для обоих вопросов: {query: (k_вопроса, {"S": (верный, соперник), "Sp": (...)})}."""
    sid = rec["id"]
    assert rec["twins"] == [0, 1], f"{sid}: близнецы не объекты 0 и 1"
    o = rec["objects"]
    mask = np.asarray(Image.open(f"{DATA}/masks/{sid}.png"))
    blk = mask.reshape(g, p, g, p).transpose(0, 2, 1, 3).reshape(g * g, p * p)   # [патч, пиксели]
    has0, has1 = (blk == 1).any(1), (blk == 2).any(1)      # в маске объект k помечен числом k+1
    twin = has0 | has1
    other = (blk >= 3).any(1)
    empty = (blk == 0).all(1)
    ty, tx = np.where(twin)[0] // g, np.where(twin)[0] % g
    dist = np.array([np.min(np.maximum(abs(q // g - ty), abs(q % g - tx))) for q in range(g * g)])
    groups = dict(unchanged=np.where(other & ~twin)[0], far_bg=np.where(empty & (dist >= FAR))[0],
                  twin0=np.where(has0 & ~has1)[0], twin1=np.where(has1 & ~has0)[0])
    if check_pixels:                  # проверка 2
        S = np.asarray(Image.open(f"{DATA}/images/{sid}_S.png").convert("RGB"))
        Sp = np.asarray(Image.open(f"{DATA}/images/{sid}_Sp.png").convert("RGB"))
        for k in ("unchanged", "far_bg"):
            for q in groups[k]:
                py, px = divmod(int(q), g)
                assert np.array_equal(S[py * p:(py + 1) * p, px * p:(px + 1) * p],
                                      Sp[py * p:(py + 1) * p, px * p:(px + 1) * p]), f"{sid}: патч {q} ({k}) в S и Sp различается"
    ci = {c: i for i, c in enumerate(colors)}
    answers = {}
    for query in QUERIES:
        # left: меньше x центра, при равенстве меньше y; upper: меньше y, при равенстве меньше x (как в probe_v2)
        if query == "left":
            k0, k1 = (o[0]["center"][0], o[0]["center"][1]), (o[1]["center"][0], o[1]["center"][1])
        else:
            k0, k1 = (o[0]["center"][1], o[0]["center"][0]), (o[1]["center"][1], o[1]["center"][0])
        q = 0 if k0 < k1 else 1
        oq, oo = o[q], o[1 - q]
        assert oq["color_Sp"] == oo["color"] and oo["color_Sp"] == oq["color"], f"{sid}: в Sp цвета не переставлены"
        answers[query] = (q, {"S": (ci[oq["color"]], ci[oo["color"]]), "Sp": (ci[oq["color_Sp"]], ci[oo["color_Sp"]])})
    return groups, answers


# ------------------------------------------------------------------
# 2. Извлечение: средние по группам на 9 слоях
# ------------------------------------------------------------------
@torch.no_grad()
def extract(model, meta, infos, device, arch, out_file, log_every=50):
    n_pre = getattr(model, "num_prefix_tokens", 1)
    D = model.embed_dim
    n_img = 2 * len(meta)
    F = np.zeros((len(LAYERS), n_img, len(STORE), D), dtype=np.float16)
    valid = np.zeros((n_img, len(STORE)), dtype=bool)
    paths = [f"{DATA}/images/{r['id']}_{v}.png" for r in meta for v in ("S", "Sp")]
    t0 = time.time()
    for s in range(0, n_img, BATCH):
        x = torch.stack([load_image(pth, arch) for pth in paths[s:s + BATCH]]).to(device)
        outs = run_layers(model, x)
        hs = [outs[l].float().cpu().numpy() for l in LAYERS]          # [B, T, D] на каждом слое
        for b in range(x.shape[0]):
            i = s + b
            grp = infos[i // 2][0]
            for gi, gname in enumerate(STORE):
                if gname == "cls":
                    idx = [0] if n_pre > 0 else []
                else:
                    idx = [int(q) + n_pre for q in grp[gname]]
                if not idx:
                    continue
                valid[i, gi] = True
                for li, h in enumerate(hs):
                    F[li, i, gi] = h[b, idx].mean(0)
        if (s // BATCH) % log_every == 0:
            print(f"    картинок {min(s + BATCH, n_img)}/{n_img}  ({time.time() - t0:.0f} с)", flush=True)
    np.savez(out_file, F=F, valid=valid, ids=np.array([r["id"] for r in meta]), layers=np.array(LAYERS),
             store=np.array(STORE))
    return F, valid


# ------------------------------------------------------------------
# 3. Проба (как probe_v2.run_probe)
# ------------------------------------------------------------------
def make_model(name):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=C_REG, max_iter=3000))
    return make_pipeline(StandardScaler(), RidgeClassifier(alpha=RIDGE_ALPHA))


def run_probe(X, y_true, y_rival, scene_idx, folds, models, n_scenes):
    """Балл каждой сцены (0 / 0,5 / 1 — среднее по S и Sp); NaN, если у сцены группа пустая."""
    scores = {m: np.zeros(len(X)) for m in models}
    n_warn = 0
    for f in range(N_FOLDS):
        test = folds[scene_idx] == f
        if test.sum() == 0:
            continue
        for m in models:
            clf = make_model(m)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)
                clf.fit(X[~test], y_true[~test])
                n_warn += sum(issubclass(x.category, ConvergenceWarning) for x in w)
            d = clf.predict_proba(X[test]) if m == "logreg" else clf.decision_function(X[test])
            col = {c: i for i, c in enumerate(clf.classes_)}
            # если цвет в обучении не встречался (бывает только на крошечных тестовых прогонах) — балл −∞
            pick = lambda ys: np.array([d[k, col[c]] if c in col else -np.inf for k, c in enumerate(ys)])
            pt, pr = pick(y_true[test]), pick(y_rival[test])
            scores[m][test] = np.where(pt > pr, 1.0, np.where(pt == pr, 0.5, 0.0))
    cnt = np.bincount(scene_idx, minlength=n_scenes)
    out = {}
    for m in models:
        ps = np.bincount(scene_idx, weights=scores[m], minlength=n_scenes) / np.maximum(cnt, 1)
        ps[cnt == 0] = np.nan
        out[m] = ps
    return out, n_warn


def boot(ps, rng, n=N_BOOT):
    ps = ps[~np.isnan(ps)]
    idx = rng.integers(0, len(ps), size=(n, len(ps)))
    b = ps[idx].mean(1)
    return float(ps.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), len(ps)


def probe_net(F, valid, infos, folds, models, has_cls):
    """Возвращает scores[query][group][layer_index][model] = балл по сценам; и число предупреждений."""
    n_sc = len(infos)
    groups = (["cls"] if has_cls else []) + ["unchanged", "far_bg", "twin"]
    scores = {q: {g: [] for g in groups} for q in QUERIES}
    n_warn = 0
    t0 = time.time()
    for li, l in enumerate(LAYERS):
        for q in QUERIES:
            for g in groups:
                X, yt, yr, sc = [], [], [], []
                for si, (_, answers) in enumerate(infos):
                    k, lab = answers[q]
                    gname = f"twin{k}" if g == "twin" else g
                    gi = STORE.index(gname)
                    for vi, v in enumerate(("S", "Sp")):
                        i = 2 * si + vi
                        if valid[i, gi]:
                            X.append(F[li, i, gi])
                            yt.append(lab[v][0]); yr.append(lab[v][1]); sc.append(si)
                X = np.asarray(X, dtype=np.float32)
                ps, nw = run_probe(X, np.array(yt), np.array(yr), np.array(sc), folds, models, n_sc)
                n_warn += nw
                scores[q][g].append(ps)
        print(f"    проба L{l:02d} готова  ({time.time() - t0:.0f} с)", flush=True)
    return scores, groups, n_warn


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mae", "ijepa"], required=True)
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--out", default=os.path.expanduser("~/binding-vit-data/probe_big"),
                    help="куда класть средние по группам (вне iCloud)")
    ap.add_argument("--limit", type=int, default=0, help="только для проверки в облаке: первые N сцен")
    ap.add_argument("--test-random-main", action="store_true",
                    help="только для проверки в облаке: вместо обученной сети — случайная seed 99")
    args = ap.parse_args()
    arch = args.model
    ft.ARCH = arch
    stem = f"probe_big_{arch}" + (f"_test{args.limit}" if args.limit else "") + ("_rmain" if args.test_random_main else "")
    os.makedirs(args.out, exist_ok=True)
    main_name = SHORT[arch]
    nets = [(main_name, 99 if args.test_random_main else None)] + [(f"{arch}_rand{s}", s) for s in range(args.n_random)]
    print(f"Опыт: проба привязки «левый» / «верхний» на {LONG[arch]} + {args.n_random} случайные сети   выход: {stem}.*")
    import timm
    print(f"timm {timm.__version__}, torch {torch.__version__}")
    if args.limit or args.test_random_main:
        print("⚠️ ПРОВЕРОЧНЫЙ прогон (облако): числа не для статьи.")
    t0 = time.time()

    meta = [json.loads(line) for line in open(f"{DATA}/meta.jsonl")]
    if args.limit:
        meta = meta[:args.limit]
    colors = sorted({ob[k] for r in meta for ob in r["objects"] for k in ("color", "color_Sp")})
    infos = [scene_groups(r, colors) for r in meta]
    n_g = {k: sum(1 for gr, _ in infos if len(gr[k])) for k in ("unchanged", "far_bg", "twin0", "twin1")}
    nfar = [len(gr["far_bg"]) for gr, _ in infos]
    print(f"Сцен: {len(meta)}, картинок S+Sp: {2 * len(meta)}. Цвета ({len(colors)}): {', '.join(colors)}")
    print(f"Сетка {G}×{G} патчей по {P} px. Сцен с непустой группой: нетронутые {n_g['unchanged']}, "
          f"дальний фон {n_g['far_bg']} (патчей в среднем {np.mean(nfar):.1f}), близнец 0 {n_g['twin0']}, близнец 1 {n_g['twin1']}")
    print("Проверка 2 ✅ пиксели «нетронутых» и «дальнего фона» в S и Sp совпадают во всех сценах")
    print(f"Слои: {', '.join(f'L{l:02d}' for l in LAYERS)}.  Проба: логрег C={C_REG} (основная); ридж alpha={RIDGE_ALPHA:g} "
          f"(устойчивость, только {main_name}); {N_FOLDS} фолдов по сценам. Уровень случайности = 50%")
    # фолды — как в probe_v2: перестановка всех сцен с seed 0
    folds = np.random.default_rng(0).permutation(len(meta)) % N_FOLDS

    device = pick_device()
    all_scores, weights_of, warn_total, fps = {}, {}, 0, {}
    for name, seed in nets:
        cache = os.path.join(args.out, f"{arch}_{'main' if seed is None else 'seed' + str(seed)}"
                                       f"{'_test' + str(args.limit) if args.limit else ''}.npz")
        if seed is None:
            model, weights = ft.load_model(None)
        else:
            model, weights = ft.load_model(seed)
            if name == main_name:
                weights = "ПРОВЕРКА: " + weights
        model.eval()
        fp = fingerprint(model)
        fps[name] = fp
        weights_of[name] = weights
        note = ""
        if seed in EXPECTED_FP:
            note = "✅ совпадает с flip_test" if abs(fp - EXPECTED_FP[seed]) < 0.01 else "⚠️ НЕ совпадает — проверь весь вывод"
        print(f"\n### {name}: {weights}   отпечаток весов {fp:.4f}  {note}", flush=True)
        has_cls = getattr(model, "num_prefix_tokens", 1) > 0
        assert len(model.blocks) + 1 == 33, "ожидалась ViT-H/14 с 32 блоками"
        if os.path.exists(cache):
            z = np.load(cache)
            ok = (list(z["ids"]) == [r["id"] for r in meta] and list(z["layers"]) == LAYERS)
            assert ok, f"{cache}: не тот набор сцен/слоёв — удали файл и запусти снова"
            F, valid = z["F"], z["valid"]
            print(f"  средние по группам взяты из {cache} (уже сняты раньше)")
        else:
            model.to(device)
            xx = torch.stack([load_image(f"{DATA}/images/{meta[0]['id']}_S.png", arch),
                              load_image(f"{DATA}/images/{meta[0]['id']}_Sp.png", arch)]).to(device)
            check_forward(model, xx, run_layers(model, xx))        # проверка 3
            print(f"  снимаю средние по группам -> {cache}", flush=True)
            F, valid = extract(model, meta, infos, device, arch, cache)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        models = ["logreg", "ridge"] if seed is None or name == main_name else ["logreg"]
        sc, groups, nw = probe_net(F, valid, infos, folds, models, has_cls)
        warn_total += nw
        all_scores[name] = sc
        print(f"  готово ({time.time() - t0:.0f} с), предупреждений «не сошлось»: {nw}", flush=True)

    # ---------------- отчёт ----------------
    rng = np.random.default_rng(0)
    rand = [n for n, s in nets[1:]]
    groups = list(all_scores[main_name]["left"].keys())
    rows = []
    res = {}
    for name in all_scores:
        for q in QUERIES:
            for g in all_scores[name][q]:
                for li, l in enumerate(LAYERS):
                    for m, ps in all_scores[name][q][g][li].items():
                        acc, lo, hi, n = boot(ps, rng)
                        res[(name, q, g, l, m)] = (acc, lo, hi, n)
                        rows.append([name, weights_of[name], m, q, g, f"L{l:02d}", f"{acc:.5f}", f"{lo:.5f}", f"{hi:.5f}", n])
    with open(f"{stem}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["net", "weights", "probe", "query", "group", "layer", "acc", "ci_lo", "ci_hi", "n_scenes"])
        w.writerows(rows)

    # проверка 1: L00 ровно 50% (CLS, нетронутые, дальний фон — у всех сетей и проб)
    l00 = [res[(n, q, g, 0, m)][0] for n in all_scores for q in QUERIES for g in groups if g != "twin"
           for m in all_scores[n][q][g][0]]
    ok1 = all(abs(x - 0.5) < 1e-12 for x in l00)
    print(f"\nПроверка 1 (L00: CLS / нетронутые / дальний фон ровно 50.0%): {'✅' if ok1 else '⚠️ проверь весь вывод'}  "
          f"{sorted(set(round(x * 100, 3) for x in l00))}")

    for q in QUERIES:
        print(f"\n==== Вопрос «какого цвета {QWORD[q]} близнец», логрег (50% = мешок признаков) ====")
        for g in groups:
            n_sc = res[(main_name, q, g, LAYERS[-1], 'logreg')][3]
            print(f"\n== {NAMES[g]} ==   сцен: {n_sc}")
            print(f"слой   {main_name} [95% инт.]       ридж     случайные: среднее (мин–макс)")
            for l in LAYERS:
                a, lo, hi, _ = res[(main_name, q, g, l, "logreg")]
                rd = res.get((main_name, q, g, l, "ridge"))
                rr = [res[(n, q, g, l, "logreg")][0] for n in rand]
                rs = f"{np.mean(rr) * 100:5.1f}% ({min(rr) * 100:.1f}–{max(rr) * 100:.1f})" if rr else "—"
                print(f"L{l:02d}  {a * 100:5.1f}% [{lo * 100:5.1f}–{hi * 100:5.1f}]   {rd[0] * 100 if rd else float('nan'):5.1f}%   {rs}")

    print(f"\n==== Разрыв «верхний − левый» (п.п.), логрег; 95% инт. — парный бутстрэп по сценам ====")
    gaps = {}
    for g in groups:
        if g == "twin":
            continue
        print(f"\n== {NAMES[g]} ==")
        print(f"слой   {main_name} [95% инт.]          случайные (мин–макс)")
        for li, l in enumerate(LAYERS):
            def gap_of(n, n_boot=N_BOOT):
                d = all_scores[n]["upper"][g][li]["logreg"] - all_scores[n]["left"][g][li]["logreg"]
                return boot(d, np.random.default_rng(0), n_boot)
            a, lo, hi, _ = gap_of(main_name)
            gaps[(g, l)] = (a, lo, hi)
            rr = [gap_of(n)[0] * 100 for n in rand]
            rs = f"{min(rr):5.1f} … {max(rr):5.1f}" if rr else "—"
            print(f"L{l:02d}  {a * 100:5.1f} [{lo * 100:5.1f} … {hi * 100:5.1f}]     {rs}")

    print(f"\nПредупреждений «обучение не сошлось» всего: {warn_total}  {'✅' if warn_total == 0 else '⚠️ проверь весь вывод'}")
    print(f"\nИТОГ на L{LAYERS[-1]:02d} (логрег; «выше» = нижняя граница 95% инт. {main_name} больше числа):")
    for q in QUERIES:
        for g in groups:
            a, lo, hi, _ = res[(main_name, q, g, LAYERS[-1], "logreg")]
            rr = [res[(n, q, g, LAYERS[-1], "logreg")][0] for n in rand]
            if g == "twin":
                okc = a > 0.9 and all(x > 0.9 for x in rr)
                print(f"  {QWORD[q]:8s} {NAMES[g]:28s} {main_name} {a * 100:5.1f}%   случайные мин {min(rr) * 100 if rr else float('nan'):5.1f}%   "
                      f"контроль (у всех > 90%): {'✅' if okc else '⚠️ проверь весь вывод'}")
                continue
            print(f"  {QWORD[q]:8s} {NAMES[g]:28s} {main_name} {a * 100:5.1f}% [{lo * 100:.1f}–{hi * 100:.1f}]   "
                  f"случайные макс {max(rr) * 100 if rr else float('nan'):5.1f}%   выше 50%: {'да ✅' if lo > 0.5 else 'нет'}   "
                  f"выше всех случайных: {'да ✅' if rr and lo > max(rr) else 'нет'}")
    a, lo, hi = gaps[("far_bg", LAYERS[-1])]
    print(f"\nРазрыв «верхний − левый», дальний фон, L{LAYERS[-1]:02d}: {main_name} {a * 100:.1f} [{lo * 100:.1f} … {hi * 100:.1f}] п.п.  "
          f"(для сравнения, реестр 25: DINO 18,6; CLIP 9,0 — их последний слой L12)")
    print(f"Вывод по правилу (MAE против I-JEPA) — после обоих прогонов: python probe_big_compare.py")

    np.savez_compressed(f"{stem}_scores.npz", ids=np.array([r["id"] for r in meta]), layers=np.array(LAYERS),
                        nets=np.array(list(all_scores)), groups=np.array(groups), queries=np.array(QUERIES),
                        weights=np.array([weights_of[n] for n in all_scores]),
                        logreg=np.stack([[[np.stack([all_scores[n][q][g][li]["logreg"] for li in range(len(LAYERS))])
                                           for g in groups] for q in QUERIES] for n in all_scores]))
    # график: точность «левый» и «верхний» по слоям, дальний фон и CLS/нетронутые
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4.2), sharey=True, squeeze=False)
    for ax, g in zip(axes[0], groups):
        for q, c in (("left", "C3"), ("upper", "C0")):
            y = [res[(main_name, q, g, l, "logreg")] for l in LAYERS]
            ax.plot(LAYERS, [v[0] * 100 for v in y], "o-", c=c, label=f"{main_name}: {QWORD[q]}")
            ax.fill_between(LAYERS, [v[1] * 100 for v in y], [v[2] * 100 for v in y], color=c, alpha=0.15)
            for i, n in enumerate(rand):
                ax.plot(LAYERS, [res[(n, q, g, l, "logreg")][0] * 100 for l in LAYERS], "--", c=c, alpha=0.35, lw=1,
                        label=f"случайные: {QWORD[q]}" if i == 0 else None)
        ax.axhline(50, ls=":", c="k", lw=1)
        ax.set_title(NAMES[g])
        ax.set_xticks(LAYERS, [f"L{l:02d}" for l in LAYERS], fontsize=7)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0][0].set_ylabel("точность, %  (50 = мешок признаков)")
    fig.suptitle(f"Проба «левый» / «верхний» близнец (логрег): {LONG[arch]} против {len(rand)} случайных сетей", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=150)
    print(f"\nГрафик: {stem}.png   Числа: {stem}.csv, {stem}_scores.npz   ({time.time() - t0:.0f} с)")


if __name__ == "__main__":
    main()

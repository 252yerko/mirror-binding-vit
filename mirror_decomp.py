"""
mirror_decomp.py — опыт «A + D»: зеркальное разложение представления и нелинейная проба. Добавлено 13.11.

ВОПРОС ОПЫТА:
  Знание «кто слева» может жить только в той части представления, которая меняется при отражении
  слева-направо (у картинки и её отражения ответ противоположный). Почему у слепых к зеркалу сетей
  «левый» читается хуже «верхнего»:
    — МАСКИРОВКА: сторона в сети есть, но тонет в одинаковом для картинки и отражения содержании;
    — СТИРАНИЕ: сторона действительно потеряна.
  И (D): не артефакт ли разрыв «верхний − левый» линейной пробы.

РАЗЛОЖЕНИЕ (для картинки x и её отражения mx; h(·) — среднее по группе токенов на слое):
  симметричная часть  s = (h(x) + h(mx)) / 2,   несимметричная  a = (h(x) − h(mx)) / 2.
  Для «левого» mx — отражение слева-направо, для «верхнего» — сверху-вниз.
  Группа «дальний фон» у mx — отражённый набор патчей (пустые патчи ≥ 3 патчей от обоих близнецов);
  проверяется пересчётом по отражённой маске.

СЕТИ (ViT-B): DINO ViT-B/16, CLIP ViT-B/16 (OpenAI), DINOv2 ViT-B/14 с регистрами, DeiT III ViT-B/16
  (те же загрузчики и нормировка, что в extract.py / more_nets.py) + по одной случайной сети (seed 0).

СЦЕНЫ:
  Правила — сцены data_v2 только из кругов / квадратов / крестов (фигуры не меняются ни при одном отражении),
  без сцен, где у близнецов равны x или y (иначе «левый» / «верхний» у отражения не определён).
  Для D (обычные картинки, отражения не нужны) — все 2000 сцен.
  Фолды — те же, что везде: перестановка всех 2000 сцен с seed 0, % 5.

ПРОБЫ (StandardScaler + логрег C = 0,1, max_iter 3000; балл картинки 1 / 0,5 / 0; балл сцены — среднее по S и Sp):
  обычная     — учится и проверяется на h(x) (как раньше);
  та же на отражениях — обычная проба (обучена на h(x)) проверяется на h(mx) тех же тестовых сцен; верный ответ
                у mx — цвет другого близнеца. Симметричная подсказка даёт верный ответ на x и неверный на mx;
  честная     — (справка) учится на h(x) ИЛИ h(mx) (случайно, seed 0; столько же картинок), проверяется на x и mx;
  несимм.     — учится и проверяется на a;
  симм.       — учится на s; проверка: на паре (x, mx) признаки одинаковые, метки противоположные ->
                по построению ровно 50% (проверка кода, не результат).
  MLP (D)     — StandardScaler + MLPClassifier(256 нейронов, adam, max_iter 200, random_state 0); сила L2 (alpha)
                из {0,01; 0,1; 1; 10} выбирается в каждом внешнем фолде по вложенной проверке (3 фолда по сценам)
                ТОЛЬКО на вопросе «верхний»; та же alpha — для «левого». Все 2000 сцен, дальний фон, L12.

ПРАВИЛА (записаны ДО запуска, реестр 129–130; каждая обученная сеть; дальний фон, L12):
  A1 (подсказки): Δ = точность обычной пробы на отражениях − на исходных картинках («левый»), парный бутстрэп
     по сценам (1000, seed 0). [Изменено 13.11 до регистрации: в дизайне, показанном до ставки, была «честная − обычная»;
     облачный тест на искусственных признаках показал, что смешанная проба, проверяемая только на исходных, симметричную
     подсказку не ловит (97,7% при подсказке), а проверка той же пробы на отражениях ловит.]
     Верхняя граница < −2 п.п. -> «часть сигнала — симметричные подсказки»; нижняя > −2 п.п. -> «подсказок нет»;
     иначе «не решено».
  A2 (маскировка / стирание): разрыв_полный = обычная(верхний) − обычная(левый);
     разрыв_несимм = несимм(верхний, a ↑↓) − несимм(левый, a ←→); R = разрыв_несимм / разрыв_полный,
     95% интервал — бутстрэп по сценам (1000, seed 0). Применимо, если нижняя граница разрыва_полного > 0.
     Верхняя граница R < 0,5 -> «в основном маскировка»; нижняя > 0,5 -> «в основном стирание»; иначе «смешано».
  D (нелинейная проба): допуск — нижняя граница (MLP − логрег) на «верхнем» > −1 п.п.; иначе «не информативно».
     При допуске: нижняя граница разрыва MLP > 0 -> «разрыв не артефакт линейной пробы»; иначе «не показано».
  Ставки Ерназара (реестр 130, 13.11 11:42): 130a DINO A1 «подсказок нет»; 130b DINO A2 «в основном стирание»;
     130c CLIP A2 не «в основном стирание»; 130d DINO разрыв MLP 10,0–25,0 п.п. (точка, округление 0,1;
     если D «не информативно» — не засчитывается); 130e выводы A2 у DINO и CLIP разные (считает --summary).

СПРАВКА (не правила): все слои; CLS; доля несимметричной части E = ср.||a||² / ср.||h(x) − среднее||²;
  проба на «верхнем» с честной пробой; случайные сети.

ВСТРОЕННЫЕ ПРОВЕРКИ:
  1. L00: все пробы на CLS и дальнем фоне — ровно 50,0%.
  2. Отражения: пиксели mx = зеркальный поворот x; группы mx = группы, пересчитанные по отражённой маске;
     пиксели дальнего фона в S и Sp совпадают (как раньше).
  3. Наш пошаговый прогон = собственный прогон модели (макс. разница < 1e-3).
  4. Пустая серая картинка: a = 0 (наибольшая разница < 1e-4).
  5. Симм. проба: на каждой паре (x, mx) баллы в сумме ровно 1 -> 50,0%.
  6. Контроль «близнец-вопрос» > 90% на L12 (обычная проба, все 2000 сцен).
  7. Случайная сеть отличается от обученной (отпечаток весов).
  8. Нормировка, сетка, служебные токены, 12 блоков.
  9. Обычная проба (все 2000 сцен, дальний фон, L12) совпадает с прежними числами в пределах 0,5 п.п.

ЗАПУСК (из папки binding-vit; по одной команде):
  caffeinate -i python mirror_decomp.py --model dino   2>&1 | tee mirror_decomp_dino_log.txt
  caffeinate -i python mirror_decomp.py --model clip   2>&1 | tee mirror_decomp_clip_log.txt
  caffeinate -i python mirror_decomp.py --model dinov2 2>&1 | tee mirror_decomp_dinov2_log.txt
  caffeinate -i python mirror_decomp.py --model deit3  2>&1 | tee mirror_decomp_deit3_log.txt
  python mirror_decomp.py --summary 2>&1 | tee mirror_decomp_summary_log.txt
  -> mirror_decomp_<model>.csv; средние по группам — в ~/binding-vit-data/mirror_decomp/ (повторный запуск берёт готовое).
"""

import argparse
import csv
import json
import os
import time
import warnings

import numpy as np
import torch
from PIL import Image
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import flip_test as ft
import probe_big as pb
import more_nets as mn
from extract import load_image, run_layers, fingerprint, pick_device, load_model as ex_load_model

DATA = "data_v2"
LAYERS = list(range(13))
N_FOLDS, N_BOOT, C_REG = 5, 1000, 0.1
MLP_ALPHAS = [0.01, 0.1, 1.0, 10.0]
BATCH = 16
MIR = {"left": "h", "upper": "v"}          # какое отражение меняет ответ на вопрос
QWORD = {"left": "левый", "upper": "верхний"}
CONF = {
    "dino":   dict(short="DINO", P=16, G=14, n_pre=1, norm="dino"),
    "clip":   dict(short="CLIP", P=16, G=14, n_pre=1, norm="clip"),
    "dinov2": dict(short="DINOv2", P=14, G=16, n_pre=5, norm="dinov2"),
    "deit3":  dict(short="DeiT III", P=16, G=14, n_pre=1, norm="deit3"),
}
# прежние числа «обычной» пробы (все 2000 сцен, дальний фон, L12): файл, где взять (проверка 9)
OLD = {"dino": ("probe_v2.csv", "probe_v2_upper.csv"), "clip": ("probe_v2_clip.csv", "probe_v2_clip_upper.csv"),
       "dinov2": ("more_nets_dinov2.csv",), "deit3": ("more_nets_deit3.csv",)}
BETS = {"dino": dict(a1="подсказок нет", a2="в основном стирание", d=(10.0, 25.0)),
        "clip": dict(a2_not="в основном стирание")}


# ------------------------------------------------------------------
# загрузка сети
# ------------------------------------------------------------------
def load_net(arch, seed):
    if arch in ("dino", "clip"):
        m, w, _ = ex_load_model(arch, seed)
    else:
        m, w = mn.load_net(arch, seed)
    m.eval()
    return m, w


def check_setup(model, arch):
    c = CONF[arch]
    n_pre = getattr(model, "num_prefix_tokens", 1)
    pe = model.patch_embed
    grid = tuple(pe.grid_size) if hasattr(pe, "grid_size") else (224 // pe.patch_size,) * 2
    ok = grid == (c["G"], c["G"]) and n_pre == c["n_pre"] and len(model.blocks) == 12
    if arch in ("dinov2", "deit3"):
        ok &= mn.check_setup(model, arch)
    print(f"  Проверка 8: сетка {grid}; служебных токенов {n_pre}; блоков {len(model.blocks)}  {'✅' if ok else '⚠️ проверь весь вывод'}")
    return ok


def forward_check(model, x):
    with torch.no_grad():
        ref = model(x)
        last = run_layers(model, x)[-1]
        mine = model.norm(last)[:, 0] if hasattr(model, "prepare_tokens") else model.forward_head(model.norm(last))
    return float((mine - ref).abs().max())


# ------------------------------------------------------------------
# группы патчей: из маски (для проверки отражений)
# ------------------------------------------------------------------
def groups_from_mask(mask, P, G):
    blk = mask.reshape(G, P, G, P).transpose(0, 2, 1, 3).reshape(G * G, P * P)
    has0, has1 = (blk == 1).any(1), (blk == 2).any(1)
    twin = has0 | has1
    empty = (blk == 0).all(1)
    ty, tx = np.where(twin)[0] // G, np.where(twin)[0] % G
    dist = np.array([np.min(np.maximum(abs(q // G - ty), abs(q % G - tx))) for q in range(G * G)])
    return dict(far_bg=np.where(empty & (dist >= pb.FAR))[0], twin0=np.where(has0 & ~has1)[0], twin1=np.where(has1 & ~has0)[0])


def mirror_idx(idx, G, kind):
    r, c = np.asarray(idx) // G, np.asarray(idx) % G
    return np.sort(r * G + (G - 1 - c)) if kind == "h" else np.sort((G - 1 - r) * G + c)


def scene_info(rec, colors, P, G):
    """Группы для x, mx(←→), mx(↑↓) и ответы. Проверка 2 — здесь."""
    grp, ans = pb.scene_groups(rec, colors, p=P, g=G, check_pixels=True)
    mask = np.asarray(Image.open(f"{DATA}/masks/{rec['id']}.png"))
    base = groups_from_mask(mask, P, G)
    for k in base:
        assert np.array_equal(np.sort(base[k]), np.sort(grp[k])), f"{rec['id']}: группа {k} не совпала с probe_big"
    out = {"o": {k: np.sort(grp[k]) for k in ("far_bg", "twin0", "twin1")}}
    for kind, fl in (("h", mask[:, ::-1]), ("v", mask[::-1, :])):
        gm = groups_from_mask(np.ascontiguousarray(fl), P, G)
        for k in ("far_bg", "twin0", "twin1"):
            mi = mirror_idx(out["o"][k], G, kind)
            assert np.array_equal(mi, np.sort(gm[k])), f"{rec['id']}: отражённая группа {k} ({kind}) не совпала с маской"
        out[kind] = {k: mirror_idx(out["o"][k], G, kind) for k in ("far_bg", "twin0", "twin1")}
    return out, ans


# ------------------------------------------------------------------
# извлечение средних по группам
# ------------------------------------------------------------------
@torch.no_grad()
def extract(model, recs, infos, mirrors, layers, device, arch, n_pre):
    """Возвращает F[слои, сцены, 2 версии, len(mirrors), 2 группы (cls, far_bg), D] и
    T[сцены, 2 версии, 2 близнеца, D] (L12, только исходная картинка, контроль)."""
    D = model.embed_dim
    F = np.zeros((len(layers), len(recs), 2, len(mirrors), 2, D), dtype=np.float32)
    T = np.zeros((len(recs), 2, 2, D), dtype=np.float32)
    jobs = [(si, vi, mi) for si in range(len(recs)) for vi in range(2) for mi in range(len(mirrors))]
    t0 = time.time()
    for s in range(0, len(jobs), BATCH):
        part = jobs[s:s + BATCH]
        xs = []
        for si, vi, mi in part:
            x = load_image(f"{DATA}/images/{recs[si]['id']}_{('S', 'Sp')[vi]}.png", CONF[arch]["norm"])
            k = mirrors[mi]
            xs.append(x if k == "o" else (x.flip(2) if k == "h" else x.flip(1)))   # [3,H,W]: W — ось 2, H — ось 1
        outs = run_layers(model, torch.stack(xs).to(device))
        hs = [outs[l].float().cpu().numpy() for l in layers]
        h12 = outs[12].float().cpu().numpy()
        for b, (si, vi, mi) in enumerate(part):
            g = infos[si][0][mirrors[mi]]
            idx = [int(q) + n_pre for q in g["far_bg"]]
            for li, h in enumerate(hs):
                F[li, si, vi, mi, 0] = h[b, 0]
                F[li, si, vi, mi, 1] = h[b, idx].mean(0)
            if mirrors[mi] == "o":
                for t in range(2):
                    T[si, vi, t] = h12[b, [int(q) + n_pre for q in g[f"twin{t}"]]].mean(0)
        if (s // BATCH) % 50 == 0:
            print(f"    картинок {min(s + BATCH, len(jobs))}/{len(jobs)}  ({time.time() - t0:.0f} с)", flush=True)
    return F, T


# ------------------------------------------------------------------
# пробы
# ------------------------------------------------------------------
def make_clf(kind, alpha=None):
    if kind == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256,), alpha=alpha, max_iter=200,
                                                             random_state=0))
    return make_pipeline(StandardScaler(), LogisticRegression(C=C_REG, max_iter=3000))


def fit(clf, X, y):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(X, y)
        return sum(issubclass(x.category, ConvergenceWarning) for x in w)


def score(clf, X, yt, yr):
    d = clf.predict_proba(X)
    col = {c: i for i, c in enumerate(clf.classes_)}
    pick = lambda ys: np.array([d[k, col[c]] if c in col else -np.inf for k, c in enumerate(ys)])
    pt, pr = pick(yt), pick(yr)
    return np.where(pt > pr, 1.0, np.where(pt == pr, 0.5, 0.0))


def per_scene(sc, scene, n):
    cnt = np.bincount(scene, minlength=n)
    out = np.bincount(scene, weights=sc, minlength=n) / np.maximum(cnt, 1)
    out[cnt == 0] = np.nan
    return out


def probes_A(Fl, labs, folds, rng_seed=0):
    """Fl: [сцены, 2 версии, 3 отражения (o,h,v), D] для одной группы и слоя. labs[q]: [сцены, 2 версии, 2] (верный, соперник).
    Возвращает баллы по сценам для обычной / честной / несимм. / симм.-пары и число предупреждений."""
    n = Fl.shape[0]
    res, nw = {}, 0
    sc_idx = np.repeat(np.arange(n), 2)
    for q in ("left", "upper"):
        mi = 1 if MIR[q] == "h" else 2
        Xo = Fl[:, :, 0].reshape(2 * n, -1)
        Xm = Fl[:, :, mi].reshape(2 * n, -1)
        yt = labs[q][:, :, 0].reshape(-1)
        yr = labs[q][:, :, 1].reshape(-1)
        pick_m = np.random.default_rng(rng_seed).random(2 * n) < 0.5     # честная: какие обучающие картинки — отражения
        A = (Xo - Xm) / 2
        S = (Xo + Xm) / 2
        out = {k: np.zeros(2 * n) for k in ("normal", "normal_m", "honest", "anti")}
        pair_ok, sym_sc = True, np.zeros(2 * n)
        for f in range(N_FOLDS):
            te = folds[sc_idx] == f
            tr = ~te
            if te.sum() == 0:
                continue
            c = make_clf("logreg"); nw += fit(c, Xo[tr], yt[tr]); out["normal"][te] = score(c, Xo[te], yt[te], yr[te])
            out["normal_m"][te] = score(c, Xm[te], yr[te], yt[te])          # та же проба на отражениях: верный ответ — цвет другого близнеца
            Xh = np.where(pick_m[:, None], Xm, Xo)
            yh = np.where(pick_m, yr, yt)
            c = make_clf("logreg"); nw += fit(c, Xh[tr], yh[tr])
            out["honest"][te] = (score(c, Xo[te], yt[te], yr[te]) + score(c, Xm[te], yr[te], yt[te])) / 2
            c = make_clf("logreg"); nw += fit(c, A[tr], yt[tr]); out["anti"][te] = score(c, A[te], yt[te], yr[te])
            c = make_clf("logreg"); nw += fit(c, S[tr], yt[tr])
            s1, s2 = score(c, S[te], yt[te], yr[te]), score(c, S[te], yr[te], yt[te])
            pair_ok &= bool(np.all(s1 + s2 == 1.0))
            sym_sc[te] = (s1 + s2) / 2
        for k in out:
            res[(q, k)] = per_scene(out[k], sc_idx, n)
        res[(q, "sym_pair")] = per_scene(sym_sc, sc_idx, n)
        res[(q, "pair_ok")] = pair_ok
    return res, nw


def probe_plain(X, yt, yr, folds, kind="logreg", alphas=None):
    """X: [сцены, 2, D]. Обычная проба (логрег или MLP). Для MLP alphas[f] — сила L2 во внешнем фолде f."""
    n = X.shape[0]
    sc_idx = np.repeat(np.arange(n), 2)
    Xf, ytf, yrf = X.reshape(2 * n, -1), yt.reshape(-1), yr.reshape(-1)
    out, nw = np.zeros(2 * n), 0
    for f in range(N_FOLDS):
        te = folds[sc_idx] == f
        if te.sum() == 0:
            continue
        c = make_clf(kind, None if alphas is None else alphas[f])
        nw += fit(c, Xf[~te], ytf[~te])
        out[te] = score(c, Xf[te], ytf[te], yrf[te])
    return per_scene(out, sc_idx, n), nw


def choose_alphas(X, yt, yr, folds):
    """Для каждого внешнего фолда — alpha MLP по вложенной проверке (3 фолда по сценам) только на «верхнем»."""
    n = X.shape[0]
    chosen = {}
    for f in range(N_FOLDS):
        tr_sc = np.where(folds != f)[0]
        inner = np.random.default_rng(f).permutation(len(tr_sc)) % 3
        best, best_a = -1, None
        for a in MLP_ALPHAS:
            accs = []
            for k in range(3):
                trs, tes = tr_sc[inner != k], tr_sc[inner == k]
                c = make_clf("mlp", a)
                fit(c, X[trs].reshape(-1, X.shape[-1]), yt[trs].reshape(-1))
                accs.append(score(c, X[tes].reshape(-1, X.shape[-1]), yt[tes].reshape(-1), yr[tes].reshape(-1)).mean())
            if np.mean(accs) > best:
                best, best_a = float(np.mean(accs)), a
        chosen[f] = best_a
    return chosen


def boot(ps, rng, n=N_BOOT):
    ok = ~np.isnan(ps)
    ps = ps[ok]
    idx = rng.integers(0, len(ps), size=(n, len(ps)))
    b = ps[idx].mean(1)
    return float(ps.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), len(ps)


def boot_ratio(num, den, rng, n=N_BOOT):
    ok = ~(np.isnan(num) | np.isnan(den))
    num, den = num[ok], den[ok]
    idx = rng.integers(0, len(num), size=(n, len(num)))
    b = num[idx].mean(1) / den[idx].mean(1)
    return float(num.mean() / den.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), len(num)


# ------------------------------------------------------------------
def run(args):
    arch, c = args.model, CONF[args.model]
    P, G, n_pre = c["P"], c["G"], c["n_pre"]
    test = bool(args.limit or args.test_random_main)
    stem = f"mirror_decomp_{arch}" + (f"_test{args.limit}" if args.limit else "") + ("_rmain" if args.test_random_main else "")
    main_name = c["short"]
    nets = [(main_name, 99 if args.test_random_main else None)] + [(f"{arch}_rand{s}", s) for s in range(args.n_random)]
    import timm
    print(f"Опыт A + D: зеркальное разложение и нелинейная проба — {main_name} + {args.n_random} случайная   выход: {stem}.csv")
    print(f"timm {timm.__version__}, torch {torch.__version__}")
    if test:
        print("⚠️ ПРОВЕРОЧНЫЙ прогон (облако): числа не для статьи.")
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    pb.FAR = 3

    meta = [json.loads(line) for line in open(f"{DATA}/meta.jsonl")]
    if args.limit:
        meta = meta[:args.limit]
    colors = sorted({ob[k] for r in meta for ob in r["objects"] for k in ("color", "color_Sp")})
    folds_all = np.random.default_rng(0).permutation(len(meta)) % N_FOLDS
    infos = [scene_info(r, colors, P, G) for r in meta]

    def distinct(r):
        a, b = r["objects"][0]["center"], r["objects"][1]["center"]
        return a[0] != b[0] and a[1] != b[1]
    rule_i = [i for i, r in enumerate(meta) if all(o["shape"] in ft.SYM_SHAPES for o in r["objects"]) and distinct(r)]
    rest_i = [i for i in range(len(meta)) if i not in set(rule_i)]
    print(f"Сцен всего {len(meta)}; для правил (круг / квадрат / крест, у близнецов разные x и y): {len(rule_i)}")
    print("Проверка 2 ✅ пиксели дальнего фона в S и Sp совпадают; группы отражений = группы по отражённой маске (все сцены)")

    # метки
    def labels(idx_list):
        L = {}
        for q in ("left", "upper"):
            L[q] = np.array([[infos[i][1][q][1][v] for v in ("S", "Sp")] for i in idx_list])   # [сцены, 2, 2]
        return L
    lab_rule, lab_all = labels(rule_i), labels(range(len(meta)))
    twin_q = {q: np.array([infos[i][1][q][0] for i in range(len(meta))]) for q in ("left", "upper")}

    x0 = load_image(f"{DATA}/images/{meta[rule_i[0]]['id']}_S.png", c["norm"])
    blank = x0.clone()[None]
    blank[:] = blank[0, :, :1, :1]

    device = pick_device()
    rows, verdicts, warn_total, fps, ok = [], {}, 0, {}, {"3": True, "4": True, "8": True, "1": True, "5": True, "6": True, "9": None}
    store = {}
    for name, seed in nets:
        model, weights = load_net(arch, seed)
        if name == main_name and seed is not None:
            weights = "ПРОВЕРКА: " + weights
        fps[name] = fingerprint(model)
        print(f"\n### {name}: {weights}   отпечаток весов {fps[name]:.4f}", flush=True)
        ok["8"] &= check_setup(model, arch)
        model.to(device)
        xx = torch.stack([load_image(f"{DATA}/images/{meta[0]['id']}_{v}.png", c["norm"]) for v in ("S", "Sp")]).to(device)
        d3 = forward_check(model, xx)
        ok["3"] &= d3 < 1e-3
        print(f"  Проверка 3 (наши слои = прогон самой модели): макс. разница {d3:.2e}  {'✅' if d3 < 1e-3 else '⚠️ проверь весь вывод'}")
        with torch.no_grad():
            fb = [run_layers(model, b.to(device)) for b in (blank, blank.flip(3), blank.flip(2))]
        z = max(float((fb[0][l] - fb[k][l]).abs().max()) for l in LAYERS for k in (1, 2))
        ok["4"] &= z < 1e-4
        print(f"  Проверка 4 (пустая картинка: отражение ничего не меняет, a = 0): макс. разница {z:.2e}  {'✅' if z < 1e-4 else '⚠️ проверь весь вывод'}")

        tag = f"{arch}_{'main' if seed is None else 'seed' + str(seed)}{'_test' + str(args.limit) if args.limit else ''}"
        cache = os.path.join(args.out, tag + ".npz")
        if os.path.exists(cache):
            zz = np.load(cache)
            assert list(zz["ids"]) == [r["id"] for r in meta], f"{cache}: не тот набор — удали файл и запусти снова"
            FR, TR, FX, TX = zz["FR"], zz["TR"], zz["FX"], zz["TX"]
            print(f"  средние по группам взяты из {cache}")
        else:
            print(f"  снимаю: сцены правил ({len(rule_i)}) × S, Sp × (исходная, ←→, ↑↓), все слои", flush=True)
            FR, TR = extract(model, [meta[i] for i in rule_i], [infos[i] for i in rule_i], ["o", "h", "v"], LAYERS, device, arch, n_pre)
            print(f"  снимаю: остальные сцены ({len(rest_i)}) × S, Sp, только L12", flush=True)
            FX, TX = extract(model, [meta[i] for i in rest_i], [infos[i] for i in rest_i], ["o"], [12], device, arch, n_pre)
            np.savez(cache, FR=FR.astype(np.float16), TR=TR.astype(np.float16), FX=FX.astype(np.float16),
                     TX=TX.astype(np.float16), ids=np.array([r["id"] for r in meta]))
            FR, TR, FX, TX = (a.astype(np.float16) for a in (FR, TR, FX, TX))   # как из кэша: тот же результат при повторе
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        FR, TR, FX, TX = (a.astype(np.float32) for a in (FR, TR, FX, TX))

        # ---- A: все слои, CLS и дальний фон, сцены правил
        folds_r = folds_all[rule_i]
        R = {}
        for li, l in enumerate(LAYERS):
            for gi, g in enumerate(("cls", "far_bg")):
                r_, nw = probes_A(FR[li, :, :, :, gi], lab_rule, folds_r)
                warn_total += nw
                R[(g, l)] = r_
                if l == 0:
                    for q in ("left", "upper"):
                        for k in ("normal", "normal_m", "honest", "anti"):
                            ok["1"] &= bool(np.allclose(r_[(q, k)], 0.5))
                ok["5"] &= all(r_[(q, "pair_ok")] for q in ("left", "upper"))
            print(f"    A: L{l:02d} готов ({time.time() - t0:.0f} с)", flush=True)
        # доля несимметричной части (справка)
        E = {}
        for li, l in enumerate(LAYERS):
            for gi, g in enumerate(("cls", "far_bg")):
                Xo = FR[li, :, :, 0, gi].reshape(-1, FR.shape[-1])
                var = ((Xo - Xo.mean(0)) ** 2).sum(1).mean()
                for k, mi in (("h", 1), ("v", 2)):
                    a = (FR[li, :, :, 0, gi] - FR[li, :, :, mi, gi]) / 2
                    E[(g, l, k)] = float((a ** 2).sum(-1).mean() / max(var, 1e-30))

        # ---- обычная проба на всех сценах, L12 (проверки 6 и 9; база для D)
        X12 = np.zeros((len(meta), 2, FR.shape[-1]), dtype=np.float32)
        X12[rule_i] = FR[12, :, :, 0, 1]
        X12[rest_i] = FX[0, :, :, 0, 1]
        TW = np.zeros((len(meta), 2, 2, FR.shape[-1]), dtype=np.float32)
        TW[rule_i], TW[rest_i] = TR, TX
        plain = {}
        for q in ("left", "upper"):
            plain[(q, "logreg")], nw = probe_plain(X12, lab_all[q][:, :, 0], lab_all[q][:, :, 1], folds_all)
            warn_total += nw
            tw = TW[np.arange(len(meta)), :, twin_q[q]]
            ctrl, nw = probe_plain(tw, lab_all[q][:, :, 0], lab_all[q][:, :, 1], folds_all)
            warn_total += nw
            acc_c = np.nanmean(ctrl)
            ok["6"] &= acc_c > 0.9
            print(f"  контроль «близнец-вопрос», {QWORD[q]}, L12: {acc_c * 100:.1f}%  {'✅' if acc_c > 0.9 else '⚠️'}")

        # ---- D: MLP только у обученной сети
        if seed is None or name == main_name:
            print(f"  D: подбор alpha MLP по «верхнему» (вложенная проверка) …", flush=True)
            alphas = choose_alphas(X12, lab_all["upper"][:, :, 0], lab_all["upper"][:, :, 1], folds_all)
            print(f"  D: alpha по фолдам: {alphas}", flush=True)
            for q in ("left", "upper"):
                plain[(q, "mlp")], _ = probe_plain(X12, lab_all[q][:, :, 0], lab_all[q][:, :, 1], folds_all, "mlp", alphas)
            print(f"  D готов ({time.time() - t0:.0f} с)", flush=True)
        store[name] = dict(R=R, E=E, plain=plain, weights=weights)

    # ---------------- отчёт ----------------
    for name, st in store.items():
        rng = np.random.default_rng(0)
        for (g, l), r_ in st["R"].items():
            for q in ("left", "upper"):
                for k in ("normal", "normal_m", "honest", "anti", "sym_pair"):
                    a, lo, hi, n = boot(r_[(q, k)], rng)
                    rows.append([name, st["weights"], f"probe_{k}", g, q, f"L{l:02d}", f"{a:.5f}", f"{lo:.5f}", f"{hi:.5f}", n])
            for k in ("h", "v"):
                rows.append([name, st["weights"], f"anti_energy_{k}", g, "", f"L{l:02d}", f"{st['E'][(g, l, k)]:.6f}", "", "", len(rule_i)])
        for (q, k), ps in st["plain"].items():
            a, lo, hi, n = boot(ps, rng)
            rows.append([name, st["weights"], f"all2000_{k}", "far_bg", q, "L12", f"{a:.5f}", f"{lo:.5f}", f"{hi:.5f}", n])
    with open(f"{stem}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["net", "weights", "measure", "group", "query", "layer", "value", "ci_lo", "ci_hi", "n_scenes"])
        w.writerows(rows)

    st = store[main_name]
    rand = [n for n, _ in nets[1:]]
    print(f"\nПроверка 1 (L00: все пробы на CLS и дальнем фоне ровно 50,0%): {'✅' if ok['1'] else '⚠️ проверь весь вывод'}")
    print(f"Проверка 5 (симм. проба: на каждой паре x / mx баллы в сумме ровно 1): {'✅' if ok['5'] else '⚠️ проверь весь вывод'}")
    print(f"Проверка 6 (контроль > 90% на L12 у всех сетей): {'✅' if ok['6'] else '⚠️ проверь весь вывод'}")
    ok7 = all(abs(fps[n] - fps[main_name]) > 1e-3 for n in rand)
    print(f"Проверка 7 (случайная сеть отличается от обученной): {'✅' if ok7 else '⚠️ проверь весь вывод'}  "
          + ", ".join(f"{n} {fps[n]:.4f}" for n in fps))
    print(f"Проверки 3 / 4 / 8: {'✅' if ok['3'] else '⚠️'} / {'✅' if ok['4'] else '⚠️'} / {'✅' if ok['8'] else '⚠️'}")
    print(f"Предупреждений «обучение не сошлось» (логрег): {warn_total}  {'✅' if warn_total == 0 else '⚠️ проверь весь вывод'}")
    pl = {q: float(np.nanmean(st["plain"][(q, "logreg")])) * 100 for q in ("left", "upper")}
    if not test:
        old = old_numbers(arch)
        d9 = max(abs(pl[q] - old[q]) for q in old)
        ok["9"] = d9 <= 0.5
        print(f"Проверка 9 (обычная проба, все сцены, дальний фон, L12 = прежние числа ±0,5 п.п.): сейчас «левый» {pl['left']:.1f}, "
              f"«верхний» {pl['upper']:.1f}; было {old['left']:.1f} / {old['upper']:.1f}  {'✅' if ok['9'] else '⚠️ проверь весь вывод'}")

    print("\n==== Пробы на сценах правил (дальний фон и CLS), точность % ====")
    for g in ("far_bg", "cls"):
        print(f"\n== {mn.NAMES[g]} ==   (обычная на исходных / та же на отражениях / честная / несимм.; случайная сеть: обычная)")
        print("слой    левый: обыч отраж  чест  несим   |  верхний: обыч отраж  чест  несим   |  случ. лев/верх   | доля несимм. ←→ / ↑↓")
        for l in LAYERS:
            r_ = st["R"][(g, l)]
            m = lambda q, k: np.nanmean(r_[(q, k)]) * 100
            rr = [store[n]["R"][(g, l)] for n in rand]
            rs = " ".join(f"{np.nanmean(x[('left', 'normal')]) * 100:4.1f}/{np.nanmean(x[('upper', 'normal')]) * 100:4.1f}" for x in rr)
            print(f"L{l:02d}   {m('left', 'normal'):5.1f} {m('left', 'normal_m'):5.1f} {m('left', 'honest'):5.1f} {m('left', 'anti'):5.1f}   |   "
                  f"{m('upper', 'normal'):5.1f} {m('upper', 'normal_m'):5.1f} {m('upper', 'honest'):5.1f} {m('upper', 'anti'):5.1f}   |   {rs}   | "
                  f"{st['E'][(g, l, 'h')]:.3f} / {st['E'][(g, l, 'v')]:.3f}")

    # ---------------- правила ----------------
    r12 = st["R"][("far_bg", 12)]
    rng = np.random.default_rng(0)
    d1 = r12[("left", "normal_m")] - r12[("left", "normal")]
    a1, a1lo, a1hi, n1 = boot(d1, rng)
    v1 = "часть сигнала — симметричные подсказки" if a1hi < -0.02 else ("подсказок нет" if a1lo > -0.02 else "не решено")
    gfull = r12[("upper", "normal")] - r12[("left", "normal")]
    ganti = r12[("upper", "anti")] - r12[("left", "anti")]
    gf, gflo, gfhi, _ = boot(gfull, np.random.default_rng(0))
    ga, galo, gahi, _ = boot(ganti, np.random.default_rng(0))
    if gflo > 0:
        R_, Rlo, Rhi, _ = boot_ratio(ganti, gfull, np.random.default_rng(0))
        v2 = "в основном маскировка" if Rhi < 0.5 else ("в основном стирание" if Rlo > 0.5 else "смешано")
    else:
        R_ = Rlo = Rhi = float("nan")
        v2 = "не применимо (обычный разрыв не больше 0)"
    rows_rule = [["A1_delta_mirror_minus_orig_left", a1, a1lo, a1hi], ["A2_gap_full", gf, gflo, gfhi],
                 ["A2_gap_anti", ga, galo, gahi], ["A2_R", R_, Rlo, Rhi]]
    v3 = None
    if (("upper", "mlp")) in st["plain"]:
        dmu = st["plain"][("upper", "mlp")] - st["plain"][("upper", "logreg")]
        m1, m1lo, m1hi, _ = boot(dmu, np.random.default_rng(0))
        gm = st["plain"][("upper", "mlp")] - st["plain"][("left", "mlp")]
        g3, g3lo, g3hi, _ = boot(gm, np.random.default_rng(0))
        valid = m1lo > -0.01
        v3 = ("разрыв не артефакт линейной пробы" if g3lo > 0 else "не показано") if valid else "не информативно"
        rows_rule += [["D_mlp_minus_logreg_upper", m1, m1lo, m1hi], ["D_gap_mlp", g3, g3lo, g3hi]]
    with open(f"{stem}.csv", "a", newline="") as fh:
        w = csv.writer(fh)
        for k, v, lo, hi in rows_rule:
            w.writerow([main_name, st["weights"], k, "far_bg", "", "L12", f"{v:.5f}", f"{lo:.5f}", f"{hi:.5f}", len(rule_i)])
        w.writerow([main_name, st["weights"], "VERDICT_A1", "far_bg", "", "L12", v1, "", "", ""])
        w.writerow([main_name, st["weights"], "VERDICT_A2", "far_bg", "", "L12", v2, "", "", ""])
        if v3:
            w.writerow([main_name, st["weights"], "VERDICT_D", "far_bg", "", "L12", v3, "", "", ""])

    print(f"\nВЫВОД ПО ПРАВИЛАМ (реестр 129–130, записано до запуска), {main_name}, дальний фон, L12, сцен правил {len(rule_i)}:")
    print(f"  A1: обычная проба на отражениях − на исходных («левый») = {a1 * 100:+.1f} [{a1lo * 100:+.1f} … {a1hi * 100:+.1f}] п.п. -> «{v1}»   (порог −2 п.п.)")
    print(f"  A2: разрыв обычный {gf * 100:.1f} [{gflo * 100:.1f} … {gfhi * 100:.1f}], в несимм. части {ga * 100:.1f} [{galo * 100:.1f} … {gahi * 100:.1f}] п.п.; "
          f"R = {R_:.2f} [{Rlo:.2f} … {Rhi:.2f}] -> «{v2}»   (порог 0,5)")
    if v3:
        print(f"  D: MLP − логрег на «верхнем» = {m1 * 100:+.1f} [{m1lo * 100:+.1f} … {m1hi * 100:+.1f}] п.п. (допуск: нижняя > −1); "
              f"разрыв MLP {g3 * 100:.1f} [{g3lo * 100:.1f} … {g3hi * 100:.1f}] п.п. -> «{v3}»")
        print(f"     MLP: «левый» {np.nanmean(st['plain'][('left', 'mlp')]) * 100:.1f}%, «верхний» {np.nanmean(st['plain'][('upper', 'mlp')]) * 100:.1f}%; "
              f"логрег: {pl['left']:.1f}% / {pl['upper']:.1f}%")
    if not test and arch in BETS:
        b = BETS[arch]
        print("\nСТАВКА ЕРНАЗАРА (реестр 130, 13.11 11:42):")
        if "a1" in b:
            print(f"  130a A1 «{b['a1']}» -> «{v1}»  {'ПОПАЛ ✅' if v1 == b['a1'] else 'НЕ ПОПАЛ ❌'}")
            print(f"  130b A2 «{b['a2']}» -> «{v2}»  {'ПОПАЛ ✅' if v2 == b['a2'] else 'НЕ ПОПАЛ ❌'}")
            if v3 == "не информативно":
                print("  130d D — «не информативно»: ставка не засчитывается")
            elif v3:
                h = b["d"][0] <= round(g3 * 100, 1) <= b["d"][1]
                print(f"  130d разрыв MLP {b['d'][0]:.1f}–{b['d'][1]:.1f} -> {g3 * 100:.1f}  {'ПОПАЛ ✅' if h else 'НЕ ПОПАЛ ❌'}")
        if "a2_not" in b:
            print(f"  130c A2 не «{b['a2_not']}» -> «{v2}»  {'ПОПАЛ ✅' if v2 != b['a2_not'] else 'НЕ ПОПАЛ ❌'}")
        print("  130e (DINO и CLIP разные) — после обоих прогонов: python mirror_decomp.py --summary")
    print(f"\nЧисла: {stem}.csv   ({time.time() - t0:.0f} с)")


def old_numbers(arch):
    files = OLD[arch]
    out = {}
    if arch in ("dino", "clip"):
        for q, f in zip(("left", "upper"), files):
            for r in csv.DictReader(open(f, encoding="utf-8")):
                if r["probe"] == "logreg" and r["group"] == "far_bg" and r["layer"] == "L12" and "обучен" in r["weights"]:
                    out[q] = float(r["acc"]) * 100
    else:
        rows = list(csv.DictReader(open(files[0], encoding="utf-8")))
        main = rows[0]["net"]
        for r in rows:
            if r["net"] == main and r["measure"] == "probe" and r["group"] == "far_bg" and r["layer"] == "L12":
                out[r["query"]] = float(r["value"]) * 100
    return out


def summary():
    v = {}
    for arch in CONF:
        f = f"mirror_decomp_{arch}.csv"
        if not os.path.exists(f):
            continue
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if r["measure"].startswith("VERDICT_"):
                v[(CONF[arch]["short"], r["measure"][8:])] = r["value"]
    print("Выводы по правилам (дальний фон, L12):")
    for (n, k), x in sorted(v.items()):
        print(f"  {n:9s} {k}: «{x}»")
    if ("DINO", "A2") in v and ("CLIP", "A2") in v:
        diff = v[("DINO", "A2")] != v[("CLIP", "A2")]
        print(f"\nСТАВКА 130e (выводы A2 у DINO и CLIP разные): DINO «{v[('DINO', 'A2')]}», CLIP «{v[('CLIP', 'A2')]}» -> "
              f"{'ПОПАЛ ✅' if diff else 'НЕ ПОПАЛ ❌'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(CONF))
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--n-random", type=int, default=1)
    ap.add_argument("--out", default=os.path.expanduser("~/binding-vit-data/mirror_decomp"))
    ap.add_argument("--limit", type=int, default=0, help="только для проверки в облаке: первые N сцен")
    ap.add_argument("--test-random-main", action="store_true", help="только для проверки в облаке: вместо обученной — случайная seed 99")
    args = ap.parse_args()
    if args.summary:
        summary()
    else:
        assert args.model, "нужно --model или --summary"
        run(args)


if __name__ == "__main__":
    main()

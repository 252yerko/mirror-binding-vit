"""
patch_twins.py — Часть 3, шаг 2а: на каком слое привязка «уезжает» с токенов близнецов? (DINO ViT-B/16)
(реестр 59: дизайн и ставка Ерназара записаны 9.11 11:23 ДО написания кода)

ВОПРОС
  «Какого цвета левый близнец» читается из дальнего фона на L12 (probe_v2: 64,9%). Дальний фон близнецов
  напрямую не видит — информация приходит через внимание. Откуда она приходит: всё время с токенов самих
  близнецов или к какому-то слою уже переписана в другие токены?

ЧТО ДЕЛАЕМ (подмена активаций, interchange intervention)
  Пара картинок одной сцены: S и Sp (близнецы поменялись цветами; остальные пиксели совпадают).
  База — одна картинка пары, источник — другая (оба направления: S←Sp и Sp←S).
  На слое L (L00 = вход в блок 1, …, L11 = выход блока 11) токены ОБОИХ близнецов (все патчи, где в маске
  есть близнец) в прогоне базы заменяются значениями этих же токенов из прогона источника; дальше сеть
  считает как обычно до L12. Условия подмены:
    twins   — токены близнецов (ОСНОВНОЕ, правило);
    control — столько же других токенов: сначала нетронутые объекты, затем ближний фон (не дальний фон и
              не CLS), выбор с seed по сцене (СПРАВКА: на L00 обязан давать 0 — эти токены в S и Sp одинаковы);
    rest    — все токены, кроме близнецов (включая CLS и дальний фон) (СПРАВКА: куда «переехала» привязка).

ЧТО МЕРЯЕМ
  Проба «левый» на дальнем фоне L12 (логрег C=0,1, StandardScaler, как probe_v2.make_model) обучается на
  ОБЫЧНОЙ сети и дальше не меняется; кросс-валидация по сценам — те же 5 фолдов, что в probe_v2: проба
  обучается на 4 фолдах (чистые S и Sp) и применяется к подменённым прогонам сцен 5-го фолда (отложенные сцены).
  «Ответ как у источника»: проба ставит цвет левого близнеца источника выше цвета левого близнеца базы
  (это два цвета близнецов; так же сравнивает probe_v2: правильный против соперника).
  Учитываются только пары, где фиксированная проба на ЧИСТЫХ картинках права и для базы, и для источника.
  T(L) = доля таких прогонов (оба направления), где после подмены на слое L ответ как у источника.
  Интервалы: бутстрэп по сценам, 10 000 повторов, seed 0, 95%.

ПРАВИЛО ВЫВОДА (до запуска; в OSF — дословно)
  L* = первый слой из L01…L11, на котором T_twins(L) (точечная оценка) < 50%; если такого нет — «ни на каком».
  Дополнительно сообщаются T_twins(L11) и интервалы для всех слоёв.
  Проверка кода: T_twins(L00) должно быть 100% (подмена на входе = подмена картинки), T_control(L00) = 0%.
  Случайные DINO (seed 0/1/2) — только twins; у них проба ≈ 50%, поэтому учитываемых пар мало; справка.

СТАВКА ЕРНАЗАРА (реестр 59, 9.11 11:23, до кода)
  59a: первый слой с T < 50% — около L07 (попал, если L06–L08); 59b: T(L11) около 15% (попал, если 10,0–20,0%);
  уверенность в 59a 30%.

ВСТРОЕННЫЕ ПРОВЕРКИ
  1. Наш пошаговый прогон = прогон блоков модели (extract.run_layers): разница ~0.
  2. Подмена на L00 токенов близнецов даёт ровно активации источника на L12 (все токены): разница ~0.
  3. Подмена control на L00 не меняет ничего (те же пиксели): разница ~0.
  4. Признаки дальнего фона чистой сети = способ probe_v2 (float16 → среднее); точность фиксированной
     пробы на чистых картинках = probe_v2 (дальний фон L12 «левый» 64,875%).

ЗАПУСК (на Маке, из папки binding-vit, с зарядкой; ~1,5 ч: DINO три условия ~40 мин, 3 случайные DINO ~40 мин):
  caffeinate -i python patch_twins.py 2>&1 | tee patch_twins_log.txt
  Промежуточные результаты — ~/binding-vit-data/patch_twins/ (при обрыве та же команда продолжит с места).

ЧЕГО ЭТОТ ОПЫТ НЕ ПОКАЖЕТ
  Подмена целых токенов меняет в них всё сразу (цвет, форму, место), поэтому опыт показывает, ГДЕ на каждом
  слое лежит информация, от которой зависит чтение привязки, а не в каком подпространстве и не как она
  устроена (это DAS, следующий шаг). Одна модель (DINO ViT-B/16).
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image

import ablate_blocks as ab          # общие: DATA, BATCH, group_feats (как probe_v2)
import probe_v2 as pv
from extract import load_model, fingerprint, pick_device, run_layers

DATA = ab.DATA
G, P = 14, 16                       # сетка патчей 14×14 по 16 пикселей (как probe_v2)
LAYERS = list(range(0, 12))         # слой подмены: L00 … L11
CONDS = ["twins", "control", "rest"]
N_BOOT = 10000
EXPECTED_FP = ab.EXPECTED_FP
REF_LEFT_FAR = 0.64875              # probe_v2.csv: дальний фон L12 «левый»
BET = dict(layer=(6, 8), t11=(10.0, 20.0), conf="30%")


# ------------------------------------------------------------------
# 1. Токены сцены
# ------------------------------------------------------------------
def scene_tokens(rec, far_bg, unchanged):
    """twins: все патчи с близнецом (маска 1 или 2) -> номера токенов (+1 за CLS);
    control: столько же токенов из нетронутых, затем ближний фон (не дальний фон, не CLS, не близнецы)."""
    mask = np.asarray(Image.open(f"{DATA}/masks/{rec['id']}.png"))
    blk = mask.reshape(G, P, G, P).transpose(0, 2, 1, 3).reshape(G * G, P * P)
    twin = ((blk == 1) | (blk == 2)).any(1)
    twins = [p + 1 for p in np.where(twin)[0]]
    far = set(far_bg)
    near_bg = [p + 1 for p in range(G * G) if not twin[p] and (p + 1) not in far and (p + 1) not in set(unchanged)]
    rng = np.random.default_rng(int(rec["id"]))
    pool1 = list(rng.permutation(unchanged)) if unchanged else []
    pool2 = list(rng.permutation(near_bg)) if near_bg else []
    control = (pool1 + pool2)[:len(twins)]
    rest = [t for t in range(197) if t not in set(twins)]
    return {"twins": twins, "control": [int(t) for t in control], "rest": rest}


# ------------------------------------------------------------------
# 2. Прогоны
# ------------------------------------------------------------------
@torch.no_grad()
def clean_layers(model, x):
    """[13, B, 197, 768]: L00 (prepare_tokens) и выходы блоков 1..12 — как extract.run_layers для DINO."""
    h = model.prepare_tokens(x)
    outs = [h]
    for blk in model.blocks:
        h = blk(h)
        outs.append(h)
    return outs


@torch.no_grad()
def from_layer(model, h, L):
    """Продолжить прогон с активаций слоя L (выход блока L) до L12."""
    for blk in model.blocks[L:]:
        h = blk(h)
    return h


def norm_batch(u8):
    x = torch.from_numpy(u8).permute(0, 3, 1, 2).float() / 255.0
    return (x - ab.MEAN) / ab.STD


def far_feat(h, rows, ginfo):
    """[B, 768]: среднее по дальнему фону, как probe_v2 (float16 → float32 → среднее)."""
    return ab.group_feats(h, rows, ginfo)[:, 0]


def run_net(name, model, imgs, ginfo, toks, device, cache, tag, conds):
    """Признаки дальнего фона L12: clean [N_img, 768] и подменённые [N_img, 12, len(conds), 768] (float16).
    Строка 2i = S (база S, источник Sp), 2i+1 = Sp (база Sp, источник S)."""
    fp = round(fingerprint(model), 4)
    os.makedirs(cache, exist_ok=True)
    path = f"{cache}/{tag}{name}_{'-'.join(conds)}.npz"
    if os.path.exists(path) and int(np.load(path)["fp_ok"]) == int(fp * 10000):
        print(f"  признаки взяты из кэша {path}")
        d = np.load(path)
        return d["clean"], d["patched"]
    n_sc = len(toks)
    clean = np.zeros((2 * n_sc, 768), np.float32)
    patched = np.zeros((2 * n_sc, len(LAYERS), len(conds), 768), np.float16)
    t0 = time.time()
    bs = max(1, ab.BATCH // 2)                       # сцен в пачке (по 2 картинки)
    for s in range(0, n_sc, bs):
        sc = list(range(s, min(n_sc, s + bs)))
        rows = [r for i in sc for r in (2 * i, 2 * i + 1)]
        x = norm_batch(imgs[rows]).to(device)
        H = clean_layers(model, x)                   # список 13 × [B,197,768]
        clean[rows] = far_feat(H[12], rows, ginfo)
        src = [r ^ 1 for r in range(len(rows))]      # источник: другая картинка пары
        for li, L in enumerate(LAYERS):
            for ci, c in enumerate(conds):
                h = H[L].clone()
                for b, r in enumerate(rows):
                    t = toks[r // 2][c]
                    if t:
                        h[b, t] = H[L][src[b], t]
                out = from_layer(model, h, L)
                patched[rows, li, ci] = far_feat(out, rows, ginfo).astype(np.float16)
        if s == 0 or (s // bs) % 50 == 0:
            print(f"    {name}: сцен {min(n_sc, s + bs)}/{n_sc}, {time.time() - t0:.0f} с", flush=True)
    np.savez(path, clean=clean, patched=patched, fp_ok=int(fp * 10000))
    print(f"  {name}: подмены готовы, {time.time() - t0:.0f} с", flush=True)
    return clean, patched


# ------------------------------------------------------------------
# 3. Фиксированная проба и T(L)
# ------------------------------------------------------------------
def fixed_probe_T(clean, patched, labels, scene_idx, folds, conds):
    """Проба на чистых признаках (4 фолда) -> применяется к подменённым прогонам 5-го фолда.
    Возвращает: acc_clean (балл как probe_v2, по сценам), T[cond][L] = (scene_ids, доли по сценам), n_pairs, n_warn."""
    y_true, y_rival = labels                         # цвет левого близнеца базы и «соперник» (= цвет левого у источника)
    n_img = len(clean)
    score_clean = np.zeros(n_img)
    src_pref = np.zeros((n_img, len(LAYERS), len(conds)))
    n_warn = 0
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    for f in range(pv.N_FOLDS):
        test = folds[scene_idx] == f
        clf = pv.make_model("logreg")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            clf.fit(clean[~test], y_true[~test])
            n_warn += sum(issubclass(x.category, ConvergenceWarning) for x in w)
        col = {c: i for i, c in enumerate(clf.classes_)}
        it = np.where(test)[0]
        ct = np.array([col[c] for c in y_true[it]])
        cr = np.array([col[c] for c in y_rival[it]])
        d = clf.predict_proba(clean[it])
        pt, pr = d[np.arange(len(it)), ct], d[np.arange(len(it)), cr]
        score_clean[it] = np.where(pt > pr, 1.0, np.where(pt == pr, 0.5, 0.0))
        for li in range(len(LAYERS)):
            for ci in range(len(conds)):
                d = clf.predict_proba(patched[it, li, ci].astype(np.float32))
                pt, pr = d[np.arange(len(it)), ct], d[np.arange(len(it)), cr]
                src_pref[it, li, ci] = np.where(pr > pt, 1.0, np.where(pr == pt, 0.5, 0.0))   # предпочёл цвет источника
    # учитываем прогон «база r, источник r^1», если чистая проба права и на базе, и на источнике
    ok = (score_clean == 1.0) & (score_clean[np.arange(n_img) ^ 1] == 1.0)
    T = {}
    for ci, c in enumerate(conds):
        T[c] = []
        for li in range(len(LAYERS)):
            v = src_pref[ok, li, ci]
            s_id = scene_idx[ok]
            sums = np.bincount(s_id, weights=v, minlength=scene_idx.max() + 1)
            cnt = np.bincount(s_id, minlength=scene_idx.max() + 1)
            keep = cnt > 0
            T[c].append(sums[keep] / cnt[keep])     # доля по сцене (1 или 2 прогона)
    per_scene_clean = np.bincount(scene_idx, weights=score_clean) / np.bincount(scene_idx)
    return per_scene_clean, T, int(ok.sum()), n_warn


def boot(v, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    b = v[idx].mean(1)
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# ------------------------------------------------------------------
# 4. main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="только для проверки: первые N сцен")
    ap.add_argument("--main-seed", type=int, default=None, help="только для проверки в облаке: случайные веса вместо DINO")
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--cache", default=os.path.expanduser("~/binding-vit-data/patch_twins"))
    args = ap.parse_args()
    stem = "patch_twins" + (f"_test{args.limit}" if args.limit else "")
    tag = (f"test{args.limit}_" if args.limit else "") + (f"mainseed{args.main_seed}_" if args.main_seed is not None else "")
    t_start = time.time()

    meta_all = [json.loads(l) for l in open(f"{DATA}/meta.jsonl")]
    colors = sorted({o[k] for r in meta_all for o in r["objects"] for k in ("color", "color_Sp")})
    folds_all = np.random.default_rng(0).permutation(len(meta_all)) % pv.N_FOLDS
    meta = meta_all[:args.limit] if args.limit else meta_all
    if args.limit:
        print(f"⚠️ ПРОВЕРОЧНЫЙ прогон: только {len(meta)} сцен (--limit). Для настоящих чисел запускай без --limit.")
    pv.QUERY = "left"
    pv.GROUPS[3] = "left_twin"
    ginfo, toks, y_t, y_r = [], [], [], []
    for rec in meta:
        groups, lab = pv.scene_info(rec, colors)
        ginfo.append({"far_bg": groups["far_bg"], "left_twin": groups["left_twin"], "upper_twin": []})
        toks.append(scene_tokens(rec, groups["far_bg"], groups["unchanged"]))
        for v in ("S", "Sp"):
            y_t.append(lab[v][0])
            y_r.append(lab[v][1])
    labels = (np.array(y_t), np.array(y_r))
    n_sc = len(meta)
    scene_idx = np.repeat(np.arange(n_sc), 2)
    folds = folds_all[:n_sc]
    n_tw = np.array([len(t["twins"]) for t in toks])
    n_ct = np.array([len(t["control"]) for t in toks])
    print(f"Сцен: {n_sc}. Токенов близнецов на сцену: {n_tw.min()}–{n_tw.max()} (в среднем {n_tw.mean():.1f}); "
          f"control столько же: {'✅' if np.array_equal(n_tw, n_ct) else f'⚠️ в {int((n_ct < n_tw).sum())} сценах меньше'}")
    print(f"Проба: «левый», дальний фон L12, логрег C={pv.C_REG}, обучена на чистой сети, {pv.N_FOLDS} фолдов по сценам (как probe_v2)")

    imgs = np.zeros((2 * n_sc, 224, 224, 3), np.uint8)
    for i, rec in enumerate(meta):
        for k, v in enumerate(("S", "Sp")):
            imgs[2 * i + k] = np.asarray(Image.open(f"{DATA}/images/{rec['id']}_{v}.png").convert("RGB"))

    device = pick_device()
    nets = [("DINO", args.main_seed)] + [(f"rand{s}", s) for s in range(args.n_random)]
    res = {}
    for name, seed in nets:
        model, weights, _ = load_model("dino", seed)
        fp = fingerprint(model)
        extra = ""
        if name.startswith("rand") and seed in EXPECTED_FP:
            extra = "✅ совпадает с extract.py" if abs(fp - EXPECTED_FP[seed]) < 0.01 else "⚠️ НЕ совпадает — проверь весь вывод"
        print(f"\n### {name}: {weights}   отпечаток весов {fp:.4f}  {extra}  устройство: {device}")
        model.eval().to(device)
        if name == "DINO":
            run_checks(model, imgs, toks, device)
        conds = CONDS if name == "DINO" else ["twins"]
        clean, patched = run_net(name, model, imgs, ginfo, toks, device, args.cache, tag, conds)
        acc, T, n_ok, nw = fixed_probe_T(clean, patched, labels, scene_idx, folds, conds)
        res[name] = (acc, T, n_ok, nw, conds)
        print(f"  точность фиксированной пробы на чистых картинках: {acc.mean() * 100:.3f}%   "
              f"учтённых прогонов (проба права и на базе, и на источнике): {n_ok}   «не сошлось»: {nw}")
        if name == "DINO" and args.main_seed is None and not args.limit:
            ok = abs(acc.mean() - REF_LEFT_FAR) < 5e-6
            print(f"  Проверка 4б (= probe_v2, дальний фон L12 «левый» {REF_LEFT_FAR * 100:.3f}%): "
                  f"{'✅ точно' if ok else ('✅ (разница < 0,5 п.п.)' if abs(acc.mean() - REF_LEFT_FAR) < 0.005 else '⚠️ проверь весь вывод')}")
        del model
    report(res, stem)
    print(f"\nВсего: {time.time() - t_start:.0f} с")


@torch.no_grad()
def run_checks(model, imgs, toks, device):
    x = norm_batch(imgs[:4]).to(device)
    H = clean_layers(model, x)
    d1 = float((H[12] - run_layers(model, x)[-1]).abs().max())
    print(f"Проверка 1 (наш прогон = прогон блоков модели): макс. разница {d1:.2e}  {'✅' if d1 < 1e-3 else '⚠️ проверь весь вывод'}")
    d2, d3 = 0.0, 0.0
    for c, store in (("twins", "d2"), ("control", "d3")):
        h = H[0].clone()
        for b in range(4):
            t = toks[b // 2][c]
            if t:
                h[b, t] = H[0][b ^ 1, t]
        out = from_layer(model, h, 0)
        if c == "twins":
            d2 = float((out - H[12][[1, 0, 3, 2]]).abs().max())
        else:
            d3 = float((out - H[12]).abs().max())
    print(f"Проверка 2 (подмена близнецов на L00 = прогон источника, все токены L12): макс. разница {d2:.2e}  "
          f"{'✅' if d2 < 1e-3 else '⚠️ проверь весь вывод'}")
    print(f"Проверка 3 (подмена control на L00 ничего не меняет): макс. разница {d3:.2e}  {'✅' if d3 < 1e-3 else '⚠️ проверь весь вывод'}")


def report(res, stem):
    import csv
    acc, T, n_ok, nw, conds = res["DINO"]
    warn = sum(r[3] for r in res.values())
    rows_csv = []
    print("\n" + "=" * 100)
    print("T(L) — доля прогонов, где после подмены на слое L фиксированная проба отвечает как у источника, % [95% инт.]")
    print(f"{'слой':5s}  " + "   ".join(f"{c:>22s}" for c in conds) + "   " +
          "   ".join(f"{n} (twins)" for n in res if n != "DINO"))
    stats = {}
    for li, L in enumerate(LAYERS):
        line = f"L{L:02d}   "
        for c in conds:
            m, lo, hi = boot(T[c][li])
            stats[(c, L)] = (m, lo, hi)
            line += f"   {m * 100:6.1f} [{lo * 100:5.1f}–{hi * 100:5.1f}]   "
            rows_csv.append(["DINO", c, f"L{L:02d}", f"{m:.5f}", f"{lo:.5f}", f"{hi:.5f}", len(T[c][li])])
        for n, (a2, T2, n2, _, _) in res.items():
            if n == "DINO":
                continue
            m2 = boot(T2["twins"][li])
            line += f"   {m2[0] * 100:6.1f}"
            rows_csv.append([n, "twins", f"L{L:02d}", f"{m2[0]:.5f}", f"{m2[1]:.5f}", f"{m2[2]:.5f}", len(T2['twins'][li])])
        print(line)
    for n, (a2, T2, n2, _, _) in res.items():
        if n != "DINO":
            print(f"  {n}: точность чистой пробы {a2.mean() * 100:.1f}%, учтённых прогонов {n2}")

    t00, c00 = stats[("twins", 0)][0], stats[("control", 0)][0]
    print(f"\nПроверка 5 (T_twins(L00) = 100%, T_control(L00) = 0%): {t00 * 100:.1f}% / {c00 * 100:.1f}%  "
          f"{'✅' if t00 > 0.999 and c00 < 0.001 else '⚠️ проверь весь вывод'}")
    print(f"Предупреждений «обучение не сошлось» всего: {warn}  {'✅' if warn == 0 else '⚠️ проверь весь вывод'}")

    Lstar = next((L for L in LAYERS[1:] if stats[("twins", L)][0] < 0.5), None)
    t11 = stats[("twins", 11)]
    print(f"\nВЫВОД ПО ПРАВИЛУ (реестр 59, записано до запуска): первый слой, где T_twins < 50% — "
          f"{'L%02d' % Lstar if Lstar is not None else 'ни на каком (T ≥ 50% до L11)'};  "
          f"T_twins(L11) = {t11[0] * 100:.1f}% [{t11[1] * 100:.1f}–{t11[2] * 100:.1f}]")
    print("  Оговорки: одна модель; подмена целых токенов (цвет + форма + место сразу) — опыт показывает, где лежит "
          "информация, от которой зависит чтение, а не подпространство и не механизм.")
    t11r = round(t11[0] * 100, 1)
    hit_a = Lstar is not None and BET["layer"][0] <= Lstar <= BET["layer"][1]
    hit_b = BET["t11"][0] <= t11r <= BET["t11"][1]
    print(f"\nСТАВКА ЕРНАЗАРА (реестр 59, 9.11 11:23, до кода):")
    print(f"  59a первый слой с T < 50%: L{BET['layer'][0]:02d}–L{BET['layer'][1]:02d} (около L07) -> вышло "
          f"{'L%02d' % Lstar if Lstar is not None else 'ни на каком'}   {'ПОПАЛ ✅' if hit_a else 'НЕ ПОПАЛ ❌'}")
    print(f"  59b T(L11): от {BET['t11'][0]} до {BET['t11'][1]}% -> вышло {t11r:.1f}%   {'ПОПАЛ ✅' if hit_b else 'НЕ ПОПАЛ ❌'}")
    print(f"  (уверенность в 59a: {BET['conf']})")

    with open(f"{stem}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["net", "condition", "layer", "T", "T_lo", "T_hi", "n_scenes"])
        w.writerows(rows_csv)
    plot(stats, conds, res, stem)
    print(f"\nГрафик: {stem}.png   Числа: {stem}.csv")


def plot(stats, conds, res, stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for c, col in zip(conds, ("C0", "C2", "C1")):
        m = [stats[(c, L)][0] * 100 for L in LAYERS]
        lo = [stats[(c, L)][1] * 100 for L in LAYERS]
        hi = [stats[(c, L)][2] * 100 for L in LAYERS]
        ax.plot(LAYERS, m, "o-", color=col, label=c)
        ax.fill_between(LAYERS, lo, hi, color=col, alpha=0.2)
    for n, (_, T2, _, _, _) in res.items():
        if n != "DINO":
            ax.plot(LAYERS, [boot(T2["twins"][li])[0] * 100 for li in range(len(LAYERS))], ":", color="gray", lw=1)
    ax.axhline(50, c="k", lw=0.8, ls="--")
    ax.set_xticks(LAYERS, [f"L{L:02d}" for L in LAYERS], fontsize=8)
    ax.set_ylabel("T(L): ответ как у источника, %")
    ax.set_title("DINO: подмена токенов на слое L, чтение фиксированной пробой «левый» (дальний фон L12)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=150)


if __name__ == "__main__":
    main()

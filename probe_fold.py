"""
probe_fold.py — опыт «СКЛАДЫВАНИЕ»: что стирает сеть, слепая к зеркалу слева-направо —
саму горизонтальную привязку или только «лево/право»?

ЗАЧЕМ. Если бы сеть была ПОЛНОСТЬЮ слепа к зеркалу ←→, проба «какого цвета левый близнец»
дала бы ~50% просто по построению: картинка и её зеркало дают один вектор, а правильные ответы
у них разные (то же рассуждение, что для «мешка признаков»). Значит, падение «левого» вместе
с зеркальной слепотой отчасти следует из математики. Неочевидный вопрос другой:
  «СКЛАДЫВАНИЕ» — сеть «сложила картинку пополам» по вертикальной оси: помнит, НАСКОЛЬКО близнец
                   далеко от середины, но не С КАКОЙ стороны. Привязка сохранена, стёрт только знак.
  «СТИРАНИЕ»     — сеть выбросила горизонтальную привязку целиком.
Различаем вопросом, ответ на который зеркало ←→ НЕ меняет:
  «какого цвета близнец, который БЛИЖЕ К СЕРЕДИНЕ по горизонтали»  (меньше |x − 111,5|).
Контроль сложности — тот же вопрос по вертикали (меньше |y − 111,5|): сеть не слепа к зеркалу ↑↓,
поэтому он показывает, насколько хорошо вообще читается вопрос «ближе к середине».

ПОЧЕМУ СЕРЕДИНА, А НЕ ТРЕТЬЯ ФИГУРА. «Левый» читается линейной пробой, если в токенах есть
«цвет × позиция»: проба сравнивает близнецов по одной и той же функции от x. «Ближе к середине» —
тоже одна и та же функция от x (|x − 111,5|), поэтому для линейной пробы он не труднее «левого».
«Ближе к третьей фигуре» — функция и от позиции третьей фигуры; линейная проба такое не построит,
если сеть сама не посчитала отношения между фигурами. Сравнение было бы нечестным.

ПОЧЕМУ 111,5. Картинка 224 px, столбцы 0…223; зеркало ←→ переводит столбец x в 223 − x
(фигуры — «штампы» с центром в целом пикселе, generate_v2.py). Середина между x и 223 − x = 111,5,
поэтому |x − 111,5| у близнеца и у его зеркального отражения одинаковый.

ЧЕТЫРЕ ВОПРОСА (близнец-«вопрос»; при равенстве — следующее правило в скобках):
  left   «левый»                               меньше x        (выше)                      — как в probe_v2.py
  upper  «верхний»                             меньше y        (левее)                     — как в probe_v2.py --query upper
  cx     «ближе к середине по горизонтали»     меньше |x−111,5| (ближе к середине по верт.; левее)
  cy     «ближе к середине по вертикали»       меньше |y−111,5| (ближе к середине по гориз.; выше)
Уровень «мешка признаков» у всех четырёх — ровно 50%: в S и Sp места одинаковые, цвета близнецов
переставлены, правильный ответ разный.

КАК СЧИТАЕМ. Та же проба, что в probe_v2.py: StandardScaler + логрег C=0,1, те же 5 фолдов по сценам,
обучение на ВСЕХ 2000 сценах (как в probe_v2.py — тогда «левый» и «верхний» совпадают с прошлыми
прогонами до 5 знаков, это проверка 1). Сравниваем на ОБЩЕМ подмножестве сцен, где все четыре вопроса
различимы сетью: разница расстояний близнецов до середины ≥ 16 px (один патч) И по горизонтали, И по
вертикали (на Маке 1043 сцены из 2000; из этого следует и |dx| ≥ 16, |dy| ≥ 16). Одни и те же сцены
для всех вопросов -> парный бутстрэп по сценам для разрывов.
  разрыв «сторона»  = «верхний» − «левый»                       (у DINO на всех сценах ≈ 20 п.п. на L12)
  разрыв «середина» = «ближе по вертикали» − «ближе по горизонтали»
  D = разрыв «сторона» − разрыв «середина»

ПРЕДСКАЗАНИЯ И ПРАВИЛО ВЫВОДА (записаны ДО запуска на DINO и CLIP, 6.10; см. журнал):
  «складывание»   на L12 разрыв «середина» мал: нижняя граница 95% ДИ для D > 0 И разрыв «середина»
                  ≤ половины разрыва «сторона». «Ближе по горизонтали» не падает с глубиной так, как «левый».
  «стирание»      D неотличимо от 0 (95% ДИ включает 0): разрыв «середина» ≈ разрыву «сторона».
  «промежуточно»  D > 0, но разрыв «середина» больше половины разрыва «сторона».
  «не отвечает»   (а) «ближе по вертикали» на L12 не выше всех случайных сетей — вопрос «ближе к середине»
                  не читается даже по вертикали; или (б) разрыва «сторона» на этом подмножестве нет.
  Асимметрия выводов: успех «ближе по горизонтали» — сильное свидетельство складывания; провал —
  более слабое свидетельство стирания (сохранённая информация могла быть в другом, нечитаемом виде).

ВСТРОЕННЫЕ ПРОВЕРКИ:
  1. «Левый» и «верхний» по всем сценам = probe_v2*.csv до 5 знаков (та же проба, те же фолды).
  2. На L00 CLS и дальний фон в S и Sp совпадают до бита -> ровно 50,0% у всех вопросов, всех сетей,
     и на всех сценах, и на подмножестве.
  3. Контроль: проба по патчам САМОГО близнеца-«вопроса» на L00 и L12 > 90% у всех сетей и вопросов
     (проверяет, что ответы посчитаны правильно).
  4. Ответы четырёх вопросов не связаны друг с другом: совпадение близнеца-«вопроса» ≈ 50% (печатается).
  5. Число предупреждений «обучение не сошлось».

ЗАПУСК (из папки binding-vit; оценка ~30 мин на 4 сети — логрег учится вдвое чаще, чем в probe_v2.py):
  caffeinate -i python probe_fold.py 2>&1 | tee probe_fold_log.txt
  caffeinate -i python probe_fold.py --model clip 2>&1 | tee probe_fold_clip_log.txt
  -> таблицы, график probe_fold[_clip].png, числа probe_fold[_clip].csv,
     баллы каждой сцены probe_fold[_clip]_scores.npz (сохраняются после КАЖДОЙ сети)
"""

import argparse
import csv
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import probe_v2 as pv

MID = 111.5                 # середина картинки по обеим осям (см. «ПОЧЕМУ 111,5»)
MARGIN = 16                 # px: разница расстояний до середины, при которой вопрос различим (один патч)
QUERIES = ["left", "upper", "cx", "cy"]
QWORD = {"left": "левый", "upper": "верхний",
         "cx": "ближе к середине по горизонтали", "cy": "ближе к середине по вертикали"}
QSHORT = {"left": "левый", "upper": "верхний", "cx": "сер.←→", "cy": "сер.↑↓"}
GROUPS = ["cls", "far_bg"]
NAMES = {"cls": "CLS", "far_bg": "дальний фон"}
CONTROL_LAYERS = [0, 12]
N_BOOT = 1000
NETS = {"dino": ["acts_v2", "acts_v2_rand0", "acts_v2_rand1", "acts_v2_rand2"],
        "clip": ["acts_v2_clip", "acts_v2_clip_rand0", "acts_v2_clip_rand1", "acts_v2_clip_rand2"]}
REF_CSV = {"dino": {"left": "probe_v2.csv", "upper": "probe_v2_upper.csv"},
           "clip": {"left": "probe_v2_clip.csv", "upper": "probe_v2_clip_upper.csv"}}
SHORT = {"обученные DINO": "DINO", "обученный CLIP (OpenAI)": "CLIP"}


def twin_key(o, q):
    """Ключ для выбора близнеца-«вопроса»: выбираем близнеца с МЕНЬШИМ ключом (дальше — правила при равенстве)."""
    x, y = o["center"]
    return {"left": (x, y), "upper": (y, x),
            "cx": (abs(x - MID), abs(y - MID), x, y),
            "cy": (abs(y - MID), abs(x - MID), y, x)}[q]


def query_twin(rec, q):
    o0, o1 = rec["objects"][0], rec["objects"][1]
    return 0 if twin_key(o0, q) < twin_key(o1, q) else 1


def labels_for(rec, q, ci):
    """(правильный цвет, цвет-соперник) для S и Sp — как в probe_v2.scene_info."""
    k = query_twin(rec, q)
    oq, oo = rec["objects"][k], rec["objects"][1 - k]
    assert oq["color_Sp"] == oo["color"] and oo["color_Sp"] == oq["color"], f"{rec['id']}: в Sp цвета не переставлены"
    return {"S": (ci[oq["color"]], ci[oo["color"]]), "Sp": (ci[oq["color_Sp"]], ci[oo["color_Sp"]])}


def control_tokens(rec, q):
    """Токены патчей самого близнеца-«вопроса» (без патчей, которые он делит с другим близнецом)."""
    k = query_twin(rec, q)
    mask = np.asarray(Image.open(f"{pv.DATA}/masks/{rec['id']}.png"))
    blk = mask.reshape(pv.G, pv.P, pv.G, pv.P).transpose(0, 2, 1, 3).reshape(pv.G * pv.G, pv.P * pv.P)
    own, other = (blk == k + 1).any(1), (blk == (1 - k) + 1).any(1)
    return [p + 1 for p in np.where(own & ~other)[0]]


def probe_scores(X, labs, scene_idx, folds, n_scenes):
    """Балл каждой сцены (0 / 0.5 / 1, среднее по S и Sp); NaN — сцены нет в группе."""
    yt = np.array([a for a, _ in labs])
    yr = np.array([b for _, b in labs])
    per_scene, nw = pv.run_probe(X, yt, yr, scene_idx, folds)
    full = np.full(n_scenes, np.nan)
    full[np.unique(scene_idx)] = per_scene["logreg"]
    return full, nw


def analyse_net(acts, meta, base_groups, labels, ctrl, folds):
    rows = {(r["id"], r["version"]): int(r["row"]) for r in csv.DictReader(open(f"{acts}/index.csv"))}
    weights = json.load(open(f"{acts}/info.json")).get("weights", "обученные DINO")
    print(f"\n### {acts}/   веса: {weights}", flush=True)
    n = len(meta)
    sc, warn, t0 = {}, 0, time.time()
    for l in range(pv.N_LAYERS):
        A = np.load(f"{acts}/L{l:02d}.npy", mmap_mode="r")
        feats = {g: ([], [], []) for g in GROUPS}             # X, номер сцены, версия (S / Sp)
        cfeat = {q: ([], [], []) for q in QUERIES}
        for si, rec in enumerate(meta):
            for v in ("S", "Sp"):
                h = np.asarray(A[rows[(rec["id"], v)]], dtype=np.float32)
                for g in GROUPS:
                    toks = base_groups[si][g]
                    if toks:                                   # как в probe_v2: пустую группу пропускаем
                        feats[g][0].append(h[toks].mean(0))
                        feats[g][1].append(si)
                        feats[g][2].append(v)
                if l in CONTROL_LAYERS:
                    for q in QUERIES:
                        if ctrl[q][si]:
                            cfeat[q][0].append(h[ctrl[q][si]].mean(0))
                            cfeat[q][1].append(si)
                            cfeat[q][2].append(v)
        line = []
        for g in GROUPS:
            X, sidx, ver = np.array(feats[g][0]), np.array(feats[g][1]), feats[g][2]
            for q in QUERIES:
                labs = [labels[q][s][v] for s, v in zip(sidx, ver)]
                sc[(q, g, l)], nw = probe_scores(X, labs, sidx, folds, n)
                warn += nw
            line.append(f"{NAMES[g]}: " + " ".join(f"{QSHORT[q]} {np.nanmean(sc[(q, g, l)]) * 100:4.1f}" for q in QUERIES))
        if l in CONTROL_LAYERS:
            for q in QUERIES:
                X, sidx, ver = np.array(cfeat[q][0]), np.array(cfeat[q][1]), cfeat[q][2]
                labs = [labels[q][s][v] for s, v in zip(sidx, ver)]
                sc[(q, "control", l)], nw = probe_scores(X, labs, sidx, folds, n)
                warn += nw
        print(f"  L{l:02d}  " + "   ".join(line) + f"   (все сцены, {time.time() - t0:.0f} с)", flush=True)
    print(f"  предупреждений «обучение не сошлось»: {warn}")
    return weights, sc, warn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["dino", "clip"], default="dino", help="какую модель и её случайные сети брать")
    ap.add_argument("--acts", nargs="+", default=None, help="(для проверок) свои папки: первая — обученная сеть")
    ap.add_argument("--tag", default=None, help="приписка к файлам; по умолчанию '' для dino и 'clip' для clip")
    args = ap.parse_args()
    nets = args.acts or NETS[args.model]
    tag = args.tag if args.tag is not None else ("" if args.model == "dino" else "clip")
    stem = "probe_fold" + (f"_{tag}" if tag else "")
    pv.MODELS = ["logreg"]

    meta = [json.loads(line) for line in open(f"{pv.DATA}/meta.jsonl")]
    n = len(meta)
    colors = sorted({o[k] for r in meta for o in r["objects"] for k in ("color", "color_Sp")})
    ci = {c: i for i, c in enumerate(colors)}
    folds = np.random.default_rng(0).permutation(n) % pv.N_FOLDS          # как в probe_v2.py
    pv.QUERY, pv.GROUPS[3] = "left", "left_twin"
    base_groups = [pv.scene_info(rec, colors)[0] for rec in meta]         # + проверка пикселей S/Sp
    labels = {q: [labels_for(rec, q, ci) for rec in meta] for q in QUERIES}
    ctrl = {q: [control_tokens(rec, q) for rec in meta] for q in QUERIES}

    cen = np.array([[o["center"] for o in r["objects"][:2]] for r in meta], dtype=float)   # [сцена, близнец, (x, y)]
    ax_ = np.abs(np.abs(cen[:, 0, 0] - MID) - np.abs(cen[:, 1, 0] - MID))
    ay_ = np.abs(np.abs(cen[:, 0, 1] - MID) - np.abs(cen[:, 1, 1] - MID))
    SUB = (ax_ >= MARGIN) & (ay_ >= MARGIN)
    dx, dy = np.abs(cen[:, 0, 0] - cen[:, 1, 0]), np.abs(cen[:, 0, 1] - cen[:, 1, 1])
    nobj = np.array([r["set_size"] for r in meta])
    print(f"Сцен: {n}. Сети: {', '.join(nets)}. Выход: {stem}.csv / .png / _scores.npz")
    print(f"Общее подмножество (разница расстояний до середины ≥ {MARGIN} px по обеим осям): {SUB.sum()} сцен; "
          f"объектов 2/3/4: {', '.join(str(int((nobj[SUB] == k).sum())) for k in (2, 3, 4))}; "
          f"средний разнос |dx| {dx[SUB].mean():.1f} px (все сцены {dx.mean():.1f}), |dy| {dy[SUB].mean():.1f} ({dy.mean():.1f})")
    tw = {q: np.array([query_twin(r, q) for r in meta]) for q in QUERIES}
    agree = [(a, b, (tw[a][SUB] == tw[b][SUB]).mean()) for i, a in enumerate(QUERIES) for b in QUERIES[i + 1:]]
    ok4 = all(abs(p - 0.5) < 0.1 for _, _, p in agree)
    print("Проверка 4 (один и тот же близнец у двух вопросов, на подмножестве; ≈50% = ответы не связаны): " +
          ", ".join(f"{QSHORT[a]}/{QSHORT[b]} {p * 100:.1f}%" for a, b, p in agree) + f"  {'✅' if ok4 else '⚠️ проверь весь вывод'}")
    print(f"Проверка пикселей ✅ (в дальнем фоне S и Sp совпадают во всех сценах)")
    print(f"Проба: StandardScaler + логрег C={pv.C_REG}, {pv.N_FOLDS} фолдов по сценам (как probe_v2.py). Уровень мешка признаков = 50%")

    res, total_warn = [], 0
    for a in nets:
        wt, sc, w = analyse_net(a, meta, base_groups, labels, ctrl, folds)
        res.append((a, wt, sc))
        total_warn += w
        np.savez_compressed(f"{stem}_scores.npz", sub=SUB, nets=np.array([x[0] for x in res]),
                            **{f"{x[0]}|{q}|{g}|L{l:02d}": s for x in res for (q, g, l), s in x[2].items()})
        print(f"  (баллы сцен сохранены в {stem}_scores.npz: {len(res)} из {len(nets)} сетей)", flush=True)

    (a0, w0, s0), rand = res[0], res[1:]
    ref = SHORT.get(w0, f"[{w0}]")
    print(f"\nСравниваем: {w0} ({a0}) против {len(rand)} случайных сетей ({', '.join(x[0] for x in rand) or '—'})")

    # --- проверка 1: «левый» / «верхний» по всем сценам = probe_v2*.csv
    worst, n_cmp, n_all = 0.0, 0, 0
    for q in ("left", "upper"):
        f = REF_CSV[args.model][q]
        refd = {}
        if os.path.exists(f):
            refd = {(r["acts"], r["group"], r["layer"]): float(r["acc"]) for r in csv.DictReader(open(f)) if r["probe"] == "logreg"}
        for a, _, sc in res:
            for g in GROUPS:
                for l in range(pv.N_LAYERS):
                    n_all += 1
                    key = (a, g, f"L{l:02d}")
                    if key in refd:
                        n_cmp += 1
                        worst = max(worst, abs(round(float(np.nanmean(sc[(q, g, l)])), 5) - refd[key]))
    ok1 = n_cmp == n_all and worst < 1e-5
    print(f"\nПроверка 1 («левый»/«верхний» по всем сценам = {REF_CSV[args.model]['left']} / {REF_CSV[args.model]['upper']}): "
          f"сверено {n_cmp} из {n_all} чисел, макс. расхождение {worst:.5f}  {'✅' if ok1 else '⚠️ проверь весь вывод'}")

    # --- проверка 2: L00 ровно 50%
    l00 = [v for _, _, sc in res for q in QUERIES for g in GROUPS
           for v in (np.nanmean(sc[(q, g, 0)]), np.nanmean(sc[(q, g, 0)][SUB]))]
    ok2 = all(abs(v - 0.5) < 1e-12 for v in l00)
    print(f"Проверка 2 (L00: CLS и дальний фон ровно 50,0% у всех вопросов и сетей, все сцены и подмножество): "
          f"{'✅' if ok2 else '⚠️ проверь весь вывод'}  {sorted(set(round(v * 100, 3) for v in l00))}")

    # --- проверка 3: контроль
    cvals = [(a, q, l, float(np.nanmean(sc[(q, 'control', l)]))) for a, _, sc in res for q in QUERIES for l in CONTROL_LAYERS]
    ok3 = all(v > 0.9 for *_, v in cvals)
    print(f"Проверка 3 (контроль: патчи самого близнеца-«вопроса», L00 и L12, > 90%): мин {min(v for *_, v in cvals) * 100:.1f}%  "
          f"{'✅' if ok3 else '⚠️ проверь весь вывод'}")
    print(f"Проверка 5 (предупреждений «обучение не сошлось»): {total_warn}  {'✅' if total_warn == 0 else '⚠️ проверь весь вывод'}")

    # --- парный бутстрэп на подмножестве
    rng = np.random.default_rng(0)
    sub_idx = np.where(SUB)[0]
    bidx = rng.integers(0, len(sub_idx), size=(N_BOOT, len(sub_idx)))

    def stats(sc, g, l):
        """Точности 4 вопросов и разрывы на подмножестве: значение и 95% ДИ (один бутстрэп для всех)."""
        m = {q: sc[(q, g, l)][sub_idx] for q in QUERIES}
        assert not any(np.isnan(v).any() for v in m.values()), "в подмножестве есть сцена без группы"
        val = {q: m[q].mean() for q in QUERIES}
        bs = {q: m[q][bidx].mean(1) for q in QUERIES}
        val["gap_side"], bs["gap_side"] = val["upper"] - val["left"], bs["upper"] - bs["left"]
        val["gap_mid"], bs["gap_mid"] = val["cy"] - val["cx"], bs["cy"] - bs["cx"]
        val["D"], bs["D"] = val["gap_side"] - val["gap_mid"], bs["gap_side"] - bs["gap_mid"]
        return {k: (float(val[k]), float(np.percentile(bs[k], 2.5)), float(np.percentile(bs[k], 97.5))) for k in val}

    table = {(a, g, l): stats(sc, g, l) for a, _, sc in res for g in GROUPS for l in range(pv.N_LAYERS)}
    KEYS = QUERIES + ["gap_side", "gap_mid", "D"]
    with open(f"{stem}.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["acts", "weights", "group", "layer", "subset", "stat", "value", "ci_lo", "ci_hi", "n_scenes"])
        for a, wt, sc in res:
            for g in GROUPS:
                for l in range(pv.N_LAYERS):
                    for q in QUERIES:
                        wr.writerow([a, wt, g, f"L{l:02d}", "all", q, f"{np.nanmean(sc[(q, g, l)]):.5f}", "", "",
                                     int(np.sum(~np.isnan(sc[(q, g, l)])))])
                    for k in KEYS:
                        v, lo, hi = table[(a, g, l)][k]
                        wr.writerow([a, wt, g, f"L{l:02d}", "common", k, f"{v:.5f}", f"{lo:.5f}", f"{hi:.5f}", len(sub_idx)])
            for q in QUERIES:
                for l in CONTROL_LAYERS:
                    wr.writerow([a, wt, "control", f"L{l:02d}", "all", q, f"{np.nanmean(sc[(q, 'control', l)]):.5f}", "", "",
                                 int(np.sum(~np.isnan(sc[(q, 'control', l)])))])

    # --- таблицы
    for g in GROUPS:
        print(f"\n================ {NAMES[g]}: {ref}, общее подмножество ({len(sub_idx)} сцен), логрег, % ================")
        print("слой   левый  верхний  сер.←→  сер.↑↓ | разрыв «сторона» [95% ДИ] | разрыв «середина» [95% ДИ] | D [95% ДИ]")
        for l in range(pv.N_LAYERS):
            t = table[(a0, g, l)]
            f = lambda k: f"{t[k][0] * 100:5.1f} [{t[k][1] * 100:5.1f}…{t[k][2] * 100:5.1f}]"
            print(f"L{l:02d}  " + "  ".join(f"{t[q][0] * 100:5.1f}" for q in QUERIES) +
                  f"  | {f('gap_side')} | {f('gap_mid')} | {f('D')}")
        if rand:
            print("случайные сети на том же подмножестве, L12 (среднее; мин–макс): " + ", ".join(
                f"{QSHORT[q]} {np.mean([table[(a, g, 12)][q][0] for a, _, _ in rand]) * 100:.1f} "
                f"({min(table[(a, g, 12)][q][0] for a, _, _ in rand) * 100:.1f}–{max(table[(a, g, 12)][q][0] for a, _, _ in rand) * 100:.1f})"
                for q in QUERIES))

    # --- итог по правилу, записанному до запуска
    print(f"\nИТОГ на L12 ({ref}) по правилу, записанному ДО запуска:")
    for g in GROUPS:
        t = table[(a0, g, 12)]
        gs, gm, D = t["gap_side"], t["gap_mid"], t["D"]
        cy_lo = t["cy"][1]
        cy_rmax = max(table[(a, g, 12)]["cy"][0] for a, _, _ in rand) if rand else 0.5
        if cy_lo <= cy_rmax:
            verdict = "НЕ ОТВЕЧАЕТ: «ближе к середине по вертикали» не выше всех случайных сетей"
        elif gs[1] <= 0:
            verdict = "НЕ ОТВЕЧАЕТ: разрыва «верхний − левый» на этом подмножестве нет"
        elif D[1] > 0 and gm[0] <= gs[0] / 2:
            verdict = "СКЛАДЫВАНИЕ: разрыв «середина» мал, D > 0"
        elif D[1] <= 0 <= D[2]:
            verdict = "СТИРАНИЕ: разрыв «середина» неотличим от разрыва «сторона» (ДИ для D включает 0)"
        elif D[1] > 0:
            verdict = "ПРОМЕЖУТОЧНО: D > 0, но разрыв «середина» больше половины разрыва «сторона»"
        else:
            verdict = "НЕОЖИДАННО: разрыв «середина» больше разрыва «сторона» — проверь весь вывод"
        print(f"  {NAMES[g]:12s} «сторона» {gs[0] * 100:5.1f} [{gs[1] * 100:.1f}…{gs[2] * 100:.1f}]   "
              f"«середина» {gm[0] * 100:5.1f} [{gm[1] * 100:.1f}…{gm[2] * 100:.1f}]   D {D[0] * 100:5.1f} [{D[1] * 100:.1f}…{D[2] * 100:.1f}]"
              f"   сер.↑↓ {t['cy'][0] * 100:.1f} (нижн. гр. {cy_lo * 100:.1f}; случайные макс {cy_rmax * 100:.1f})\n      → {verdict}")
    print("  (разрывы в п.п.; ДИ — парный бутстрэп по сценам подмножества)")
    print("  Траектория (CLS): " + "; ".join(
        f"{QSHORT[q]} L03 {table[(a0, 'cls', 3)][q][0] * 100:.1f} → L12 {table[(a0, 'cls', 12)][q][0] * 100:.1f}" for q in QUERIES))

    # --- график: сверху — точности 4 вопросов, снизу — разрывы; слева CLS, справа дальний фон
    L = np.arange(pv.N_LAYERS)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), sharex=True)
    style = {"left": ("C3", "--"), "upper": ("C3", "-"), "cx": ("C0", "--"), "cy": ("C0", "-")}
    for j, g in enumerate(GROUPS):
        axa, axg = axes[0, j], axes[1, j]
        for q in QUERIES:
            c, ls = style[q]
            v = [table[(a0, g, l)][q] for l in L]
            axa.plot(L, [x[0] * 100 for x in v], ls, c=c, marker="o", ms=3, label=f"«{QWORD[q]}»")
            axa.fill_between(L, [x[1] * 100 for x in v], [x[2] * 100 for x in v], color=c, alpha=0.08)
        for i, (a, _, _) in enumerate(rand):
            for q in QUERIES:
                axa.plot(L, [table[(a, g, l)][q][0] * 100 for l in L], ":", c="gray", alpha=0.5, lw=1,
                         label="случайные сети (все вопросы)" if i == 0 and q == "left" else None)
        axa.axhline(50, ls=":", c="k", lw=1)
        axa.set_title(f"{NAMES[g]}, {ref}: точность пробы на общем подмножестве ({len(sub_idx)} сцен)", fontsize=9)
        axa.set_ylabel("точность, %  (50 = мешок признаков)")
        axa.legend(fontsize=7)
        axa.grid(alpha=0.3)
        for k, c, lab in (("gap_side", "C3", "разрыв «сторона»: верхний − левый"),
                          ("gap_mid", "C0", "разрыв «середина»: ближе↑↓ − ближе←→")):
            v = [table[(a0, g, l)][k] for l in L]
            axg.plot(L, [x[0] * 100 for x in v], "o-", c=c, ms=3, label=lab)
            axg.fill_between(L, [x[1] * 100 for x in v], [x[2] * 100 for x in v], color=c, alpha=0.15)
        axg.axhline(0, ls=":", c="k", lw=1)
        axg.set_ylabel("разрыв, п.п.")
        axg.set_xticks(L, [f"L{l:02d}" for l in L], fontsize=7)
        axg.legend(fontsize=7)
        axg.grid(alpha=0.3)
    fig.suptitle(f"Опыт «складывание» ({ref}): стирает ли сеть горизонтальную привязку или только «лево/право»?  "
                 "складывание → синий разрыв намного меньше красного; стирание → разрывы одинаковые", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=150)
    print(f"\nГрафик: {stem}.png   Числа: {stem}.csv   Баллы сцен: {stem}_scores.npz")


if __name__ == "__main__":
    main()

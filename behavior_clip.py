"""
behavior_clip.py — Часть 1 «Поведение»: выбирает ли сам CLIP правильную подпись,
если подписи различаются только тем, какой цвет где (слева/справа или сверху/снизу).

Зачем. Все прежние опыты читали привязку пробой. Здесь отвечает сама сеть: CLIP сравнивает
картинку с двумя подписями — правильной и перевёрнутой (цвета близнецов переставлены).

Дизайн (журнал, реестр 96; записан до кода вместе со ставкой).
- Модель: OpenAI CLIP ViT-B/16 — картинка через timm (та же сеть, что в пробах и ablate_clip),
  текст через open_clip «ViT-B-16-quickgelu», openai.
- Данные: data_v2, картинки S и Sp. Сцены: центры близнецов разнесены >= 32 px и по x, и по y
  (оба вопроса на одних и тех же сценах). Справочно — все 2000 сцен (определение «левого» как в probe_v2).
- Подписи: 4 шаблона на ось, сходство (косинус) с правильной и с перевёрнутой подписью усредняется по шаблонам;
  ответ верен, если правильная подпись ближе. «Мешок признаков» даёт ровно 50% по построению
  (одинаковые S и Sp -> одна и та же подпись, верна только для одной; проверка 5).
- Правило 1 (каждая ось, 95%): «выше угадывания», если нижняя граница > 50%
  (ярлык «чуть выше угадывания», если точечная < 55%); иначе «на уровне угадывания».
- Правило 2: Δ = Верт − Гор, парный бутстрэп по сценам (95%): «лево/право хуже» — весь интервал > 0;
  «лучше» — весь < 0; иначе «не различает».
- Дополнительный тест: выключение средним внимания 7–12 и MLP 7–12 (средние — из шага 1, ablate_clip),
  интервал 97,5%: «возвращает» — нижняя граница ΔГор > 0 И нижняя граница (ΔГор − ΔВерт) > 0;
  «не причастна» — верхняя граница (ΔГор − ΔВерт) < 3 п.п.; иначе «не различает».
  Пометка «разрушение», если точность называния цвета (задача clip_check) < 65%.
- Контроли: 3 случайных CLIP (картинная часть случайная, текстовая обученная) — ожидаемо около 50%.
- Бутстрэп: 10 000 повторов, seed 0 (ablate_blocks.boot_paired).

Запуск на Маке:
  caffeinate -dims python behavior_clip.py 2>&1 | tee behavior_clip_log.txt
Проверочный прогон (облако, случайные веса):
  python behavior_clip.py --limit 60 --main-seed 5 --text-random --n-random 1
"""
import argparse
import csv
import hashlib
import json
import os
import time

import numpy as np
import torch
from PIL import Image

import ablate_blocks as ab                 # бутстрэп, правило, средние (файл не меняется)
import ablate_clip as ac                   # прогон CLIP с выключением и проверки 1–3 (файл не меняется)
from extract import load_model, fingerprint, pick_device

DATA = ab.DATA
SEP = 32                                   # px: минимальный разнос центров близнецов по каждой оси
BLOCKS = set(range(7, 13))
CONDS = [("clean", set(), set()), ("attn_07-12", BLOCKS, set()), ("mlp_07-12", set(), BLOCKS)]
ALPHA = 0.05                               # правила 1 и 2
ALPHA_ABL = 0.05 / 2                       # 2 условия выключения
COLOR_BROKEN = 0.65
EXPECTED_FP = ac.EXPECTED_FP
MEANS_STEP1 = os.path.expanduser("~/binding-vit-data/ablate_clip/CLIP_means.pt")

# {A} — цвет близнеца слева (Гор) или сверху (Верт), {B} — другого; {s} — форма
TEMPL = {
    "h": ["a {A} {s} on the left and a {B} {s} on the right",
          "a {B} {s} on the right and a {A} {s} on the left",
          "a photo of a {A} {s} to the left of a {B} {s}",
          "a photo of a {B} {s} to the right of a {A} {s}"],
    "v": ["a {A} {s} on top and a {B} {s} at the bottom",
          "a {B} {s} at the bottom and a {A} {s} on top",
          "a photo of a {A} {s} above a {B} {s}",
          "a photo of a {B} {s} below a {A} {s}"],
}
AXIS = {"h": "Гор (лево/право)", "v": "Верт (верх/низ)"}
# ставка Ерназара (реестр 96, 12.11 10:21) и засчёт, объявленный до кода
BET = dict(h=(50.0, 55.0, "Б"), v=(55.1, 60.0, "В"), rule2="не различает", attn="не различает")


# ------------------------------------------------------------------
# 1. Сцены, подписи
# ------------------------------------------------------------------
def scene_rows(meta, only_sep):
    """Для каждой сцены: (индекс, форма, {версия: (цвет слева, цвет справа, цвет сверху, цвет снизу)})."""
    out = []
    for i, r in enumerate(meta):
        assert r["twins"] == [0, 1], f"{r['id']}: близнецы не объекты 0 и 1"
        o0, o1 = r["objects"][0], r["objects"][1]
        (x0, y0), (x1, y1) = o0["center"], o1["center"]
        if only_sep and (abs(x0 - x1) < SEP or abs(y0 - y1) < SEP):
            continue
        L, R = (o0, o1) if (x0, y0) < (x1, y1) else (o1, o0)      # как probe_v2: меньше x, при равенстве меньше y
        U, D = (o0, o1) if (y0, x0) < (y1, x1) else (o1, o0)      # меньше y = выше на картинке
        assert L["color_Sp"] == R["color"] and R["color_Sp"] == L["color"], f"{r['id']}: в Sp цвета не переставлены"
        col = {"S": (L["color"], R["color"], U["color"], D["color"]),
               "Sp": (L["color_Sp"], R["color_Sp"], U["color_Sp"], D["color_Sp"])}
        out.append((i, r["twin_shape"], col))
    return out


def caption_sets(rows):
    """Для каждой картинки (сцена × версия) и оси: 4 правильные и 4 перевёрнутые подписи."""
    caps = {"h": [], "v": []}
    for _, s, col in rows:
        for v in ("S", "Sp"):
            l, r, u, d = col[v]
            for ax, (a, b) in (("h", (l, r)), ("v", (u, d))):
                caps[ax].append(([t.format(A=a, B=b, s=s) for t in TEMPL[ax]],
                                 [t.format(A=b, B=a, s=s) for t in TEMPL[ax]]))
    return caps


@torch.no_grad()
def encode_texts(oc, tok, strings):
    uniq = sorted(set(strings))
    E = []
    for k in range(0, len(uniq), 256):
        e = oc.encode_text(tok(uniq[k:k + 256])).float()
        E.append(e / e.norm(dim=-1, keepdim=True))
    E = torch.cat(E)
    return {s: E[k] for k, s in enumerate(uniq)}


def scores(img_emb, img_idx, caps, T):
    """Разность средних косинусов (правильная − перевёрнутая) для каждой картинки; > 0 — ответ верен."""
    out = np.zeros(len(caps))
    for k, ((good, bad), ii) in enumerate(zip(caps, img_idx)):
        e = img_emb[ii]
        out[k] = float(torch.stack([T[t] for t in good]).mv(e).mean() - torch.stack([T[t] for t in bad]).mv(e).mean())
    return out


# ------------------------------------------------------------------
# 2. Картинки -> эмбеддинги CLIP (с выключением частей блоков)
# ------------------------------------------------------------------
@torch.no_grad()
def image_emb(model, imgs, device, a_off, m_off, means):
    out = []
    for s in range(0, len(imgs), ab.BATCH):
        x = ac.norm_batch(imgs[s:s + ab.BATCH]).to(device)
        _, h = ac.forward(model, x, a_off, m_off, means)
        e = model.forward_head(model.norm(h)).float().cpu()
        out.append(e / e.norm(dim=-1, keepdim=True))
    return torch.cat(out)


@torch.no_grad()
def compute_means(model, imgs, device):
    acc = ab.new_acc()
    for s in range(0, len(imgs), ab.BATCH):
        x = ac.norm_batch(imgs[s:s + ab.BATCH]).to(device)
        ac.forward(model, x, acc=acc)
        acc["n"] += len(x)
    return ab.to_means(acc)


@torch.no_grad()
def color_naming(emb_S, meta_S, T_col, colors):
    """Задача clip_check: цвет с наибольшим сходством «a photo of a {c} object» есть на сцене."""
    M = torch.stack([T_col[f"a photo of a {c} object"] for c in colors])
    top = (emb_S @ M.T).argmax(-1).tolist()
    return float(np.mean([colors[k] in {o["color"] for o in r["objects"]} for r, k in zip(meta_S, top)]))


# ------------------------------------------------------------------
# 3. Подсчёт и правила
# ------------------------------------------------------------------
def per_scene(sc):
    """sc: разности по картинкам в порядке (сцена: S, Sp) -> доля верных по сцене (0, 0.5, 1)."""
    return (sc.reshape(-1, 2) > 0).mean(1)


def rule1(t):
    m, lo, _ = t
    if lo > 0.5:
        return "чуть выше угадывания" if m < 0.55 else "выше угадывания"
    return "на уровне угадывания"


def rule2(t):
    if t[1] > 0:
        return "лево/право хуже"
    if t[2] < 0:
        return "лучше"
    return "не различает"


def abl_verdict(dh, dd):
    v = ab.verdict(dh, dd)                 # то же правило, что в Части 3
    return {"стирает": "возвращает"}.get(v, v)


def pct(t):
    return f"{100 * t[0]:5.1f} [{100 * t[1]:5.1f} … {100 * t[2]:5.1f}]"


def dpct(t):
    return f"{100 * t[0]:+5.1f} [{100 * t[1]:+5.1f} … {100 * t[2]:+5.1f}]"


# ------------------------------------------------------------------
# 4. Встроенные проверки
# ------------------------------------------------------------------
@torch.no_grad()
def check_open_clip(model, oc, imgs, device):
    x = ac.norm_batch(imgs[:8])
    a = image_emb(model, imgs[:8], device, set(), set(), None)
    b = oc.encode_image(x).float()
    b = b / b.norm(dim=-1, keepdim=True)
    c = float((a * b).sum(-1).min())
    print(f"Проверка 4 (картинка: наш прогон timm = open_clip.encode_image): косинус мин {c:.6f}  "
          f"{'✅' if c >= 0.9999 else '⚠️ проверь весь вывод'}")


def check_bag(sc_h, sc_v, emb, img_idx, caps, T):
    """Проверка 5: если эмбеддинг Sp заменить эмбеддингом S, в каждой сцене верна ровно одна из двух картинок."""
    idx = np.array(img_idx).copy()
    idx[1::2] = idx[0::2]
    ok = True
    for ax in ("h", "v"):
        s = scores(emb, idx, caps[ax], T)
        ok &= bool(np.all(np.abs(s[0::2] + s[1::2]) < 1e-6)) and bool(np.all((s.reshape(-1, 2) > 0).sum(1) == 1))
    print(f"Проверка 5 («мешок признаков»: одинаковые S и Sp -> ровно 50% в каждой сцене): "
          f"{'✅' if ok else '⚠️ проверь весь вывод'}")
    # Проверка 6: «идеальная картинка» = направление (среднее правильных − среднее перевёрнутых)
    # -> разность обязана быть > 0 у каждой картинки (100% верных); проверяет знак и индексы подсчёта
    ok6 = True
    for ax in ("h", "v"):
        E = torch.stack([torch.stack([T[t] for t in g]).mean(0) - torch.stack([T[t] for t in b]).mean(0)
                         for g, b in caps[ax]])
        E = E / E.norm(dim=-1, keepdim=True)
        ok6 &= bool(np.all(scores(E, list(range(len(E))), caps[ax], T) > 0))
    print(f"Проверка 6 (подсчёт: «картинка» = направление правильная − перевёрнутая -> 100% верных): "
          f"{'✅' if ok6 else '⚠️ проверь весь вывод'}")


# ------------------------------------------------------------------
# 5. main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="только для проверки: первые N сцен")
    ap.add_argument("--main-seed", type=int, default=None, help="только для проверки: случайные веса вместо CLIP")
    ap.add_argument("--text-random", action="store_true", help="только для проверки в облаке: текстовая часть без весов")
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--cache", default=os.path.expanduser("~/binding-vit-data/behavior_clip"))
    args = ap.parse_args()
    test = bool(args.limit or args.main_seed is not None or args.text_random)
    stem = "behavior_clip" + (f"_test{args.limit}" if test else "")
    tag = (f"test{args.limit}_s{args.main_seed}_" if test else "")
    t_start = time.time()
    torch.manual_seed(0)

    meta_all = [json.loads(l) for l in open(f"{DATA}/meta.jsonl")]
    meta = meta_all[:args.limit] if args.limit else meta_all
    if test:
        print(f"⚠️ ПРОВЕРОЧНЫЙ прогон ({len(meta)} сцен; случайные веса). Цифры — не результат.")
    md5 = hashlib.md5(open(f"{DATA}/meta.jsonl", "rb").read()).hexdigest()
    print(f"data_v2/meta.jsonl md5 {md5}")
    rows_sep = scene_rows(meta, True)
    rows_all = scene_rows(meta, False)
    print(f"Сцен всего: {len(rows_all)}; с разносом >= {SEP} px и по x, и по y: {len(rows_sep)} "
          f"(картинок {2 * len(rows_sep)})")
    colors = sorted({o[k] for r in meta_all for o in r["objects"] for k in ("color", "color_Sp")})

    imgs = np.zeros((2 * len(meta), 224, 224, 3), np.uint8)
    for i, rec in enumerate(meta):
        for k, v in enumerate(("S", "Sp")):
            imgs[2 * i + k] = np.asarray(Image.open(f"{DATA}/images/{rec['id']}_{v}.png").convert("RGB"))
    idx_sep = [2 * i + k for i, _, _ in rows_sep for k in (0, 1)]
    idx_all = [2 * i + k for i, _, _ in rows_all for k in (0, 1)]
    caps_sep, caps_all = caption_sets(rows_sep), caption_sets(rows_all)

    import open_clip
    oc, _, _ = open_clip.create_model_and_transforms("ViT-B-16-quickgelu",
                                                     pretrained=None if args.text_random else "openai")
    oc.eval()
    tok = open_clip.get_tokenizer("ViT-B-16-quickgelu")
    strings = [t for ax in caps_all for g, b in caps_all[ax] for t in g + b] + \
              [f"a photo of a {c} object" for c in colors]
    T = encode_texts(oc, tok, strings)
    print(f"Текст: open_clip ViT-B-16-quickgelu ({'БЕЗ весов — проверка' if args.text_random else 'openai'}); "
          f"разных подписей {len(T)}; шаблонов на ось {len(TEMPL['h'])}")
    for ax in ("h", "v"):
        print(f"  {AXIS[ax]}: " + " | ".join(TEMPL[ax]))

    device = pick_device()
    os.makedirs(args.cache, exist_ok=True)
    nets = [("CLIP", args.main_seed)] + [(f"rand{s}", s) for s in range(args.n_random)]
    res, csv_rows = {}, []
    for name, seed in nets:
        model, weights, _ = load_model("clip", seed)
        fp = fingerprint(model)
        note = ""
        if name.startswith("rand") and seed in EXPECTED_FP:
            note = "✅ совпадает с extract.py" if abs(fp - EXPECTED_FP[seed]) < 0.01 else "⚠️ НЕ совпадает — проверь весь вывод"
        print(f"\n### {name}: {weights}   отпечаток весов {fp:.4f}  {note}   устройство: {device}")
        model.eval().to(device)
        conds = CONDS if name == "CLIP" else CONDS[:1]
        means = None
        if name == "CLIP":
            ac.run_checks(model, imgs, device)
            if not args.text_random:
                check_open_clip(model, oc, imgs, device)
            if not test and os.path.exists(MEANS_STEP1):
                means = torch.load(MEANS_STEP1)
                print(f"  средние вкладов — из шага 1 ({MEANS_STEP1}, md5 "
                      f"{hashlib.md5(open(MEANS_STEP1, 'rb').read()).hexdigest()}; 4000 картинок, обычная CLIP)")
            else:
                means = compute_means(model, imgs, device)
                print(f"  средние вкладов посчитаны заново на {len(imgs)} картинках (обычная сеть)")
        res[name] = {}
        for cname, a_off, m_off in conds:
            p = f"{args.cache}/{tag}{name}_{cname}.npz"
            if os.path.exists(p) and int(np.load(p)["fp_ok"]) == int(round(fp, 4) * 10000):
                emb = torch.from_numpy(np.load(p)["emb"])
            else:
                t1 = time.time()
                emb = image_emb(model, imgs, device, a_off, m_off, means)
                np.savez(p, emb=emb.numpy(), fp_ok=int(round(fp, 4) * 10000))
                print(f"  {cname:11s} эмбеддинги: {time.time() - t1:4.0f} с", flush=True)
            r = {}
            for key, idx, caps in (("sep", idx_sep, caps_sep), ("all", idx_all, caps_all)):
                r[key] = {ax: scores(emb, idx, caps[ax], T) for ax in ("h", "v")}
            r["color"] = color_naming(emb[0::2], meta, T, colors)
            if name == "CLIP" and cname == "clean":
                check_bag(r["sep"]["h"], r["sep"]["v"], emb, idx_sep, caps_sep, T)
            res[name][cname] = r
        del model

    # ---------------- отчёт
    print("\n" + "=" * 78)
    print(f"РЕЗУЛЬТАТ (сцены с разносом >= {SEP} px по обеим осям, n = {len(rows_sep)}; 95% интервал, бутстрэп по сценам)")
    print("=" * 78)
    out = {}
    for name, _ in nets:
        for cname in res[name]:
            r = res[name][cname]
            ph, pv_ = per_scene(r["sep"]["h"]), per_scene(r["sep"]["v"])
            th, tv, td = ab.boot_paired([ph, pv_, pv_ - ph], ALPHA)
            one = float(np.mean((r["sep"]["h"].reshape(-1, 2) > 0).sum(1) == 1))
            out[(name, cname)] = (ph, pv_)
            print(f"{name:6s} {cname:11s} Гор {pct(th)}   Верт {pct(tv)}   Верт−Гор {dpct(td)}   "
                  f"называние цвета {100 * r['color']:.1f}%")
            if name == "CLIP" and cname == "clean":
                print(f"   доля сцен, где по Гор верна ровно одна из S/Sp (как у «мешка»): {100 * one:.1f}%")
            for ax, t in (("h", th), ("v", tv)):
                csv_rows.append([name, cname, ax, "sep", f"{t[0]:.5f}", f"{t[1]:.5f}", f"{t[2]:.5f}", len(rows_sep)])
            csv_rows.append([name, cname, "v-h", "sep", f"{td[0]:.5f}", f"{td[1]:.5f}", f"{td[2]:.5f}", len(rows_sep)])
            csv_rows.append([name, cname, "color", "S", f"{r['color']:.5f}", "", "", len(meta)])

    ph, pv_ = out[("CLIP", "clean")]
    th, tv, td = ab.boot_paired([ph, pv_, pv_ - ph], ALPHA)
    print("\nПравило 1 (каждая ось):")
    print(f"  Гор:  {pct(th)} -> «{rule1(th)}»")
    print(f"  Верт: {pct(tv)} -> «{rule1(tv)}»")
    print(f"Правило 2 (Δ = Верт − Гор): {dpct(td)} -> «{rule2(td)}»")

    print(f"\nДополнительный тест: выключение средним, блоки 7–12 (интервал 97,5%)")
    verd = {}
    for cname in ("attn_07-12", "mlp_07-12"):
        qh, qv = out[("CLIP", cname)]
        dh, dv, dd = ab.boot_paired([qh - ph, qv - pv_, (qh - ph) - (qv - pv_)], ALPHA_ABL)
        broken = res["CLIP"][cname]["color"] < COLOR_BROKEN
        verd[cname] = abl_verdict(dh, dd)
        print(f"  {cname}: ΔГор {dpct(dh)}   ΔВерт {dpct(dv)}   ΔГор−ΔВерт {dpct(dd)} -> «{verd[cname]}»"
              f"{'   ⚠️ пометка «разрушение» (называние цвета < 65%)' if broken else ''}")
        for k, t in (("d_h", dh), ("d_v", dv), ("d_h-d_v", dd)):
            csv_rows.append(["CLIP", cname, k, "sep", f"{t[0]:.5f}", f"{t[1]:.5f}", f"{t[2]:.5f}", len(rows_sep)])

    print(f"\nСправка (НЕ правило): все {len(rows_all)} сцен, обычная CLIP")
    ah, av = per_scene(res["CLIP"]["clean"]["all"]["h"]), per_scene(res["CLIP"]["clean"]["all"]["v"])
    tah, tav = ab.boot_paired([ah, av], ALPHA)
    print(f"  Гор {pct(tah)}   Верт {pct(tav)}")
    for ax, t in (("h", tah), ("v", tav)):
        csv_rows.append(["CLIP", "clean", ax, "all", f"{t[0]:.5f}", f"{t[1]:.5f}", f"{t[2]:.5f}", len(rows_all)])

    print("\nСтавка Ерназара (реестр 96, 12.11 10:21; засчёт объявлен до кода):")
    hits = 0
    for key, label, val in (("h", "96a Гор", th), ("v", "96b Верт", tv)):
        lo, hi, letter = BET[key]
        x = round(100 * val[0], 1)
        ok = lo <= x <= hi
        hits += ok
        print(f"  {label}: {letter} [{lo}; {hi}] -> {x} {'✅' if ok else '❌'}")
    ok = rule2(td) == BET["rule2"]; hits += ok
    print(f"  96c правило 2: «{BET['rule2']}» -> «{rule2(td)}» {'✅' if ok else '❌'}")
    ok = verd["attn_07-12"] == BET["attn"]; hits += ok
    print(f"  96d внимание 7–12: «{BET['attn']}» -> «{verd['attn_07-12']}» {'✅' if ok else '❌'}")
    print(f"  попаданий: {hits} из 4")

    with open(f"{stem}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["net", "condition", "measure", "scenes", "value", "ci_lo", "ci_hi", "n"])
        w.writerows(csv_rows)
    print(f"\nФайлы: {stem}.csv; кэш эмбеддингов {args.cache}/")
    print(f"Всего: {time.time() - t_start:.0f} с")


if __name__ == "__main__":
    main()

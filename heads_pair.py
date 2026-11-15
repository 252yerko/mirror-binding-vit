"""
heads_pair.py — опыт G: «симметричная пара голов». Добавлено 15.11 (реестр 198–199, ставка Ерназара — до кода).

ОТКУДА ВОПРОС (разведочный прогон heads_stripes.py, после опыта F, osf.io/gdnvb):
  головы DINO 4.4 и 4.5 («полосы вдоль строки», блок 4) при выключении средним по одной поднимают точность пробы
  «левый» (+2,00 и +3,30 п.п.), а выключенные вместе — нет (+0,62). Гипотеза: 4.4 и 4.5 — зеркальная друг другу пара;
  симметричная пара не создаёт сигнала стороны, одна голова без пары создаёт.
  ⚠️ Головы 4.4 и 4.5 выбраны ПОСЛЕ того, как увиден результат (на 2000 сценах data_v2). Поэтому причинная часть (G2)
  считается на 2000 НОВЫХ сценах (generate_v2.py --seed 2 -> data_g), а G1 смотрит на другое свойство (карты внимания).

СЕТИ: DINO B/16 (G1, G2), DeiT III B/16 и CLIP-OpenAI B/16 (G3); по 3 случайные сети каждой архитектуры (seed 0–2).

ВЕЛИЧИНЫ ПО КАРТАМ ВНИМАНИЯ (631 сцена data_v2 только из круга / квадрата / креста; картинка S И её отражение ←→):
  карта головы A[q, k] — только патчи (196 × 196), без самого патча (A[q, q] = 0), строки заново нормированы.
  dx — горизонтальное смещение: Σ_k A[q, k]·(столбец k − столбец q), среднее по всем 196 запросам q (сетка симметрична
       ←→, поэтому равномерное внимание даёт ровно 0), затем среднее по картинке и её отражению (внимание, которое
       идёт за содержимым сцены, при отражении меняет знак и гасится; остаётся собственный «взгляд влево / вправо»).
       dy — то же по вертикали (справка). 95% интервал — бутстрэп по сценам (1000, seed 0).
  S_прям(a, b) — корреляция Пирсона карт голов a и b на одной картинке (среднее по картинке и отражению);
  S_зерк(a, b) — корреляция карты a на картинке x с отражённой ←→ картой b на отражённой картинке
       (и наоборот, среднее): «голова a делает на x то же, что b на зеркале»;
  D(a, b) = S_зерк − S_прям — «зеркальный избыток»; печатается ВМЕСТЕ с частями. Интервал — бутстрэп по сценам.

ПРАВИЛА (записаны ДО запуска):
  G1 (DINO, пара 4.4 / 4.5) — «зеркальная пара», если выполнены все три условия, иначе «не показано»:
     (i)   dx у 4.4 и 4.5 противоположны по знаку, у каждой 95% интервал не включает 0 и |dx| больше наибольшего |dx|
           среди 3 × 144 голов случайных сетей той же архитектуры;
     (ii)  нижняя граница 95% интервала D(4.4, 4.5) > 0;
     (iii) D(4.4, 4.5) входит в тройку наибольших среди 330 пар голов внутри блоков 2–6
           (не больше 2 из остальных 329 пар имеют D не меньше).
     [Изменение после ставки, до регистрации: в (iii) сравнивается D, а не S_зерк — две одинаковые симметричные
      головы имеют S_зерк ≈ 1 и вытеснили бы настоящую зеркальную пару; ранг по S_зерк печатается справкой.
      В (i) и в G3 добавлен порог |dx| по случайным сетям: в облачном тесте у случайной сети интервал dx не включал 0
      при dx ≈ 0,01 патча, а «больше максимума трёх случайных сетей» четвёртая случайная сеть проходит с шансом ≈ 1/4.
      Условие (iii) консервативно: при трёх и более парах с большим D в блоках 2–6 пара не проходит.]
  G2 (DINO, 2000 новых сцен; дальний фон, L12; пробы обучаются заново) — resample-выключение: выход головы
     (до общей проекции, все токены) заменяется её выходом на ДРУГОЙ сцене (та же версия S / Sp; донор — по случайной
     перестановке без неподвижных точек; 3 перестановки, seed 0–2; точности усредняются по перестановкам,
     интервал — бутстрэп по сценам). Δлевый — изменение точности пробы «левый» против обычной сети.
     «Картина сохранилась», если (а) нижняя граница Δлевый > 0 и у 4.4, и у 4.5 по одной, и
     (б) Δлевый при выключении обеих (один донор на обе головы) < половины меньшего из двух одиночных; иначе «не сохранилась».
     Специфичность: контрольные головы блока 4 — 4.8, 4.7, 4.9 (наименьшие |m| в опыте F, выбраны до запуска).
     «Гашение специфично для пары», если во всех 6 парах (4.4 или 4.5) + контрольная Δлевый ≥ половины Δлевый
     соответствующей одиночной; иначе «не показано» (не оценивается, если (а) не выполнено).
  G3 (DeiT III, CLIP; справочно и DINO) — порог T = наибольшее D среди всех пар голов внутри блоков (12 × 66)
     трёх случайных сетей той же архитектуры. «Зеркальная пара найдена», если есть хотя бы одна пара с
     противоположными dx (как в G1 (i): интервалы не включают 0, |dx| больше наибольшего у случайных сетей) и D > T. Правило мягкое (головы «влево / вправо» у обученных ViT
     известны), поэтому рядом печатаются число таких пар и доля голов, имеющих зеркального партнёра.
  Справка (не правила): dy; ранг S_зерк; сходство «сама с собой» S_зерк(a, a); то же для ВЫХОДОВ голов
     (косинус вклада головы a в поток на x с отражённым вкладом b на отражённой картинке, по патчам, за вычетом
     среднего по патчам; замечание Ерназара 4);
     число пар блоков 2–6, для которых выполнены (i) и (ii); CLS как второй считыватель; «верхний» как контроль;
     пара с разными донорами; контрольные головы по одной; неаддитивность Δлевый(4.4) + Δлевый(4.5) − Δлевый(пара)
     с интервалом (бутстрэп по сценам); выключение средним (как в опыте F) на новых сценах.
  Ставки Ерназара (реестр 199, 15.11 21:02, до кода): 199a G1 — пара зеркальная; 199b G2 — картина сохранится;
     199c G3 — у DeiT III пара найдётся; 199d G3 — у CLIP пара не найдётся.

ВСТРОЕННЫЕ ПРОВЕРКИ (все должны пройти):
  1. Прогон с явным вниманием = run_layers (относительная разница < 1e-4); строки внимания в сумме 1.
  2. Resample с донором «сама картинка» = обычный прогон (относительная разница < 1e-4).
  3. Правило G1 на искусственных головах с известным ответом (пары «влево / вправо» проходят; одинаковые симметричные
     полосы, полосы по столбцу, локальные, «смотрит на объект», шум — нет) и правило G2 на искусственных числах.
  4. Сетка, служебные токены, блоки; нормировка внутри внимания — пустая.
  5. Новые сцены: L00 — пробы ровно 50,0%; контроль «близнец-вопрос» > 90%; обычная сеть в пределах ± 5 п.п. от data_v2.
  6. Доноры: неподвижных точек нет, три перестановки различны.
  7. Случайные сети отличаются от обученной (отпечаток весов).
  8. Новые сцены не совпадают ни с одной сценой data_v2.
  9. Мера m опыта F, пересчитанная в этом прогоне, совпадает с кэшем опыта F (относительная разница < 1e-3).
  10. Предупреждений «обучение не сошлось» — 0.

ЗАПУСК (из папки binding-vit):
  python generate_v2.py --n 2000 --seed 2 --out data_g                                  # ~1 мин, новые сцены
  caffeinate -i python heads_pair.py --part maps 2>&1 | tee heads_pair_maps_log.txt     # ~40 мин (G1, G3)
  caffeinate -i python heads_pair.py --part ablate 2>&1 | tee heads_pair_g2_log.txt     # ~75 мин (G2)
  -> heads_pair_heads.csv, heads_pair_pairs.csv, heads_pair_g2.csv, heads_pair_rules_*.csv; кэш — ~/binding-vit-data/heads/.
"""
import argparse
import csv
import hashlib
import json
import os
import time

import numpy as np
import torch

import heads as H
import many_nets as mn
import mirror_decomp as md
import probe_big as pb
from extract import run_layers, fingerprint, pick_device

G, NB, NH = H.G, H.NB, H.NH
PAIR = [(4, 4), (4, 5)]
CTRL = [(4, 8), (4, 7), (4, 9)]
G1_BLOCKS = (2, 3, 4, 5, 6)
ALL_BLOCKS = tuple(range(1, NB + 1))
DATA_OLD, DATA_NEW = "data_v2", "data_g"
PERMS = (0, 1, 2)
N_BOOT = 1000
BATCH_MAPS = 8
V2_CLEAN = (64.75, 83.50)            # обычная DINO на data_v2 (опыт F): «левый», «верхний», дальний фон, L12
OUT_HEADS, OUT_PAIRS, OUT_G2 = "heads_pair_heads.csv", "heads_pair_pairs.csv", "heads_pair_g2.csv"

MHt = torch.as_tensor(H.MH)
QSt = torch.as_tensor(H.QS)
COLt = torch.as_tensor(H.C_, dtype=torch.float64)
ROWt = torch.as_tensor(H.R_, dtype=torch.float64)
mark = lambda b: "✅" if b else "⚠️ проверь весь вывод"


def set_data(name):
    H.DATA = md.DATA = pb.DATA = name


# ------------------------------------------------------------------
# величины по картам внимания
# ------------------------------------------------------------------
def prep_maps(Ap):
    """[..., 196, 196] (только патчи) -> без самого патча, строки заново нормированы (float64)."""
    A = Ap.double().clone()
    A[..., QSt, QSt] = 0
    return A / A.sum(-1, keepdim=True).clamp_min(1e-30)


def offsets(A):
    """Средние смещения внимания (dx, dy) формы [...]."""
    return ((A @ COLt) - COLt).mean(-1), ((A @ ROWt) - ROWt).mean(-1)


def flip_map(A):
    """Карта, отражённая ←→: запросы и ключи переставлены зеркально."""
    return A[..., MHt, :][..., :, MHt]


def unit(V, centre):
    if centre:
        V = V - V.mean(-1, keepdim=True)
    return V / V.norm(dim=-1, keepdim=True).clamp_min(1e-30)


def pair_sims(X, Y, centre):
    """X: [B, H, L] на картинке; Y: [B, H, L] на отражённой картинке, отражённое обратно.
    S_прям [B, H, H] (среднее по картинке и отражению), S_зерк [B, H, H] (симметричная)."""
    Xn, Yn = unit(X, centre), unit(Y, centre)
    Sd = (Xn @ Xn.transpose(-1, -2) + Yn @ Yn.transpose(-1, -2)) / 2
    M = Xn @ Yn.transpose(-1, -2)
    return Sd, (M + M.transpose(-1, -2)) / 2


def block_stats(Ap):
    """Ap: [2B, H, 196, 196] — первая половина картинки, вторая — их отражения. -> dx, dy [B, H]; S_прям, S_зерк [B, H, H]."""
    B = Ap.shape[0] // 2
    A = prep_maps(Ap)
    dx, dy = offsets(A)
    X = A[:B].reshape(B, A.shape[1], -1)
    Y = flip_map(A[B:]).reshape(B, A.shape[1], -1)
    Sd, Sm = pair_sims(X, Y, True)
    return (dx[:B] + dx[B:]) / 2, (dy[:B] + dy[B:]) / 2, Sd, Sm


@torch.no_grad()
def maps_pass(model, arrs, device, mean, std, n_pre, with_out=False):
    """По сценам: dx, dy, m [n, 12, 12]; Sd, Sm (карты) и Od, Om (выходы голов) [n, 12, 12, 12]."""
    n = len(arrs)
    R = {k: np.zeros((n, NB, NH), np.float32) for k in ("dx", "dy", "m")}
    R.update({k: np.zeros((n, NB, NH, NH), np.float32) for k in ("Sd", "Sm")})
    if with_out:
        R.update({k: np.zeros((n, NB, NH, NH), np.float32) for k in ("Od", "Om")})
    worst = [0.0]
    for s in range(0, n, BATCH_MAPS):
        part = arrs[s:s + BATCH_MAPS]
        B = len(part)
        x = H.to_x(list(part) + list(part), mean, std, flips=["o"] * B + ["h"] * B).to(device)

        def cb(b, att, v, a):
            worst[0] = max(worst[0], float((att.sum(-1) - 1).abs().max()))
            Ap = att[:, :, n_pre:, n_pre:].float().cpu()
            dx, dy, Sd, Sm = block_stats(Ap)
            R["dx"][s:s + B, b - 1], R["dy"][s:s + B, b - 1] = dx.numpy(), dy.numpy()
            R["Sd"][s:s + B, b - 1], R["Sm"][s:s + B, b - 1] = Sd.numpy(), Sm.numpy()
            R["m"][s:s + B, b - 1] = H.m_from_attn(Ap[:B])[0].numpy()
            if with_out:
                W = a.proj.weight.reshape(-1, a.num_heads, a.head_dim)                 # [D, H, hd]
                o = (att @ v)[:, :, n_pre:]                                             # [2B, H, 196, hd]
                out = model.blocks[b - 1].ls1(torch.einsum("bhne,dhe->bhnd", o, W))     # вклад головы в поток
                out = out - out.mean(2, keepdim=True)                                   # без части, общей для всех патчей
                X = out[:B].reshape(B, NH, -1)
                Y = out[B:][:, :, MHt.to(out.device)].reshape(B, NH, -1)
                Od, Om = pair_sims(X, Y, False)
                R["Od"][s:s + B, b - 1], R["Om"][s:s + B, b - 1] = Od.float().cpu().numpy(), Om.float().cpu().numpy()

        H.forward(model, x, attn_cb=cb)
    return R, worst[0]


def boot_mean(X, seed=0, n=N_BOOT):
    """X: [сцены, ...] -> среднее, низ, верх (бутстрэп по сценам)."""
    rng = np.random.default_rng(seed)
    N = len(X)
    flat = X.reshape(N, -1).astype(np.float64)
    bs = np.empty((n, flat.shape[1]))
    for i in range(n):
        bs[i] = flat[rng.integers(0, N, N)].mean(0)
    sh = X.shape[1:]
    return flat.mean(0).reshape(sh), np.percentile(bs, 2.5, axis=0).reshape(sh), np.percentile(bs, 97.5, axis=0).reshape(sh)


def summarize(R):
    st = {}
    st["dx"], st["dx_lo"], st["dx_hi"] = boot_mean(R["dx"])
    st["dy"], st["dy_lo"], st["dy_hi"] = boot_mean(R["dy"])
    st["D"], st["D_lo"], st["D_hi"] = boot_mean(R["Sm"].astype(np.float64) - R["Sd"])
    st["Sd"], st["Sm"] = R["Sd"].mean(0, dtype=np.float64), R["Sm"].mean(0, dtype=np.float64)
    if "Od" in R:
        st["Od"], st["Om"] = R["Od"].mean(0, dtype=np.float64), R["Om"].mean(0, dtype=np.float64)
    return st


def sign_ci(st, b, h):
    """Знак dx: интервал не включает 0 И |dx| больше наибольшего |dx| у голов случайных сетей (st["dx_thr"], по умолчанию 0)."""
    if abs(st["dx"][b - 1, h]) <= st.get("dx_thr", 0.0):
        return 0
    return 1 if st["dx_lo"][b - 1, h] > 0 else (-1 if st["dx_hi"][b - 1, h] < 0 else 0)


def opposite(st, b, i, j):
    return sign_ci(st, b, i) * sign_ci(st, b, j) == -1


def within_pairs(blocks):
    return [(b, i, j) for b in blocks for i in range(NH) for j in range(i + 1, NH)]


def g1_rule(st, pair, blocks):
    b, i, j = pair[0][0], pair[0][1], pair[1][1]
    d = st["D"][b - 1, i, j]
    others = [st["D"][bb - 1, x, y] for bb, x, y in within_pairs(blocks) if (bb, x, y) != (b, min(i, j), max(i, j))]
    n_ge = int(sum(o >= d for o in others))
    c1, c2, c3 = opposite(st, b, i, j), bool(st["D_lo"][b - 1, i, j] > 0), n_ge <= 2
    return dict(c1=c1, c2=c2, c3=c3, n_ge=n_ge, n_others=len(others), ok=c1 and c2 and c3)


def g3_pairs(st, thr, blocks=ALL_BLOCKS):
    return [(b, i, j) for b, i, j in within_pairs(blocks) if opposite(st, b, i, j) and st["D"][b - 1, i, j] > thr]


def g2_verdict(s44, s45, pr, ctrl_pairs):
    """s44, s45: (оценка, низ) Δлевый по одной; pr: оценка Δлевый пары; ctrl_pairs: [(какая одиночная 0 / 1, оценка)]."""
    c_single = s44[1] > 0 and s45[1] > 0
    c_pair = pr < 0.5 * min(s44[0], s45[0])
    if c_single and c_pair:
        v = "картина сохранилась (по одной +, вместе гасится)"
    elif not c_single:
        v = "не сохранилась: эффекта по одной нет"
    else:
        v = "не сохранилась: вместе не гасится"
    if not c_single:
        spec = "не оценивается"
    else:
        spec = "гашение специфично для пары" if all(d >= 0.5 * (s44[0], s45[0])[k] for k, d in ctrl_pairs) else "не показано"
    return v, spec, c_single, c_pair


# ------------------------------------------------------------------
# проверка 3: искусственные головы и искусственные числа
# ------------------------------------------------------------------
def synth_block(variant, n_sc=40, seed=0):
    """12 искусственных голов, n_sc «сцен»: карты на картинке и на её отражении. [2·n_sc, 12, 196, 196]."""
    rng = np.random.default_rng(seed)
    n = G * G
    r, c = H.R_, H.C_
    dr, dc = r[None, :] - r[:, None], c[None, :] - c[:, None]          # [q, k]
    same_row, same_col = dr == 0, dc == 0
    bg = np.full((n, n), 0.2 / n)

    def row_side(sgn, weak):
        w = np.exp(-np.abs(dc) / 3.0) * same_row
        return bg + np.where(sgn * dc > 0, w, weak * w)

    row_sym = bg + same_row / G
    col_stripe = bg + same_col / G
    col_local = bg + np.exp(-np.abs(dr) / 2.0) * same_col
    local3 = bg + ((np.abs(dr) <= 1) & (np.abs(dc) <= 1))
    step = (lambda s: bg + 0.7 * (same_row & (dc == s))) if variant != "слабая" else (lambda s: bg + 0.7 * (same_col & (dr == s)))
    weak = 0.15 if variant != "слабая" else 0.6
    fixed = [row_side(-1, weak), row_side(+1, weak) if variant != "две левые" else row_side(-1, weak), row_sym, row_sym, col_stripe, local3,
             np.full((n, n), 1.0 / n), step(-1), step(+1), col_local]
    A = np.zeros((2 * n_sc, NH, n, n))
    for s in range(n_sc):
        c0, r0 = int(rng.integers(1, 5)), int(rng.integers(2, 11))
        for half, cc in ((0, c0), (1, G - 2 - c0)):                      # объект 2 × 2: на отражении — зеркально
            obj = np.zeros(n)
            obj[[(r0 + a) * G + cc + b_ for a in (0, 1) for b_ in (0, 1)]] = 1.0
            content = bg + obj[None, :] / 4
            maps = fixed + [content, content]
            for h_, m_ in enumerate(maps[:NH]):
                noise = np.exp(0.2 * rng.standard_normal((n, n)))
                A[s + half * n_sc, h_] = m_ * noise
            A[s + half * n_sc, 11] = np.exp(rng.standard_normal((n, n)))  # голова-шум
    return torch.from_numpy(A / A.sum(-1, keepdims=True)).float()


def check_rules():
    names = ["полоса влево", "полоса вправо", "полоса симм.", "полоса симм.", "столбец", "локальная 3×3", "равномерная", "сосед слева",
             "сосед справа", "столбец лок.", "на объект", "шум"]
    ok = True
    # «слабая»: стороны полосы различаются слабо (1 : 0,6), других зеркальных ←→ пар в блоке нет (головы 7, 8 — соседи сверху / снизу).
    # Условие (iii) консервативно: если в блоках есть 3 пары с большим D, слабая пара не пройдёт (см. ограничения в регистрации).
    for variant, must_pass, must_fail in (("обычная", [(0, 1), (7, 8)], [(2, 3), (4, 9), (5, 6), (2, 10), (10, 11), (3, 11)]),
                                          ("слабая", [(0, 1)], [(2, 3), (7, 8)]),
                                          ("две левые", [], [(0, 1), (2, 3)])):
        A = synth_block(variant)
        n_sc = A.shape[0] // 2
        dx, dy, Sd, Sm = block_stats(A)
        R = {k: np.zeros((n_sc, NB, NH), np.float32) for k in ("dx", "dy")}
        R.update({k: np.zeros((n_sc, NB, NH, NH), np.float32) for k in ("Sd", "Sm")})
        R["dx"][:, 0], R["dy"][:, 0], R["Sd"][:, 0], R["Sm"][:, 0] = dx.numpy(), dy.numpy(), Sd.numpy(), Sm.numpy()
        st = summarize(R)
        res = {p: g1_rule(st, [(1, p[0]), (1, p[1])], (1,)) for p in must_pass + must_fail}
        good = all(res[p]["ok"] for p in must_pass) and not any(res[p]["ok"] for p in must_fail)
        ok &= good
        print(f"  искусственный блок «{variant}»: dx: " + ", ".join(f"{nm} {st['dx'][0, h_]:+.2f}" for h_, nm in enumerate(names)))
        for p in must_pass + must_fail:
            q = res[p]
            print(f"    пара {names[p[0]]} / {names[p[1]]}: S_зерк {st['Sm'][0][p]:+.3f}, S_прям {st['Sd'][0][p]:+.3f}, D {st['D'][0][p]:+.3f} "
                  f"[{st['D_lo'][0][p]:+.3f}; {st['D_hi'][0][p]:+.3f}]; (i) {q['c1']}, (ii) {q['c2']}, (iii) {q['c3']} (выше: {q['n_ge']}) -> "
                  f"{'проходит' if q['ok'] else 'нет'}; ожидалось {'проходит' if p in must_pass else 'нет'}")
        print(f"    {mark(good)}")
    cases = [((2.0, 1.0), (3.3, 2.3), 0.6, [(0, 1.8), (1, 2.9)], ("сохранилась", "специфично")),
             ((2.0, 1.0), (3.3, 2.3), 2.5, [(0, 1.8), (1, 2.9)], ("вместе не гасится", "специфично")),
             ((2.0, -0.1), (3.3, 2.3), 0.6, [(0, 1.8)], ("по одной нет", "не оценивается")),
             ((2.0, 1.0), (3.3, 2.3), 0.6, [(0, 0.3), (1, 2.9)], ("сохранилась", "не показано"))]
    g2ok = all(exp[0] in g2_verdict(*c[:4])[0] and exp[1] in g2_verdict(*c[:4])[1] for *c, exp in [(*x[:4], x[4]) for x in cases])
    ok &= g2ok
    print(f"  правило G2 на искусственных числах (4 случая): {mark(g2ok)}")
    print(f"Проверка 3 (правила на искусственных данных с известным ответом): {mark(ok)}")
    return ok


# ------------------------------------------------------------------
# часть «maps»: G1 и G3
# ------------------------------------------------------------------
def rel_check1(model, x2):
    with torch.no_grad():
        a, b = H.forward(model, x2), run_layers(model, x2)
    return max(float((p - q).abs().max()) / max(float(q.abs().max()), 1e-12) for p, q in zip(a, b))


def run_maps(args):
    device = pick_device()
    import timm
    test = bool(args.limit or args.test_random_main)
    tagx = f"_test{args.limit}" if args.limit else ""
    print(f"Опыт G, часть «карты внимания» (G1, G3)   timm {timm.__version__}, torch {torch.__version__}")
    if test:
        print("⚠️ ПРОВЕРОЧНЫЙ прогон (облако): числа не для статьи.")
    ok = {k: True for k in ("1", "3", "4", "7", "9")}
    ok["3"] = check_rules()
    set_data(DATA_OLD)
    meta, infos, lab, folds, sym, sym_arrs = H.scenes(args)
    print(f"data_v2/meta.jsonl md5 {hashlib.md5(open(f'{DATA_OLD}/meta.jsonl', 'rb').read()).hexdigest()}; сцен {len(meta)}, симметричных {len(sym)}")
    os.makedirs(args.out, exist_ok=True)
    hrows, prows, rules, S = [], [], [], {}
    for key in args.nets.split(","):
        name, timm_name = H.NETS[key]
        t0 = time.time()
        thr, dx_thr, fps = -np.inf, 0.0, {}
        for seed in H.RAND_SEEDS:
            model, w = mn.load(timm_name, seed)
            fps[seed] = fingerprint(model)
            _, _, n_pre, nb, mean, std = mn.setup(model, H.P)
            model.to(device)
            R, wr = maps_pass(model, sym_arrs, device, mean, std, n_pre)
            ok["1"] &= wr < 1e-4
            Dr = R["Sm"].mean(0, dtype=np.float64) - R["Sd"].mean(0, dtype=np.float64)
            dvals = np.array([Dr[b - 1, i, j] for b, i, j in within_pairs(ALL_BLOCKS)])
            thr = max(thr, float(dvals.max()))
            dxr = R["dx"].mean(0)
            dx_thr = max(dx_thr, float(np.abs(dxr).max()))
            print(f"  случайная seed {seed}: отпечаток {fps[seed]:.4f}; D по 792 парам: мин {dvals.min():+.4f}, макс {dvals.max():+.4f}; "
                  f"|dx| макс {np.abs(dxr).max():.3f}")
            for b in range(NB):
                for h in range(NH):
                    hrows.append(dict(net=name, seed=seed, block=b + 1, head=h, dx=dxr[b, h], dx_lo="", dx_hi="", dy=R["dy"].mean(0)[b, h],
                                      dy_lo="", dy_hi="", self_mir="", self_out_mir=""))
            del model
        model, weights = mn.load(timm_name, 99 if args.test_random_main else None)
        fp = fingerprint(model)
        ok["7"] &= all(abs(fp - f) > 1e-3 for f in fps.values())
        print(f"\n### {name}: {weights}   отпечаток весов {fp:.4f}")
        ok4, _, n_pre, nb, mean, std = mn.setup(model, H.P)
        ok["4"] &= ok4 and all(type(b.attn.norm).__name__ == "Identity" for b in model.blocks)
        model.to(device)
        x2 = H.to_x([H.load_u8(f"{DATA_OLD}/images/{meta[0]['id']}_{v}.png") for v in ("S", "Sp")], mean, std).to(device)
        d1 = rel_check1(model, x2)
        ok["1"] &= d1 < 1e-4
        cache = os.path.join(args.out, f"pair_maps_{key}{tagx}.npz")
        if os.path.exists(cache) and not args.test_random_main:
            R = dict(np.load(cache))
            print(f"  карты взяты из {cache}")
        else:
            R, wr = maps_pass(model, sym_arrs, device, mean, std, n_pre, with_out=True)
            ok["1"] &= wr < 1e-4
            if not args.test_random_main:
                np.savez(cache, **R)
        print(f"  Проверка 1: прогон с явным вниманием = run_layers, относительная разница {d1:.2e} {mark(d1 < 1e-4)}")
        fm = os.path.join(args.out, f"m_{key}.npz")
        if not test and os.path.exists(fm):
            m_old, m_new = np.load(fm)["M"].mean(0), R["m"].mean(0)
            d9 = float(np.abs(m_old - m_new).max() / np.abs(m_old).max())
            ok["9"] &= d9 < 1e-3
            print(f"  Проверка 9: мера m опыта F пересчитана, относительная разница с кэшем {d9:.2e} {mark(d9 < 1e-3)}")
        elif not test:
            ok["9"] = False
            print(f"  Проверка 9: нет кэша {fm} ⚠️")
        st = summarize(R)
        st["dx_thr"] = dx_thr
        S[key] = st
        found = g3_pairs(st, thr)
        partners = {(b, h) for b, i, j in found for h in (i, j)}
        v3 = "зеркальная пара найдена" if found else "не найдена"
        print(f"\n  G3 ({name}): пороги по случайным сетям: T (наибольшее D) = {thr:+.4f}, наибольшее |dx| = {dx_thr:.4f}; пар с противоположными dx и D > T: {len(found)} из 792 -> «{v3}»")
        print(f"      справка: голов с зеркальным партнёром {len(partners)} из 144 ({len(partners) / 144 * 100:.0f}%); по блокам: " +
              ", ".join(f"L{b:02d} {sum(1 for bb, _, _ in found if bb == b)}" for b in ALL_BLOCKS))
        for b, i, j in sorted(found, key=lambda p: -st["D"][p[0] - 1, p[1], p[2]])[:8]:
            print(f"      {b}.{i} / {b}.{j}: dx {st['dx'][b - 1, i]:+.3f} / {st['dx'][b - 1, j]:+.3f}; S_зерк {st['Sm'][b - 1, i, j]:+.3f}, "
                  f"S_прям {st['Sd'][b - 1, i, j]:+.3f}, D {st['D'][b - 1, i, j]:+.3f} [{st['D_lo'][b - 1, i, j]:+.3f}; {st['D_hi'][b - 1, i, j]:+.3f}]; "
                  f"выходы: зерк {st['Om'][b - 1, i, j]:+.3f}, прям {st['Od'][b - 1, i, j]:+.3f}")
        rules.append(dict(rule="G3", net=name, verdict=v3, value=len(found), extra=f"T={thr:.5f}; |dx| случайных {dx_thr:.5f}; голов с партнёром {len(partners)}"))
        for b in range(NB):
            for h in range(NH):
                hrows.append(dict(net=name, seed="" if not args.test_random_main else 99, block=b + 1, head=h, dx=st["dx"][b, h],
                                  dx_lo=st["dx_lo"][b, h], dx_hi=st["dx_hi"][b, h], dy=st["dy"][b, h], dy_lo=st["dy_lo"][b, h],
                                  dy_hi=st["dy_hi"][b, h], self_mir=st["Sm"][b, h, h], self_out_mir=st["Om"][b, h, h]))
        fset = set(found)
        for b, i, j in within_pairs(ALL_BLOCKS):
            prows.append(dict(net=name, block=b, a=i, b=j, s_dir=st["Sd"][b - 1, i, j], s_mir=st["Sm"][b - 1, i, j], d=st["D"][b - 1, i, j],
                              d_lo=st["D_lo"][b - 1, i, j], d_hi=st["D_hi"][b - 1, i, j], o_dir=st["Od"][b - 1, i, j], o_mir=st["Om"][b - 1, i, j],
                              opposite=int(opposite(st, b, i, j)), g3=int((b, i, j) in fset), thr=thr))
        if key == "dino":
            (b, i), (_, j) = PAIR
            q = g1_rule(st, PAIR, G1_BLOCKS)
            v1 = "зеркальная пара" if q["ok"] else "не показано"
            wp = within_pairs(G1_BLOCKS)
            sm_rank = 1 + sum(st["Sm"][bb - 1, x, y] > st["Sm"][b - 1, i, j] for bb, x, y in wp)
            od = st["Om"][b - 1, i, j] - st["Od"][b - 1, i, j]
            od_rank = 1 + sum(st["Om"][bb - 1, x, y] - st["Od"][bb - 1, x, y] > od for bb, x, y in wp)
            print(f"\n  G1 (пара {b}.{i} / {b}.{j}):")
            for h in (i, j):
                print(f"      {b}.{h}: dx {st['dx'][b - 1, h]:+.4f} [{st['dx_lo'][b - 1, h]:+.4f}; {st['dx_hi'][b - 1, h]:+.4f}] патча;  "
                      f"dy {st['dy'][b - 1, h]:+.4f} [{st['dy_lo'][b - 1, h]:+.4f}; {st['dy_hi'][b - 1, h]:+.4f}] (справка);  "
                      f"«сама с собой»: карты S_зерк {st['Sm'][b - 1, h, h]:+.3f}, выходы {st['Om'][b - 1, h, h]:+.3f} (справка)")
            print(f"      (i)  dx противоположны, интервалы без 0, |dx| обеих > {dx_thr:.4f} (наибольшее у случайных сетей): {'да' if q['c1'] else 'нет'}")
            print(f"      (ii) S_зерк {st['Sm'][b - 1, i, j]:+.4f}, S_прям {st['Sd'][b - 1, i, j]:+.4f}, D = {st['D'][b - 1, i, j]:+.4f} "
                  f"[{st['D_lo'][b - 1, i, j]:+.4f}; {st['D_hi'][b - 1, i, j]:+.4f}]: нижняя граница > 0 — {'да' if q['c2'] else 'нет'}")
            print(f"      (iii) пар с D не меньше среди остальных {q['n_others']}: {q['n_ge']} (нужно ≤ 2) — {'да' if q['c3'] else 'нет'}")
            print(f"      -> «{v1}»")
            print(f"      справка: место по S_зерк среди 330 пар — {sm_rank}; выходы голов: зерк {st['Om'][b - 1, i, j]:+.4f}, "
                  f"прям {st['Od'][b - 1, i, j]:+.4f}, разность {od:+.4f} (место {od_rank} из 330)")
            n12 = sum(1 for bb, x, y in wp if opposite(st, bb, x, y) and st["D_lo"][bb - 1, x, y] > 0)
            print(f"      справка: пар блоков 2–6, для которых выполнены (i) и (ii): {n12} из 330")
            top = sorted(wp, key=lambda p: -st["D"][p[0] - 1, p[1], p[2]])[:6]
            print("      справка: наибольшие D в блоках 2–6: " + "; ".join(
                f"{bb}.{x} / {bb}.{y} D {st['D'][bb - 1, x, y]:+.3f} (dx {st['dx'][bb - 1, x]:+.2f} / {st['dx'][bb - 1, y]:+.2f})" for bb, x, y in top))
            rules.append(dict(rule="G1", net=name, verdict=v1, value=st["D"][b - 1, i, j],
                              extra=f"(i) {q['c1']} (ii) {q['c2']} (iii) {q['c3']}, выше {q['n_ge']}; dx {st['dx'][b - 1, i]:+.4f} / {st['dx'][b - 1, j]:+.4f}"))
        print(f"  ({time.time() - t0:.0f} с)", flush=True)
        del model
    sfx = "" if not test else "_test"
    H.write_rows(OUT_HEADS.replace(".csv", sfx + ".csv"), hrows, list(hrows[0].keys()))
    H.write_rows(OUT_PAIRS.replace(".csv", sfx + ".csv"), prows, list(prows[0].keys()))
    H.write_rows(f"heads_pair_rules_maps{sfx}.csv", rules, list(rules[0].keys()))
    if not test:
        hit = lambda b: "ПОПАЛ ✅" if b else "НЕ ПОПАЛ ❌"
        by = {(r["rule"], r["net"]): r["verdict"] for r in rules}
        print("\nСТАВКИ ЕРНАЗАРА (реестр 199, 15.11 21:02, до кода):")
        if ("G1", H.NETS["dino"][0]) in by:
            print(f"  199a G1: пара 4.4 / 4.5 зеркальная -> «{by[('G1', H.NETS['dino'][0])]}»  {hit(by[('G1', H.NETS['dino'][0])] == 'зеркальная пара')}")
        if ("G3", H.NETS["deit3"][0]) in by:
            print(f"  199c G3: у DeiT III пара найдётся -> «{by[('G3', H.NETS['deit3'][0])]}»  {hit(by[('G3', H.NETS['deit3'][0])] == 'зеркальная пара найдена')}")
        if ("G3", H.NETS["clip"][0]) in by:
            print(f"  199d G3: у CLIP пара не найдётся -> «{by[('G3', H.NETS['clip'][0])]}»  {hit(by[('G3', H.NETS['clip'][0])] == 'не найдена')}")
    print()
    for k, t in (("1", "прогон = run_layers; строки внимания в сумме 1"), ("3", "правила на искусственных данных"), ("4", "сетка, токены, блоки"),
                 ("7", "случайные сети отличаются от обученной"), ("9", "мера m опыта F воспроизведена")):
        if k == "9" and test:
            continue
        print(f"Проверка {k} ({t}): {mark(ok[k])}")


# ------------------------------------------------------------------
# часть «ablate»: G2
# ------------------------------------------------------------------
class _Stop(Exception):
    pass


def donors_for(n, seed):
    """Донор для каждой сцены: случайный цикл по всем сценам (неподвижных точек нет)."""
    order = np.random.default_rng(seed).permutation(n)
    d = np.empty(n, dtype=np.int64)
    d[order] = np.roll(order, -1)
    return d


@torch.no_grad()
def head_outputs(model, x, heads):
    """Обычный прогон до последнего нужного блока; {(блок, голова): выход головы [B, N, hd]}."""
    need = H.as_off(heads)
    last, got = max(need), {}

    def cb(b, att, v, a):
        if b in need:
            o = att @ v
            for hh in need[b]:
                got[(b, hh)] = o[:, hh].clone()
        if b == last:
            raise _Stop

    try:
        H.forward(model, x, attn_cb=cb)
    except _Stop:
        pass
    return got


@torch.no_grad()
def feats_resample(model, recs, infos, device, mean, std, n_pre, nb, spec):
    """spec: [((блок, голова), доноры [n])]. Признаки F12 [сцены, 2 (S, Sp), 2 (cls, дальний фон), D], как H.feats_pass."""
    F12 = np.zeros((len(recs), 2, 2, model.embed_dim), np.float32)
    load = lambda si, vi: H.load_u8(f"{H.DATA}/images/{recs[si]['id']}_{('S', 'Sp')[vi]}.png")
    groups = {}
    for bh, d in spec:
        groups.setdefault(d.tobytes(), (d, []))[1].append(bh)
    off = H.as_off([bh for bh, _ in spec])
    jobs = [(si, vi) for si in range(len(recs)) for vi in range(2)]
    for s in range(0, len(jobs), H.BATCH):
        part = jobs[s:s + H.BATCH]
        means = {}
        for d, hs in groups.values():
            xd = H.to_x([load(int(d[si]), vi) for si, vi in part], mean, std).to(device)
            for (b, hh), o in head_outputs(model, xd, hs).items():
                means.setdefault(b, {})[hh] = o
        x = H.to_x([load(si, vi) for si, vi in part], mean, std).to(device)
        h12 = H.forward(model, x, off=off, means=means)[nb].float().cpu().numpy()
        for i, (si, vi) in enumerate(part):
            far = [int(q) + n_pre for q in infos[si][0]["o"]["far_bg"]]
            F12[si, vi, 0] = h12[i, 0]
            F12[si, vi, 1] = h12[i, far].mean(0)
    return F12.astype(np.float16).astype(np.float32)


def hs(bh):
    return f"{bh[0]}.{bh[1]}"


def run_ablate(args):
    device = pick_device()
    import timm
    test = bool(args.limit or args.test_random_main)
    tagx = f"_test{args.limit}" if args.limit else ""
    name, timm_name = H.NETS["dino"]
    print(f"Опыт G, часть «выключение» (G2), сеть {name}   timm {timm.__version__}, torch {torch.__version__}")
    if test:
        print("⚠️ ПРОВЕРОЧНЫЙ прогон (облако): числа не для статьи.")
    ok = {k: True for k in ("1", "2", "3", "4", "5", "6", "8")}
    ok["warn"] = 0
    ok["3"] = check_rules()
    # проверка 8: новые сцены не совпадают со старыми
    sig = lambda r: json.dumps([[o["shape"], o["color"], o["center"]] for o in r["objects"]])
    new_meta = [json.loads(line) for line in open(f"{DATA_NEW}/meta.jsonl")]
    if os.path.exists(f"{DATA_OLD}/meta.jsonl"):
        old_sig = {sig(json.loads(line)) for line in open(f"{DATA_OLD}/meta.jsonl")}
        same = sum(sig(r) in old_sig for r in new_meta)
        ok["8"] &= same == 0
        print(f"Проверка 8: сцен data_g, совпавших со сценой data_v2: {same} из {len(new_meta)}  {mark(same == 0)}")
    else:
        ok["8"] = False
        print("Проверка 8: нет data_v2/meta.jsonl ⚠️")
    print(f"data_g/meta.jsonl md5 {hashlib.md5(open(f'{DATA_NEW}/meta.jsonl', 'rb').read()).hexdigest()}")
    set_data(DATA_NEW)
    meta, infos, lab, folds, sym, sym_arrs = H.scenes(args)
    n = len(meta)
    nfar = min(len(infos[i][0]["o"]["far_bg"]) for i in range(n))
    ok["5"] &= nfar >= 1
    os.makedirs(args.out, exist_ok=True)
    model, weights = mn.load(timm_name, 99 if args.test_random_main else None)
    print(f"### {name}: {weights}   отпечаток весов {fingerprint(model):.4f};  сцен {n}; патчей дальнего фона в сцене не меньше {nfar}")
    ok4, _, n_pre, nb, mean, std = mn.setup(model, H.P)
    ok["4"] &= ok4 and all(type(b.attn.norm).__name__ == "Identity" for b in model.blocks)
    model.to(device)
    x2 = H.to_x([H.load_u8(f"{DATA_NEW}/images/{meta[0]['id']}_{v}.png") for v in ("S", "Sp")], mean, std).to(device)
    d1 = rel_check1(model, x2)
    ok["1"] &= d1 < 1e-4
    with torch.no_grad():
        ref = H.forward(model, x2)[nb]
        own = head_outputs(model, x2, PAIR + CTRL)
        mm = {}
        for (b, hh), o in own.items():
            mm.setdefault(b, {})[hh] = o
        alt = H.forward(model, x2, off=H.as_off(PAIR + CTRL), means=mm)[nb]
        swp = H.forward(model, x2, off=H.as_off(PAIR), means={b: {hh: o.flip(0) for hh, o in v.items()} for b, v in mm.items()})[nb]
    d2 = float((ref - alt).abs().max() / ref.abs().max())
    d2b = float((ref - swp).abs().max() / ref.abs().max())
    ok["2"] &= d2 < 1e-4 and d2b > 1e-6
    print(f"  Проверка 1: прогон с явным вниманием = run_layers, относительная разница {d1:.2e} {mark(d1 < 1e-4)};  "
          f"Проверка 2: донор «сама картинка» — разница {d2:.2e}, чужой донор — {d2b:.2e} (должна быть > 0) {mark(d2 < 1e-4 and d2b > 1e-6)}")

    # --- обычный прогон на новых сценах (+ средние голов для справки «выключение средним»)
    cache_c = os.path.join(args.out, f"pair_clean{tagx}.npz")
    if os.path.exists(cache_c) and not args.test_random_main:
        z = np.load(cache_c)
        F12, F00, T = z["F12"], z["F00"], z["T"]
        means = {b: torch.from_numpy(z["means"][b - 1]) for b in range(1, NB + 1)}
        print(f"  обычный прогон взят из {cache_c}")
    else:
        acc = H.new_acc(model, n_pre + G * G)
        F12, F00, T = H.feats_pass(model, meta, infos, device, mean, std, n_pre, nb, acc=acc, with_l00=True)
        means = {b: (acc[b] / (2 * n)).float() for b in acc}
        if not args.test_random_main:
            np.savez(cache_c, F12=F12, F00=F00, T=T, means=np.stack([means[b].numpy() for b in range(1, NB + 1)]))
    clean = H.probes(F12, lab, folds, ok)
    l00 = H.probes(F00, lab, folds, ok)
    ok["5"] &= all(np.allclose(v, 0.5) for v in l00.values())
    for q in ("left", "upper"):
        tq = np.array([infos[i][1][q][0] for i in range(n)])
        c_, _ = md.probe_plain(T[np.arange(n), :, tq], lab[q][:, :, 0], lab[q][:, :, 1], folds)
        ok["5"] &= test or np.nanmean(c_) > 0.9
    cl, cu = np.nanmean(clean[(1, "left")]) * 100, np.nanmean(clean[(1, "upper")]) * 100
    ok["5"] &= test or (abs(cl - V2_CLEAN[0]) <= 5 and abs(cu - V2_CLEAN[1]) <= 5)
    print(f"  обычная сеть на новых сценах: «левый» {cl:.2f}% (data_v2: {V2_CLEAN[0]:.2f}), «верхний» {cu:.2f}% (data_v2: {V2_CLEAN[1]:.2f}); "
          f"CLS: «левый» {np.nanmean(clean[(0, 'left')]) * 100:.2f}%, «верхний» {np.nanmean(clean[(0, 'upper')]) * 100:.2f}%")

    # --- доноры
    dA = {p: donors_for(n, p) for p in PERMS}
    dB = {p: donors_for(n, 100 + p) for p in PERMS}
    ok["6"] &= all((d != np.arange(n)).all() for d in list(dA.values()) + list(dB.values()))
    ok["6"] &= len({d.tobytes() for d in list(dA.values()) + list(dB.values())}) == 2 * len(PERMS)
    a44, a45 = PAIR
    conds = [(f"одна {hs(a44)}", [(a44, "A")]), (f"одна {hs(a45)}", [(a45, "A")]),
             ("пара, один донор", [(a44, "A"), (a45, "A")]), ("пара, разные доноры", [(a44, "A"), (a45, "B")])]
    conds += [(f"контроль {hs(c)}", [(c, "A")]) for c in CTRL]
    conds += [(f"{hs(a)} + контроль {hs(c)}", [(a, "A"), (c, "A")]) for a in PAIR for c in CTRL]
    rows, t0, pscene = [], time.time(), {}

    def add(cname, kind, heads, Fs):
        accs = [H.probes(F, lab, folds, ok) for F in Fs]
        row = dict(cond=cname, kind=kind, heads=" ".join(hs(h) for h in heads), n_perms=len(Fs))
        for gi, g in ((1, "far"), (0, "cls")):
            for q, qn in (("left", "dleft"), ("upper", "dupper")):
                d = np.mean([a[(gi, q)] for a in accs], axis=0) - clean[(gi, q)]
                if (gi, q) == (1, "left"):
                    pscene[(kind, cname)] = d
                pt, lo, hi, _ = md.boot(d, np.random.default_rng(0))
                row.update({f"{g}_{qn}": pt, f"{g}_{qn}_lo": lo, f"{g}_{qn}_hi": hi})
            row[f"{g}_upper_acc"] = float(np.mean([np.nanmean(a[(gi, "upper")]) for a in accs]))
        per = ", ".join(f"{(np.nanmean(a[(1, 'left')]) - np.nanmean(clean[(1, 'left')])) * 100:+.2f}" for a in accs)
        rows.append(row)
        print(f"  {kind:8s} {cname:26s}: Δлевый {row['far_dleft'] * 100:+.2f} [{row['far_dleft_lo'] * 100:+.2f}; {row['far_dleft_hi'] * 100:+.2f}] "
              f"(по перестановкам: {per}), Δверхний {row['far_dupper'] * 100:+.2f} [{row['far_dupper_lo'] * 100:+.2f}; {row['far_dupper_hi'] * 100:+.2f}];  "
              f"CLS Δлевый {row['cls_dleft'] * 100:+.2f}, CLS Δверхний {row['cls_dupper'] * 100:+.2f}   ({time.time() - t0:.0f} с)", flush=True)
        return row

    print(f"\n  G2: resample-выключение, {len(PERMS)} перестановки доноров")
    for cname, spec in conds:
        Fs = []
        for p in PERMS:
            slug = "_".join(f"{hs(bh)}{k}" for bh, k in spec)
            cache = os.path.join(args.out, f"pair_rs_{slug}_p{p}{tagx}.npy")
            if os.path.exists(cache) and not args.test_random_main:
                F = np.load(cache).astype(np.float32)
            else:
                F = feats_resample(model, meta, infos, device, mean, std, n_pre, nb, [(bh, (dA if k == "A" else dB)[p]) for bh, k in spec])
                if not args.test_random_main:
                    np.save(cache, F.astype(np.float16))
            Fs.append(F)
        add(cname, "resample", [bh for bh, _ in spec], Fs)
    print("\n  справка: выключение средним (как в опыте F; средние — по 4000 картинкам новых сцен)")
    for cname, heads in ((f"одна {hs(a44)}", [a44]), (f"одна {hs(a45)}", [a45]), ("пара", list(PAIR))):
        slug = "_".join(hs(bh) for bh in heads)
        cache = os.path.join(args.out, f"pair_mean_{slug}{tagx}.npy")
        if os.path.exists(cache) and not args.test_random_main:
            F = np.load(cache).astype(np.float32)
        else:
            F = H.feats_pass(model, meta, infos, device, mean, std, n_pre, nb, off=H.as_off(heads), means=means)[0]
            if not args.test_random_main:
                np.save(cache, F.astype(np.float16))
        add(cname, "среднее", heads, [F])

    by = {(r["kind"], r["cond"]): r for r in rows}
    R = lambda c: by[("resample", c)]
    s44, s45, pr = R(f"одна {hs(a44)}"), R(f"одна {hs(a45)}"), R("пара, один донор")
    ctrl_pairs = [(k, R(f"{hs(a)} + контроль {hs(c)}")["far_dleft"]) for k, a in enumerate(PAIR) for c in CTRL]
    v2, spec_v, c_single, c_pair = g2_verdict((s44["far_dleft"], s44["far_dleft_lo"]), (s45["far_dleft"], s45["far_dleft_lo"]), pr["far_dleft"], ctrl_pairs)
    destroyed = any(r["far_upper_acc"] < 0.60 for r in rows)
    half = 0.5 * min(s44["far_dleft"], s45["far_dleft"]) * 100
    print(f"\n  G2: (а) нижние границы Δлевый по одной: {hs(a44)} {s44['far_dleft_lo'] * 100:+.2f}, {hs(a45)} {s45['far_dleft_lo'] * 100:+.2f} — "
          f"{'обе > 0' if c_single else 'не обе > 0'};  (б) вместе {pr['far_dleft'] * 100:+.2f} против половины меньшего {half:+.2f} — "
          f"{'меньше' if c_pair else 'не меньше'}")
    print(f"      -> «{v2}»{'  ⚠️ пометка «разрушение»: «верхний» < 60%' if destroyed else ''}")
    print("      специфичность: " + "; ".join(f"{hs(PAIR[k])} + контроль: {d * 100:+.2f} (нужно ≥ {0.5 * (s44, s45)[k]['far_dleft'] * 100:+.2f})"
                                               for k, d in ctrl_pairs) + f" -> «{spec_v}»")
    for kind, pname in (("resample", "пара, один донор"), ("resample", "пара, разные доноры"), ("среднее", "пара")):
        na = pscene[(kind, f"одна {hs(a44)}")] + pscene[(kind, f"одна {hs(a45)}")] - pscene[(kind, pname)]
        pt, lo, hi, _ = md.boot(na, np.random.default_rng(0))
        print(f"      справка: неаддитивность Δлевый({hs(a44)}) + Δлевый({hs(a45)}) − Δлевый({pname}), {kind}: {pt * 100:+.2f} [{lo * 100:+.2f}; {hi * 100:+.2f}]")
    sfx = "" if not test else "_test"
    with open(OUT_G2.replace(".csv", sfx + ".csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(f"heads_pair_rules_g2{sfx}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["rule", "net", "verdict", "value", "extra"])
        w.writeheader()
        w.writerow(dict(rule="G2", net=name, verdict=v2, value=pr["far_dleft"], extra=f"по одной {s44['far_dleft']:.5f} / {s45['far_dleft']:.5f}"))
        w.writerow(dict(rule="G2 специфичность", net=name, verdict=spec_v, value="", extra="; ".join(f"{d:.5f}" for _, d in ctrl_pairs)))
    if not test:
        good = v2.startswith("картина сохранилась")
        print("\nСТАВКИ ЕРНАЗАРА (реестр 199, 15.11 21:02, до кода):")
        print(f"  199b G2: картина «по одной +, вместе ≈ 0» сохранится -> «{v2}»  {'ПОПАЛ ✅' if good else 'НЕ ПОПАЛ ❌'}")
    print()
    for k, t in (("1", "прогон = run_layers"), ("2", "resample с донором «сама картинка» = обычный прогон"), ("3", "правила на искусственных данных"),
                 ("4", "сетка, токены, блоки"), ("5", "новые сцены: L00 = 50,0%, контроль «близнец-вопрос», обычная сеть ± 5 п.п."),
                 ("6", "доноры без неподвижных точек, перестановки различны"), ("8", "новые сцены не совпадают со старыми")):
        print(f"Проверка {k} ({t}): {mark(ok[k])}")
    print(f"Проверка 10 (предупреждений «не сошлось»: {ok['warn']}): {mark(ok['warn'] == 0)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("maps", "ablate", "check"), required=True)
    ap.add_argument("--nets", default="dino,deit3,clip")
    ap.add_argument("--out", default=os.path.expanduser("~/binding-vit-data/heads"))
    ap.add_argument("--limit", type=int, default=0, help="только для проверки в облаке")
    ap.add_argument("--test-random-main", action="store_true", help="только для проверки в облаке: вместо обученной — случайная seed 99")
    a = ap.parse_args()
    if a.part == "maps":
        run_maps(a)
    elif a.part == "ablate":
        run_ablate(a)
    else:
        check_rules()

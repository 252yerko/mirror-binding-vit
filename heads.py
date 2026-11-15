"""
heads.py — опыт F: «зеркальные» головы внимания. Добавлено 15.11 (реестр 174–176, ставка Ерназара — до кода).

ВОПРОС:
  Выключение всего внимания в блоках 4–12 возвращает DINO и «левый», и чувствительность к отражению слева направо
  (находка 6). Делает ли это горстка голов, которые из патча (r, c) смотрят в его зеркальный патч (r, G−1−c)?
  Такая голова смешивает патч с его отражением — механизм-кандидат для слепоты к стороне.

СЕТИ (timm 1.0.30, вход 224, патч 16, сетка 14 × 14, 12 блоков × 12 голов = 144 головы):
  DINO B/16 (главная), DeiT III B/16, CLIP-OpenAI B/16 — те же веса, что в опыте B (many_nets.py);
  по 3 случайные сети каждой архитектуры (seed 0, 1, 2) — только для меры m.

МЕРА ГОЛОВЫ m (карты внимания, последний софтмакс, 631 сцена только из круга / квадрата / креста, картинка S):
  для запроса-патча q = (r, c):
    m_h(q) = A[q, (r, G−1−c)] − среднее A[q, k] по остальным патчам строки r (без q и без зеркального);
    m_v(q) = A[q, (G−1−r, c)] − среднее A[q, k] по остальным патчам столбца c (без q и без зеркального);
  m_h — среднее по q со столбцом не 6–7, m_v — по q со строкой не 6–7 (у этих зеркальный патч — сосед);
  m = m_h − m_v; среднее по картинкам, 95% интервал — бутстрэп по сценам (1000, seed 0).
  Вычитание среднего по строке / столбцу убирает головы, которые просто смотрят «вдоль строки»;
  вычитание m_v — то, что одинаково для обеих осей. [Изменения после ставки, до кода, реестр 176.]

ПРАВИЛА (записаны ДО запуска; реестр 174–176):
  F1 (каждая сеть): p99 и p1 — 99-й и 1-й перцентили m у 3 × 144 голов случайных сетей.
      N₊ = число голов с нижней границей m > p99; N₋ = число голов с верхней границей m < p1.
      «Зеркальные (горизонтальные) головы есть», если N₊ ≥ 1 и N₊ > N₋; иначе «не показано».
  F2 (каждая сеть, причинное): топ-12 голов по m (точечные значения, все 144 головы). Выключение средним
      (вклад головы до общей проекции заменяется её средним по 4000 картинкам S и Sp обычной сети, отдельно
      для каждой позиции токена; пробы на L12 обучаются заново — как в находке 6).
      Δлевый, Δверхний — изменение точности обычной пробы (дальний фон, L12) против обычной сети, по сценам.
      Контроль: 30 случайных наборов по 12 голов с тем же числом голов в каждом блоке, что у топ-12,
      без голов топ-12 (seed 0). Если в блоке топ-голов больше 6 (свободных меньше нужного), недостающие головы
      случайного и нижнего наборов берутся из ближайших блоков (±1, затем ±2, …; сначала нижний) — правило до запуска.
      «Зеркальные головы стирают „левый“», если (а) нижняя граница Δлевый(топ-12) > 0, (б) Δлевый(топ-12) больше,
      чем у каждого из 30 случайных наборов (p = 1/31), (в) нижняя граница Δлевый − Δверхний (топ-12) > 0;
      иначе «не показано». Пометка «разрушение», если «верхний» после выключения < 60%.
  F3 (DINO): каждая из 144 голов выключается по одной; 1000 сцен (перестановка seed 0, первые 1000; фолды — те же).
      ρ Спирмена(m, Δлевый); «связь следует», если ρ > 0 и односторонний перестановочный p < 0,05 (10 000, seed 0);
      иначе «не подтверждена». Мощность мала: эффект одной головы мал против шума пробы, поэтому
      «не подтверждена» ≠ «связи нет».
  Справка (не правила): нижние 12 голов (по блокам как у топ-12); 4 и 24 топ-головы (доза); d(←→) и d(↑↓) по
      отдельности и их отношение (среднее по патчам и CLS) при каждом выключении; CLS как второй считыватель;
      сходство позиционного кода патча с кодом зеркального патча (←→ против ↑↓); m на пустой серой картинке
      (внимание только от позиций); m, взвешенное нормой переносимого вектора ‖W_O,h v‖ (OV); m на 1050 аэрофото
      (photos_lists.json); случайные головы; число «сильных» среди N₊ (m не меньше 1/197 — доли равномерного внимания;
      у случайных сетей m ≈ 0, поэтому порог p99 очень низкий и F1 по сути проверяет «m надёжно больше нуля»).
  Ставки Ерназара (реестр 175, 15.11 09:59): 175a N₊ у DINO 5–15; 175b большинство из N₊ голов DINO — в блоках 3–6
      (не засчитывается при N₊ = 0); 175c Δлевый(топ-12) у DINO от +1,0 до +3,0 п.п. (округление 0,1);
      175d F2 у DINO «не показано»; 175e N₊: DINO ≥ DeiT III ≥ CLIP и DINO > CLIP; 175f ρ в F3 от 0,00 до +0,40
      (округление 0,01); 175g F3 «не значимо» (p ≥ 0,05).

ВСТРОЕННЫЕ ПРОВЕРКИ (все должны пройти):
  1. Наш прогон с явным вниманием = run_layers (extract.py), макс. разница < 1e-3; строки внимания в сумме 1.
  2. «Выключение собственным средним»: средние по ОДНОЙ картинке → выключение всех 144 голов на ней = обычный прогон.
  3. Мера на искусственных головах: зеркальная ←→ даёт m > 0,4, зеркальная ↑↓ — m < −0,4,
     «вдоль строки», локальная 3 × 3 и равномерная — |m| < 1e-6.
  4. Сетка, служебные токены, блоки (как в many_nets.py); нормировка внутри внимания — пустая (Identity).
  5. Обычная сеть повторяет опыт B: «левый» / «верхний» (дальний фон, L12) ± 0,5 п.п., отношение ± 0,01.
  6. L00: пробы ровно 50,0%; контроль «близнец-вопрос» > 90%.
  7. Случайные сети отличаются от обученной (отпечаток весов).
  8. 30 случайных наборов различны, совпадают с топ-12 по блокам и не содержат голов топ-12.
  9. Предупреждений «обучение не сошлось» — 0.

ЗАПУСК (из папки binding-vit):
  caffeinate -i python heads.py --net dino 2>&1 | tee heads_dino_log.txt     # ~1,5–2 ч; также --net deit3, --net clip
  caffeinate -i python heads.py --f3 2>&1 | tee heads_f3_log.txt             # ~2 ч (нужен кэш --net dino)
  python heads.py --summary 2>&1 | tee heads_summary_log.txt
  -> heads_m.csv, heads_f2.csv, heads_f3.csv, heads_rules.csv; кэш — ~/binding-vit-data/heads/.
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch

import flip_test as ft
import many_nets as mn
import mirror_decomp as md
import probe_big as pb
from extract import run_layers, fingerprint, pick_device

DATA = "data_v2"
BATCH = 16
G, P, NB, NH = 14, 16, 12, 12
NETS = {"dino": ("DINO B/16", "vit_base_patch16_224.dino"),
        "deit3": ("DeiT III B/16", "deit3_base_patch16_224.fb_in1k"),
        "clip": ("CLIP-OpenAI B/16", "vit_base_patch16_clip_quickgelu_224.openai")}
RAND_SEEDS = (0, 1, 2)
K, DOSES, N_SETS = 12, (4, 24), 30
F3_SCENES = 1000
OUT_M, OUT_F2, OUT_F3 = "heads_m.csv", "heads_f2.csv", "heads_f3.csv"

# ------------------------------------------------------------------
# индексы зеркал и мера m
# ------------------------------------------------------------------
QS = np.arange(G * G)
R_, C_ = QS // G, QS % G
MH = R_ * G + (G - 1 - C_)
MV = (G - 1 - R_) * G + C_
VALID_H = np.where(np.abs(C_ - (G - 1 - C_)) > 1)[0]       # столбцы не 6–7
VALID_V = np.where(np.abs(R_ - (G - 1 - R_)) > 1)[0]       # строки не 6–7


def m_from_attn(A):
    """A: [..., 196, 196] (только патчи; строки — запросы). Возвращает (m, m_h, m_v) формы [...]."""
    A = A.double()
    A5 = A.reshape(*A.shape[:-1], G, G)                        # [..., q, row, col]
    q = torch.as_tensor(QS)
    rows = A5[..., q, torch.as_tensor(R_), :]                   # [..., 196, 14] — строка запроса
    cols = A5.transpose(-1, -2)[..., q, torch.as_tensor(C_), :]  # [..., 196, 14] — столбец запроса
    self_ = A[..., q, q]
    mir_h = A[..., q, torch.as_tensor(MH)]
    mir_v = A[..., q, torch.as_tensor(MV)]
    mh = mir_h - (rows.sum(-1) - self_ - mir_h) / (G - 2)
    mv = mir_v - (cols.sum(-1) - self_ - mir_v) / (G - 2)
    mh_ = mh[..., torch.as_tensor(VALID_H)].mean(-1)
    mv_ = mv[..., torch.as_tensor(VALID_V)].mean(-1)
    return mh_ - mv_, mh_, mv_


def check_measure():
    """Проверка 3: искусственные головы с известным ответом."""
    n = G * G
    A = torch.zeros(6, n, n, dtype=torch.float64)
    A[0] = 0.5 / (n - 1)
    A[0, QS, MH] = 0.5                                         # зеркальная ←→ (на средних столбцах MH = сосед)
    A[0, QS, QS] += 0.0
    A[1] = 0.5 / (n - 1)
    A[1, QS, MV] = 0.5                                         # зеркальная ↑↓
    for i in range(n):                                         # «вдоль строки»
        A[2, i, R_[i] * G:(R_[i] + 1) * G] = 1.0 / G
    for i in range(n):                                         # локальная 3 × 3
        nb = [rr * G + cc for rr in range(R_[i] - 1, R_[i] + 2) for cc in range(C_[i] - 1, C_[i] + 2) if 0 <= rr < G and 0 <= cc < G]
        A[3, i, nb] = 1.0 / len(nb)
    A[4] = 1.0 / n                                             # равномерная
    for i in range(n):                                         # «вдоль столбца»
        A[5, i, C_[i]::G] = 1.0 / G
    # строки зеркальных голов должны суммироваться в 1
    A[0] = A[0] / A[0].sum(-1, keepdim=True)
    A[1] = A[1] / A[1].sum(-1, keepdim=True)
    m, _, _ = m_from_attn(A)
    m = m.numpy()
    ok = m[0] > 0.4 and m[1] < -0.4 and all(abs(m[k]) < 1e-6 for k in (2, 3, 4, 5))
    names = ["зеркальная ←→", "зеркальная ↑↓", "вдоль строки", "локальная 3×3", "равномерная", "вдоль столбца"]
    print("Проверка 3 (мера на искусственных головах): " + "; ".join(f"{a} {b:+.4f}" for a, b in zip(names, m)) +
          f"  {'✅' if ok else '⚠️ проверь весь вывод'}")
    return ok


# ------------------------------------------------------------------
# прогон с явным вниманием и выключением голов
# ------------------------------------------------------------------
@torch.no_grad()
def forward(model, x, off=None, means=None, acc=None, attn_cb=None):
    """Возвращает список 13 скрытых состояний (L00 … L12).
    off: {блок 1..12: [головы]} — выход головы (до общей проекции) заменяется средним means[блок][голова] ([N, hd]);
    acc: {блок: тензор [H, N, hd] float64} — суммы выходов голов (для средних); attn_cb(b, att, v, attn_module)."""
    h = model.norm_pre(model.patch_drop(model._pos_embed(model.patch_embed(x))))
    outs = [h]
    for b, blk in enumerate(model.blocks, start=1):
        a = blk.attn
        y = blk.norm1(h)
        B, N, _ = y.shape
        qkv = a.qkv(y).reshape(B, N, 3, a.num_heads, a.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = a.q_norm(q), a.k_norm(k)
        att = ((q * a.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
        o = att @ v                                            # [B, H, N, hd]
        if attn_cb is not None:
            attn_cb(b, att, v, a)
        if acc is not None:
            acc[b] += o.sum(0).cpu().double()
        if off and b in off:
            o = o.clone()
            for hh in off[b]:
                o[:, hh] = means[b][hh].to(o.device, o.dtype)
        o = o.transpose(1, 2).reshape(B, N, a.attn_dim)
        o = a.proj(a.norm(o))
        h = h + blk.drop_path1(blk.ls1(o))
        h = h + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(h))))
        outs.append(h)
    return outs


def new_acc(model, n_tok):
    a = model.blocks[0].attn
    return {b: torch.zeros(a.num_heads, n_tok, a.head_dim, dtype=torch.float64) for b in range(1, NB + 1)}


# ------------------------------------------------------------------
# картинки и признаки
# ------------------------------------------------------------------
def load_u8(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def to_x(arrs, mean, std, flips=None):
    xs = []
    for i, a in enumerate(arrs):
        x = torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float() / 255.0
        x = (x - mean) / std
        f = None if flips is None else flips[i]
        xs.append(x if f in (None, "o") else (x.flip(2) if f == "h" else x.flip(1)))
    return torch.stack(xs)


@torch.no_grad()
def m_pass(model, arrs, device, mean, std, n_pre, with_ov=False):
    """m по картинкам: [n, 12, 12] (и OV-взвешенное, если with_ov); проверка сумм строк внимания."""
    M = np.zeros((len(arrs), NB, NH))
    MO = np.zeros((len(arrs), NB, NH)) if with_ov else None
    worst = [0.0]
    for s in range(0, len(arrs), BATCH):
        x = to_x(arrs[s:s + BATCH], mean, std).to(device)
        cur = {}

        def cb(b, att, v, a):
            worst[0] = max(worst[0], float((att.sum(-1) - 1).abs().max()))
            Ap = att[:, :, n_pre:, n_pre:].float().cpu()
            cur[(b, "m")] = m_from_attn(Ap)[0].numpy()
            if with_ov:
                W = a.proj.weight.detach().float().cpu().reshape(-1, a.num_heads, a.head_dim)   # [D, H, hd]
                Gm = torch.einsum("dhe,dhf->hef", W, W)                                         # [H, hd, hd]
                vv = v.float().cpu()
                nrm = torch.einsum("bhne,hef,bhnf->bhn", vv, Gm, vv).clamp_min(0).sqrt()        # ‖W_O,h v‖
                Aw = att.float().cpu() * nrm[:, :, None, :]
                Aw = Aw / Aw.sum(-1, keepdim=True)
                cur[(b, "o")] = m_from_attn(Aw[:, :, n_pre:, n_pre:])[0].numpy()

        forward(model, x, attn_cb=cb)
        for b in range(1, NB + 1):
            M[s:s + len(x), b - 1] = cur[(b, "m")]
            if with_ov:
                MO[s:s + len(x), b - 1] = cur[(b, "o")]
    return M, MO, worst[0]


def boot_heads(M, rng_seed=0, n=1000):
    """M: [n_img, 12, 12] -> среднее, низ, верх по головам (бутстрэп по картинкам)."""
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(M), size=(n, len(M)))
    bs = np.stack([M[i].mean(0) for i in idx])
    return M.mean(0), np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)


def pos_symmetry(model):
    """Сходство позиционного кода патча с кодом зеркального патча (минус среднее по строке / столбцу)."""
    pe = model.pos_embed[0].detach().float().cpu()[-G * G:]
    pe = pe / pe.norm(dim=-1, keepdim=True)
    S = (pe @ pe.T)[None]
    m, mh, mv = m_from_attn(S)
    return float(mh), float(mv), float(m)


@torch.no_grad()
def feats_pass(model, recs, infos, device, mean, std, n_pre, nb, off=None, means=None, acc=None, with_l00=False):
    """Признаки по сценам: F[сцены, 2 (S, Sp), 2 группы (cls, дальний фон), D] на L12 (и L00); T — близнецы на L12."""
    D = model.embed_dim
    F12 = np.zeros((len(recs), 2, 2, D), np.float32)
    F00 = np.zeros((len(recs), 2, 2, D), np.float32) if with_l00 else None
    T = np.zeros((len(recs), 2, 2, D), np.float32) if with_l00 else None
    jobs = [(si, vi) for si in range(len(recs)) for vi in range(2)]
    for s in range(0, len(jobs), BATCH):
        part = jobs[s:s + BATCH]
        arrs = [load_u8(f"{DATA}/images/{recs[si]['id']}_{('S', 'Sp')[vi]}.png") for si, vi in part]
        outs = forward(model, to_x(arrs, mean, std).to(device), off=off, means=means, acc=acc)
        hs = {0: outs[0].float().cpu().numpy(), nb: outs[nb].float().cpu().numpy()}
        for i, (si, vi) in enumerate(part):
            g = infos[si][0]["o"]
            far = [int(q) + n_pre for q in g["far_bg"]]
            for L, F in ((nb, F12), (0, F00)):
                if F is None:
                    continue
                F[si, vi, 0] = hs[L][i, 0]
                F[si, vi, 1] = hs[L][i, far].mean(0)
            if T is not None:
                for t in range(2):
                    T[si, vi, t] = hs[nb][i, [int(q) + n_pre for q in g[f"twin{t}"]]].mean(0)
    r16 = lambda a: None if a is None else a.astype(np.float16).astype(np.float32)   # как в опыте B
    return r16(F12), r16(F00), r16(T)


@torch.no_grad()
def mirror_pass(model, arrs, device, mean, std, n_pre, nb, off=None, means=None):
    """d(←→), d(↑↓) и отношение на L12: среднее по патчам и CLS (как flip_test / many_nets)."""
    PM = np.zeros((len(arrs), 3, 2, model.embed_dim), np.float32)
    jobs = [(i, k) for i in range(len(arrs)) for k in range(3)]
    for s in range(0, len(jobs), BATCH):
        part = jobs[s:s + BATCH]
        x = to_x([arrs[i] for i, _ in part], mean, std, flips=["ohv"[k] for _, k in part]).to(device)
        h = forward(model, x, off=off, means=means)[nb].float().cpu().numpy()
        for j, (i, k) in enumerate(part):
            PM[i, k, 0] = h[j, n_pre:].mean(0)
            PM[i, k, 1] = h[j, 0]
    PM = PM.astype(np.float16).astype(np.float32)
    out = {}
    for gi, g in enumerate(("mean", "cls")):
        dh = ft.cos_dist(PM[:, 0, gi], PM[:, 1, gi])
        dv = ft.cos_dist(PM[:, 0, gi], PM[:, 2, gi])
        out[g] = (float(dh.mean()), float(dv.mean())) + tuple(ft.boot_ratio(dh, dv, np.random.default_rng(0)))
    return out


def probes(F, lab, folds, ok, groups=(1, 0)):
    """Точности по сценам для «левый» и «верхний» на группах (1 = дальний фон, 0 = CLS)."""
    res = {}
    for gi in groups:
        for q in ("left", "upper"):
            a, nw = md.probe_plain(F[:, :, gi], lab[q][:, :, 0], lab[q][:, :, 1], folds)
            ok["warn"] += nw
            res[(gi, q)] = a
    return res


# ------------------------------------------------------------------
# наборы голов
# ------------------------------------------------------------------
def fill_blocks(per_block, exclude, choose):
    """Набор голов с тем же числом голов в каждом блоке, без exclude. Если в блоке свободных голов меньше нужного
    (топ-голов в нём больше 6), недостающие берутся из ближайших блоков (±1, затем ±2, …; сначала нижний) —
    правило записано до запуска. choose(список кандидатов (блок, голова), n) -> выбранные."""
    taken, s = set(exclude), []
    for b, n in sorted(per_block.items()):
        pool = [(b, h) for h in range(NH) if (b, h) not in taken]
        got = choose(pool, min(n, len(pool)))
        need, d = n - len(got), 1
        while need > 0 and d < NB:
            for bb in (b - d, b + d):
                if need > 0 and 1 <= bb <= NB:
                    pool2 = [(bb, h) for h in range(NH) if (bb, h) not in taken and (bb, h) not in got]
                    more = choose(pool2, min(need, len(pool2)))
                    got += more
                    need -= len(more)
            d += 1
        s += got
        taken |= set(got)
    return sorted(s)


def head_sets(m_point):
    order = np.argsort(-m_point.ravel(), kind="stable")
    top = {k: [(int(i) // NH + 1, int(i) % NH) for i in order[:k]] for k in (K,) + DOSES}
    per_block = {}
    for b, h in top[K]:
        per_block[b] = per_block.get(b, 0) + 1
    topset = set(top[K])
    lowest = lambda pool, n: sorted(pool, key=lambda bh: (m_point[bh[0] - 1, bh[1]], bh))[:n]
    bottom = fill_blocks(per_block, topset, lowest)
    rng = np.random.default_rng(0)
    rand_pick = lambda pool, n: [pool[i] for i in sorted(rng.choice(len(pool), n, replace=False))] if n > 0 else []
    rand_sets, seen, tries = [], set(), 0
    while len(rand_sets) < N_SETS and tries < 100000:
        tries += 1
        s = fill_blocks(per_block, topset, rand_pick)
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            rand_sets.append(s)
    exact = all(n <= NH - n for n in per_block.values())          # без займа из соседних блоков
    counts = lambda s: sorted({b: sum(1 for bb, _ in s if bb == b) for b, _ in s}.items())
    ok = (len(rand_sets) == N_SETS and all(not (set(s) & topset) and len(s) == K for s in rand_sets)
          and (not exact or all(counts(s) == sorted(per_block.items()) for s in rand_sets)))
    if not exact:
        print("  ⚠️ в каком-то блоке топ-голов больше 6: недостающие случайные / нижние головы взяты из соседних блоков (правило до запуска)")
    return top, bottom, rand_sets, per_block, ok


def as_off(heads):
    off = {}
    for b, h in heads:
        off.setdefault(b, []).append(h)
    return off


def hs_str(heads):
    return " ".join(f"{b}.{h}" for b, h in sorted(heads))


# ------------------------------------------------------------------
# основной прогон одной сети (F1, F2)
# ------------------------------------------------------------------
def scenes(args):
    meta = [json.loads(line) for line in open(f"{DATA}/meta.jsonl")]
    if args.limit:
        meta = meta[:args.limit]
    colors = sorted({ob[k] for r in meta for ob in r["objects"] for k in ("color", "color_Sp")})
    folds = np.random.default_rng(0).permutation(len(meta)) % md.N_FOLDS
    pb.FAR = 3
    infos = [md.scene_info(r, colors, P, G) for r in meta]
    lab = {q: np.array([[infos[i][1][q][1][v] for v in ("S", "Sp")] for i in range(len(meta))]) for q in ("left", "upper")}
    sym = [r for r in meta if all(o["shape"] in ft.SYM_SHAPES for o in r["objects"])]
    sym_arrs = [load_u8(f"{DATA}/images/{r['id']}_S.png") for r in sym]
    return meta, infos, lab, folds, sym, sym_arrs


def write_rows(path, rows, fields):
    """Строки этой сети заменяют её старые строки (повторный запуск не даёт дублей)."""
    nets = {r["net"] for r in rows}
    keep = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["net"] not in nets] if os.path.exists(path) else []
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in keep + list(rows):
            w.writerow(r)


def run_net(args):
    key = args.net
    name, timm_name = NETS[key]
    device = pick_device()
    import timm
    print(f"Опыт F, сеть {name}   timm {timm.__version__}, torch {torch.__version__}")
    if args.limit:
        print("⚠️ ПРОВЕРОЧНЫЙ прогон (облако): числа не для статьи.")
    ok = {k: True for k in ("1", "2", "3", "4", "5", "6", "7", "8")}
    ok["warn"] = 0
    ok["3"] = check_measure()
    meta, infos, lab, folds, sym, sym_arrs = scenes(args)
    os.makedirs(args.out, exist_ok=True)
    tagx = f"_test{args.limit}" if args.limit else ""

    # --- случайные сети: только m
    rand_m, fps = [], {}
    for seed in RAND_SEEDS:
        model, w = mn.load(timm_name, seed)
        fps[seed] = fingerprint(model)
        _, _, n_pre, nb, mean, std = mn.setup(model, P)
        model.to(device)
        M, _, wr = m_pass(model, sym_arrs, device, mean, std, n_pre)
        ok["1"] &= wr < 1e-4
        rand_m.append(M.mean(0))
        ps = pos_symmetry(model)
        print(f"  случайная seed {seed}: отпечаток {fps[seed]:.4f}; m: мин {M.mean(0).min():+.5f}, макс {M.mean(0).max():+.5f}; "
              f"поз. коды ←→ {ps[0]:+.4f}, ↑↓ {ps[1]:+.4f}")
        del model
    rand_all = np.concatenate([r.ravel() for r in rand_m])
    p99, p1 = float(np.percentile(rand_all, 99)), float(np.percentile(rand_all, 1))

    # --- обученная сеть
    model, weights = mn.load(timm_name, 99 if args.test_random_main else None)
    fp = fingerprint(model)
    ok["7"] &= all(abs(fp - f) > 1e-3 for f in fps.values())
    print(f"\n### {name}: {weights}   отпечаток весов {fp:.4f}")
    ok8, _, n_pre, nb, mean, std = mn.setup(model, P)
    ok["4"] &= ok8 and all(type(b.attn.norm).__name__ == "Identity" for b in model.blocks)
    model.to(device)
    # проверка 1: прогон = run_layers
    x2 = to_x([load_u8(f"{DATA}/images/{meta[0]['id']}_{v}.png") for v in ("S", "Sp")], mean, std).to(device)
    with torch.no_grad():
        d1 = max(float((a - b).abs().max()) for a, b in zip(forward(model, x2), run_layers(model, x2)))
    ok["1"] &= d1 < 1e-3
    # проверка 2: выключение собственным средним
    acc1 = new_acc(model, n_pre + G * G)
    forward(model, x2[:1], acc=acc1)
    means1 = {b: acc1[b].float() for b in acc1}
    with torch.no_grad():
        ref = forward(model, x2[:1])[nb]
        alt = forward(model, x2[:1], off={b: list(range(NH)) for b in range(1, NB + 1)}, means=means1)[nb]
    d2 = float((ref - alt).abs().max())
    ok["2"] &= d2 < 1e-4
    print(f"  Проверка 1: прогон с явным вниманием = run_layers, макс. разница {d1:.2e} {'✅' if d1 < 1e-3 else '⚠️'};  "
          f"Проверка 2: выключение собственным средним, разница {d2:.2e} {'✅' if d2 < 1e-4 else '⚠️'}")

    # --- m обученной сети
    cache_m = os.path.join(args.out, f"m_{key}{tagx}.npz")
    if os.path.exists(cache_m):
        z = np.load(cache_m)
        M, MO, Mgray, Maer = z["M"], z["MO"], z["Mgray"], z["Maer"]
        print(f"  m взято из {cache_m}")
    else:
        M, MO, wr = m_pass(model, sym_arrs, device, mean, std, n_pre, with_ov=True)
        ok["1"] &= wr < 1e-4
        blank = [np.full((224, 224, 3), 60, np.uint8)]
        Mgray = m_pass(model, blank, device, mean, std, n_pre)[0][0]
        Maer = np.full((NB, NH), np.nan)
        if not args.no_aerial and os.path.exists(args.lists):
            import photos
            L = json.load(open(args.lists))
            ucm = args.ucm_dir or os.path.join(photos.ROOT, "ucm")
            aer = [photos.prep_photo(os.path.join(ucm, p_)) for p_ in L["aerial"][:args.limit or None]]
            Maer = m_pass(model, aer, device, mean, std, n_pre)[0].mean(0)
        np.savez(cache_m, M=M, MO=MO, Mgray=Mgray, Maer=Maer)
    m_pt, m_lo, m_hi = boot_heads(M)
    plus = [(b + 1, h) for b in range(NB) for h in range(NH) if m_lo[b, h] > p99]
    minus = [(b + 1, h) for b in range(NB) for h in range(NH) if m_hi[b, h] < p1]
    v1 = "зеркальные головы есть" if len(plus) >= 1 and len(plus) > len(minus) else "не показано"
    u = 1.0 / (n_pre + G * G)
    strong = [(b_, h_) for b_, h_ in plus if m_pt[b_ - 1, h_] >= u]
    ps = pos_symmetry(model)
    from scipy.stats import spearmanr
    rho_g = spearmanr(m_pt.ravel(), Mgray.ravel()).statistic
    rho_a = spearmanr(m_pt.ravel(), Maer.ravel()).statistic if np.isfinite(Maer).all() else float("nan")
    rho_o = spearmanr(m_pt.ravel(), MO.mean(0).ravel()).statistic
    top, bottom, rand_sets, per_block, ok8s = head_sets(m_pt)
    ok["8"] &= ok8s
    print(f"\n  F1: перцентили случайных голов (3 × 144): p1 {p1:+.5f}, p99 {p99:+.5f}")
    print(f"      N₊ = {len(plus)} (головы блок.голова: {hs_str(plus) or '—'});  N₋ = {len(minus)} ({hs_str(minus) or '—'})  -> «{v1}»")
    print(f"      справка: из N₊ сильных (m ≥ 1/{n_pre + G * G}, т.е. не меньше доли равномерного внимания): {len(strong)} ({hs_str(strong) or '—'})")
    print(f"      справка: позиционные коды ←→ {ps[0]:+.4f}, ↑↓ {ps[1]:+.4f}, разность {ps[2]:+.4f};  ρ(m, m на серой картинке) = {rho_g:+.3f};"
          f"  ρ(m, m OV) = {rho_o:+.3f};  ρ(m, m на аэрофото) = {rho_a:+.3f}")
    print(f"      топ-12: {hs_str(top[K])}  (по блокам {dict(sorted(per_block.items()))}); m топ-12 от {m_pt.ravel()[np.argsort(-m_pt.ravel())][K - 1]:+.5f}; "
          f"наибольшее m {m_pt.max():+.5f} = {m_pt.max() * (n_pre + G * G):.2f} × равномерного внимания 1/{n_pre + G * G} (справка: величина)")
    print("      m по блокам (макс. по головам): " + ", ".join(f"L{b + 1:02d} {m_pt[b].max():+.4f}" for b in range(NB)))
    mrows = []
    for b in range(NB):
        for h in range(NH):
            mrows.append(dict(net=name, seed="" if not args.test_random_main else 99, block=b + 1, head=h, m=m_pt[b, h], m_lo=m_lo[b, h],
                              m_hi=m_hi[b, h], m_ov=MO.mean(0)[b, h], m_gray=Mgray[b, h], m_aerial=Maer[b, h],
                              plus=int((b + 1, h) in plus), minus=int((b + 1, h) in minus), p99=p99, p1=p1))
    for si, r in zip(RAND_SEEDS, rand_m):
        for b in range(NB):
            for h in range(NH):
                mrows.append(dict(net=name, seed=si, block=b + 1, head=h, m=r[b, h], m_lo="", m_hi="", m_ov="", m_gray="", m_aerial="",
                                  plus="", minus="", p99=p99, p1=p1))

    # --- F2: обычный прогон + средние голов
    cache_c = os.path.join(args.out, f"clean_{key}{tagx}.npz")
    if os.path.exists(cache_c):
        z = np.load(cache_c, allow_pickle=True)
        F12, F00, T = z["F12"], z["F00"], z["T"]
        means = {b: torch.from_numpy(z["means"][b - 1]) for b in range(1, NB + 1)}
        print(f"  обычный прогон взят из {cache_c}")
    else:
        acc = new_acc(model, n_pre + G * G)
        F12, F00, T = feats_pass(model, meta, infos, device, mean, std, n_pre, nb, acc=acc, with_l00=True)
        means = {b: (acc[b] / (2 * len(meta))).float() for b in acc}
        np.savez(cache_c, F12=F12, F00=F00, T=T, means=np.stack([means[b].numpy() for b in range(1, NB + 1)]))
    clean = probes(F12, lab, folds, ok)
    l00 = probes(F00, lab, folds, ok)
    ok["6"] &= all(np.allclose(v, 0.5) for v in l00.values())
    for q in ("left", "upper"):
        tq = np.array([infos[i][1][q][0] for i in range(len(meta))])
        c_, _ = md.probe_plain(T[np.arange(len(meta)), :, tq], lab[q][:, :, 0], lab[q][:, :, 1], folds)
        ok["6"] &= np.nanmean(c_) > 0.9
    mir_clean = mirror_pass(model, sym_arrs, device, mean, std, n_pre, nb)
    old = {r["name"]: r for r in csv.DictReader(open(mn.OUT_CSV, encoding="utf-8")) if r["seed"] == ""}
    cl, cu = clean[(1, "left")].mean(), clean[(1, "upper")].mean()
    if not args.limit and not args.test_random_main:
        o = old[name]
        good = abs(cl - float(o["left"])) * 100 <= 0.5 and abs(cu - float(o["upper"])) * 100 <= 0.5 and abs(mir_clean["mean"][2] - float(o["ratio"])) <= 0.01
        ok["5"] &= good
        print(f"  Проверка 5: «левый» {cl * 100:.2f} / в опыте B {float(o['left']) * 100:.2f}; «верхний» {cu * 100:.2f} / {float(o['upper']) * 100:.2f}; "
              f"отношение {mir_clean['mean'][2]:.3f} / {float(o['ratio']):.3f}  {'✅' if good else '⚠️ проверь весь вывод'}")
    print(f"  обычная сеть: «левый» {cl * 100:.2f}%, «верхний» {cu * 100:.2f}%; d(←→) {mir_clean['mean'][0]:.4f}, d(↑↓) {mir_clean['mean'][1]:.4f}, "
          f"отношение {mir_clean['mean'][2]:.3f}")

    conds = [("top12", top[K]), ("top4", top[4]), ("top24", top[24]), ("bottom12", bottom)] + [(f"rand{i:02d}", s) for i, s in enumerate(rand_sets)]
    if args.max_sets is not None:
        conds = conds[:4 + args.max_sets]
    rows = []
    t0 = time.time()
    for cname, heads in conds:
        cache = os.path.join(args.out, f"{key}_{cname}{tagx}.npz")
        if os.path.exists(cache):
            z = np.load(cache, allow_pickle=True)
            F = z["F"]
            mir = json.loads(str(z["mir"]))
        else:
            off = as_off(heads)
            F = feats_pass(model, meta, infos, device, mean, std, n_pre, nb, off=off, means=means)[0]
            mir = mirror_pass(model, sym_arrs, device, mean, std, n_pre, nb, off=off, means=means)
            np.savez(cache, F=F, mir=json.dumps(mir))
        res = probes(F, lab, folds, ok)
        row = dict(net=name, cond=cname, heads=hs_str(heads), n_heads=len(heads))
        for gi, g in ((1, "far"), (0, "cls")):
            dl = res[(gi, "left")] - clean[(gi, "left")]
            du = res[(gi, "upper")] - clean[(gi, "upper")]
            bl, bu, bd = (md.boot(a, np.random.default_rng(0)) for a in (dl, du, dl - du))
            row.update({f"{g}_left": res[(gi, 'left')].mean(), f"{g}_upper": res[(gi, 'upper')].mean(),
                        f"{g}_dleft": bl[0], f"{g}_dleft_lo": bl[1], f"{g}_dleft_hi": bl[2],
                        f"{g}_dupper": bu[0], f"{g}_dupper_lo": bu[1], f"{g}_dupper_hi": bu[2],
                        f"{g}_dd": bd[0], f"{g}_dd_lo": bd[1], f"{g}_dd_hi": bd[2]})
        for g in ("mean", "cls"):
            row.update({f"{g}_dh": mir[g][0], f"{g}_dv": mir[g][1], f"{g}_ratio": mir[g][2], f"{g}_ratio_lo": mir[g][3], f"{g}_ratio_hi": mir[g][4]})
        rows.append(row)
        print(f"  {cname:9s} [{hs_str(heads) if len(heads) <= 12 else str(len(heads)) + ' голов'}]: Δлевый {row['far_dleft'] * 100:+.2f} "
              f"[{row['far_dleft_lo'] * 100:+.2f}; {row['far_dleft_hi'] * 100:+.2f}], Δверхний {row['far_dupper'] * 100:+.2f}, "
              f"Δл−Δв {row['far_dd'] * 100:+.2f} [{row['far_dd_lo'] * 100:+.2f}; {row['far_dd_hi'] * 100:+.2f}];  CLS Δлевый {row['cls_dleft'] * 100:+.2f};  "
              f"d(←→) {mir['mean'][0]:.4f}, d(↑↓) {mir['mean'][1]:.4f}, отнош. {mir['mean'][2]:.3f}   ({time.time() - t0:.0f} с)", flush=True)
    clean_row = dict(net=name, cond="clean", heads="", n_heads=0, far_left=cl, far_upper=cu, cls_left=clean[(0, "left")].mean(), cls_upper=clean[(0, "upper")].mean())
    for g in ("mean", "cls"):
        clean_row.update({f"{g}_dh": mir_clean[g][0], f"{g}_dv": mir_clean[g][1], f"{g}_ratio": mir_clean[g][2],
                          f"{g}_ratio_lo": mir_clean[g][3], f"{g}_ratio_hi": mir_clean[g][4]})
    rows = [clean_row] + rows
    v2, detail = f2_verdict(rows)
    print(f"\n  F2: {detail} -> «{v2}»")
    out_m = OUT_M if not (args.limit or args.test_random_main) else f"heads_m_test{args.limit}.csv"
    out_f2 = OUT_F2 if not (args.limit or args.test_random_main) else f"heads_f2_test{args.limit}.csv"
    write_rows(out_m, mrows, list(mrows[0].keys()))
    write_rows(out_f2, rows, F2_FIELDS)
    for k, t in (("1", "прогон = run_layers; строки внимания в сумме 1"), ("2", "выключение собственным средним"), ("3", "мера на искусственных головах"),
                 ("4", "сетка / токены / блоки; нормировка внимания пустая"), ("5", "обычная сеть = опыт B"), ("6", "L00 50,0%; контроль > 90%"),
                 ("7", "случайные отличаются от обученной"), ("8", "30 случайных наборов: различны, по блокам как топ-12, без топ-12")):
        if k == "5" and (args.limit or args.test_random_main):
            print(f"Проверка 5 ({t}): (облако — не сравнивается)")
            continue
        print(f"Проверка {k} ({t}): {'✅' if ok[k] else '⚠️ проверь весь вывод'}")
    print(f"Проверка 9 (предупреждений «не сошлось»: {ok['warn']}): {'✅' if ok['warn'] == 0 else '⚠️ проверь весь вывод'}")
    print(f"\nЧисла дописаны в {out_m}, {out_f2}")


F2_FIELDS = (["net", "cond", "heads", "n_heads"] +
             [f"{g}_{s}" for g in ("far", "cls") for s in ("left", "upper", "dleft", "dleft_lo", "dleft_hi", "dupper", "dupper_lo", "dupper_hi",
                                                          "dd", "dd_lo", "dd_hi")] +
             [f"{g}_{s}" for g in ("mean", "cls") for s in ("dh", "dv", "ratio", "ratio_lo", "ratio_hi")])


def f2_verdict(rows):
    by = {r["cond"]: r for r in rows}
    t = by["top12"]
    rand = [float(r["far_dleft"]) for r in rows if r["cond"].startswith("rand")]
    a = float(t["far_dleft_lo"]) > 0
    b = len(rand) > 0 and all(float(t["far_dleft"]) > x for x in rand)
    c = float(t["far_dd_lo"]) > 0
    v = "зеркальные головы стирают «левый»" if a and b and c else "не показано"
    broken = float(t["far_upper"]) < 0.60
    if broken:
        v += " [разрушение]"
    n_ge = sum(1 for x in rand if x >= float(t["far_dleft"]))
    detail = (f"(а) нижняя граница Δлевый(топ-12) {float(t['far_dleft_lo']) * 100:+.2f} > 0: {'да' if a else 'нет'}; "
              f"(б) Δлевый(топ-12) {float(t['far_dleft']) * 100:+.2f} больше всех {len(rand)} случайных наборов "
              f"(макс. {max(rand) * 100 if rand else float('nan'):+.2f}; не меньше топ-12: {n_ge}; p = {(n_ge + 1) / (len(rand) + 1):.3f}): {'да' if b else 'нет'}; "
              f"(в) нижняя граница Δл−Δв {float(t['far_dd_lo']) * 100:+.2f} > 0: {'да' if c else 'нет'}")
    return v, detail


# ------------------------------------------------------------------
# F3: DINO, по одной голове
# ------------------------------------------------------------------
def run_f3(args):
    name, timm_name = NETS["dino"]
    device = pick_device()
    tagx = f"_test{args.limit}" if args.limit else ""
    ok = {"warn": 0}
    meta, infos, lab, folds, sym, sym_arrs = scenes(args)
    n = min(F3_SCENES, len(meta)) if not args.limit else len(meta) // 2
    sub = np.sort(np.random.default_rng(0).permutation(len(meta))[:n])
    meta_s = [meta[i] for i in sub]
    infos_s = [infos[i] for i in sub]
    lab_s = {q: lab[q][sub] for q in lab}
    folds_s = folds[sub]
    z = np.load(os.path.join(args.out, f"clean_dino{tagx}.npz"), allow_pickle=True)
    means = {b: torch.from_numpy(z["means"][b - 1]) for b in range(1, NB + 1)}
    zm = np.load(os.path.join(args.out, f"m_dino{tagx}.npz"))
    m_pt = zm["M"].mean(0)
    model, weights = mn.load(timm_name, 99 if args.test_random_main else None)
    _, _, n_pre, nb, mean, std = mn.setup(model, P)
    model.to(device)
    print(f"Опыт F, F3: {name} ({weights}), по одной голове; сцен {n}")
    F_clean = feats_pass(model, meta_s, infos_s, device, mean, std, n_pre, nb)[0]
    clean = probes(F_clean, lab_s, folds_s, ok, groups=(1,))
    rows = []
    t0 = time.time()
    heads = [(b, h) for b in range(1, NB + 1) for h in range(NH)][:args.f3_max or None]
    for b, h in heads:
        cache = os.path.join(args.out, f"f3_{b:02d}_{h:02d}{tagx}.npy")
        if os.path.exists(cache):
            F = np.load(cache)
        else:
            F = feats_pass(model, meta_s, infos_s, device, mean, std, n_pre, nb, off={b: [h]}, means=means)[0]
            np.save(cache, F)
        res = probes(F, lab_s, folds_s, ok, groups=(1,))
        dl = (res[(1, "left")] - clean[(1, "left")]).mean()
        du = (res[(1, "upper")] - clean[(1, "upper")]).mean()
        rows.append(dict(net=name, block=b, head=h, m=m_pt[b - 1, h], dleft=dl, dupper=du))
        if h == NH - 1:
            print(f"  блок {b:2d}: Δлевый по головам " + " ".join(f"{r['dleft'] * 100:+.1f}" for r in rows[-NH:]) + f"   ({time.time() - t0:.0f} с)", flush=True)
    out = OUT_F3 if not (args.limit or args.test_random_main) else f"heads_f3_test{args.limit}.csv"
    write_rows(out, rows, ["net", "block", "head", "m", "dleft", "dupper"])
    print(f"Проверка 9 (предупреждений «не сошлось»: {ok['warn']}): {'✅' if ok['warn'] == 0 else '⚠️ проверь весь вывод'}")
    print(f"Числа записаны в {out}")


# ------------------------------------------------------------------
# сводка
# ------------------------------------------------------------------
def summary(args):
    test = args.limit or args.test_random_main
    fm = OUT_M if not test else f"heads_m_test{args.limit}.csv"
    f2 = OUT_F2 if not test else f"heads_f2_test{args.limit}.csv"
    f3 = OUT_F3 if not test else f"heads_f3_test{args.limit}.csv"
    main_seed = "" if not args.test_random_main else "99"
    mr = [r for r in csv.DictReader(open(fm, encoding="utf-8")) if r["seed"] == main_seed]
    nets = [NETS[k][0] for k in ("dino", "deit3", "clip")]
    npos, v1, blocks_plus = {}, {}, {}
    print("ВЫВОД ПО ПРАВИЛАМ (реестр 174–176, записано до запуска):")
    for n in nets:
        rr = [r for r in mr if r["net"] == n]
        if not rr:
            continue
        plus = [(int(r["block"]), int(r["head"])) for r in rr if r["plus"] == "1"]
        minus = [r for r in rr if r["minus"] == "1"]
        npos[n] = len(plus)
        blocks_plus[n] = plus
        v1[n] = "зеркальные головы есть" if len(plus) >= 1 and len(plus) > len(minus) else "не показано"
        print(f"  F1 {n:17s}: N₊ = {len(plus)} ({hs_str(plus) or '—'}), N₋ = {len(minus)} -> «{v1[n]}»")
    rows2 = list(csv.DictReader(open(f2, encoding="utf-8"))) if os.path.exists(f2) else []
    v2, d_top = {}, {}
    for n in nets:
        rr = [r for r in rows2 if r["net"] == n]
        if not rr:
            continue
        v2[n], det = f2_verdict(rr)
        by = {r["cond"]: r for r in rr}
        d_top[n] = float(by["top12"]["far_dleft"])
        print(f"  F2 {n:17s}: {det} -> «{v2[n]}»")
        for c in ("top4", "top12", "top24", "bottom12"):
            r = by[c]
            print(f"      справка {c:8s}: Δлевый {float(r['far_dleft']) * 100:+.2f}, Δверхний {float(r['far_dupper']) * 100:+.2f}; CLS Δлевый {float(r['cls_dleft']) * 100:+.2f}; "
                  f"d(←→) {float(r['mean_dh']):.4f}, d(↑↓) {float(r['mean_dv']):.4f} (обычная {float(by['clean']['mean_dh']):.4f} / {float(by['clean']['mean_dv']):.4f})")
    rho3 = p3 = float("nan")
    v3 = "нет данных"
    if os.path.exists(f3):
        r3 = list(csv.DictReader(open(f3, encoding="utf-8")))
        x = np.array([float(r["m"]) for r in r3])
        y = np.array([float(r["dleft"]) for r in r3])
        rho3, p3 = mn.perm_p(x, y, +1)
        v3 = "связь следует" if rho3 > 0 and p3 < 0.05 else "не подтверждена"
        print(f"  F3 (DINO, {len(r3)} голов): ρ(m, Δлевый) = {rho3:+.3f}, p = {p3:.4f} -> «{v3}»  (мощность мала: «не подтверждена» ≠ «связи нет»)")
    if not test:
        hit = lambda b: "ПОПАЛ ✅" if b else "НЕ ПОПАЛ ❌"
        dn, tn, cn = nets
        print("\nСТАВКИ ЕРНАЗАРА (реестр 175, 15.11 09:59; мера m — после правки реестра 176):")
        if dn in npos:
            print(f"  175a N₊ у DINO 5–15 -> {npos[dn]}  {hit(5 <= npos[dn] <= 15)}")
            if npos[dn] > 0:
                k36 = sum(1 for b, _ in blocks_plus[dn] if 3 <= b <= 6)
                print(f"  175b большинство в блоках 3–6 -> {k36} из {npos[dn]}  {hit(k36 > npos[dn] / 2)}")
            else:
                print("  175b не засчитывается (N₊ = 0)")
        if dn in d_top:
            print(f"  175c Δлевый(топ-12) DINO от +1,0 до +3,0 -> {round(d_top[dn] * 100, 1) + 0.0:+.1f}  {hit(1.0 <= round(d_top[dn] * 100, 1) <= 3.0)}")
            print(f"  175d F2 DINO «не показано» -> «{v2[dn]}»  {hit(v2[dn].startswith('не показано'))}")
        if all(n in npos for n in nets):
            print(f"  175e DINO ≥ DeiT III ≥ CLIP и DINO > CLIP -> {npos[dn]} / {npos[tn]} / {npos[cn]}  "
                  f"{hit(npos[dn] >= npos[tn] >= npos[cn] and npos[dn] > npos[cn])}")
        if np.isfinite(rho3):
            print(f"  175f ρ в F3 от 0,00 до +0,40 -> {round(rho3, 2) + 0.0:+.2f}  {hit(0.0 <= round(rho3, 2) <= 0.40)}")
            print(f"  175g F3 «не значимо» (p ≥ 0,05) -> p = {p3:.4f}  {hit(p3 >= 0.05)}")
    with open("heads_rules.csv" if not test else f"heads_rules_test{args.limit}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rule", "net", "value", "p", "verdict"])
        for n in v1:
            w.writerow(["F1", n, npos[n], "", v1[n]])
        for n in v2:
            w.writerow(["F2", n, d_top[n], "", v2[n]])
        w.writerow(["F3", nets[0], rho3, p3, v3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", choices=list(NETS))
    ap.add_argument("--f3", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--out", default=os.path.expanduser("~/binding-vit-data/heads"))
    ap.add_argument("--lists", default="photos_lists.json")
    ap.add_argument("--ucm-dir", default=None, help="только для проверки в облаке")
    ap.add_argument("--no-aerial", action="store_true", help="только для проверки")
    ap.add_argument("--limit", type=int, default=0, help="только для проверки в облаке")
    ap.add_argument("--test-random-main", action="store_true", help="только для проверки в облаке: вместо обученной — случайная seed 99")
    ap.add_argument("--max-sets", type=int, default=None, help="только для проверки: число случайных наборов")
    ap.add_argument("--f3-max", type=int, default=0, help="только для проверки: число голов в F3")
    args = ap.parse_args()
    if args.summary:
        summary(args)
    elif args.f3:
        run_f3(args)
    elif args.net:
        run_net(args)
    else:
        ap.error("нужно --net dino|deit3|clip, --f3 или --summary")


if __name__ == "__main__":
    main()

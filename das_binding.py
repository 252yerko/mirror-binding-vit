"""
das_binding.py — Часть 3, шаг 2б: сколько направлений нужно, чтобы перенести «цвет левого близнеца»? (DINO ViT-B/16, DAS)
(реестр 75–76: дизайн, ставка Ерназара и правило записаны 9.11 20:38–20:50 ДО написания кода)

ВОПРОС
  Шаг 2а: до L05 чтение «левого» (фиксированная проба, дальний фон L12) зависит в основном от токенов
  близнецов, с L06 — в основном от остальных токенов. Там подменялся ВЕСЬ токен (768 чисел).
  Здесь спрашиваем: хватает ли подменить несколько НАПРАВЛЕНИЙ внутри токенов, и сколько?
  (A) близнецы, L03 — проверка «цвета»: S и Sp различаются только цветами близнецов, поэтому здесь ожидаем
      найти «где в токене близнеца лежит цвет» (оговорка Ерназара, реестр 75).
  (B) дальний фон, L08 — главный тест привязки: пикселей близнецов там нет, пиксели S и Sp совпадают; если
      малое подпространство фона переносит «цвет левого» — это связанная информация.

ЧТО ДЕЛАЕМ (distributed alignment search, Geiger et al. 2024)
  Пара картинок одной сцены: S и Sp. База — одна картинка, источник — другая (оба направления).
  Ортонормированная матрица Q (768×k, учится; сеть и проба заморожены). На слое L в каждом токене области
  (A: все токены с близнецом; B: дальний фон, ≥ 3 патчей от близнецов — как probe_v2):
      h_база ← h_база + (h_источник − h_база) · Q·Qᵀ
  то есть подменяется только проекция на k направлений. Дальше сеть считает до L12.
  Обучение: кросс-энтропия фиксированной пробы «левый» на дальнем фоне L12 к цвету левого близнеца
  ИСТОЧНИКА; Adam, lr 0,01 с линейным убыванием до 0, 400 шагов, пачка 16 сцен (пачка — по замеру скорости,
  шаги и убывание — по пилотам 1–2 на фолдах 1–4 ДО регистрации; одинаковы для всех сетей и условий), seed 0.
  k ∈ {1, 2, 4, 8, 16, 32}.

ПРОБА И РАЗБИЕНИЕ
  Проба «левый» на дальнем фоне L12 — как probe_v2 / patch_twins (StandardScaler + логрег C=0,1),
  обучена на ЧИСТОЙ сети на фолдах 1–4 (фолды по сценам — как probe_v2) и дальше не меняется.
  DAS учится на прогонах сцен фолдов 1–4; IIA меряется на отложенном фолде 0 (одно разбиение — ради
  времени; оговорено до запуска).

ЧТО МЕРЯЕМ
  IIA(k) — доля прогонов отложенного фолда, где после подмены фиксированная проба ставит цвет левого
  близнеца источника выше цвета левого близнеца базы (как T в patch_twins). Учитываются только прогоны,
  где проба на чистых картинках права и для базы, и для источника.
  Уровень полной подмены P — то же для подмены всех токенов области целиком (Q = единичная 768×768).
  Это НЕ потолок: DAS выбирает только полезные для пробы направления и может его превысить (тест 10.11).
  Интервалы: бутстрэп по сценам, 10 000 повторов, seed 0, 95%.

ПРАВИЛО ВЫВОДА (реестр 76, до запуска; в OSF — дословно). Отдельно для (A) и (B):
  Если P < 20% — «тест не применим».
  Иначе: наименьшее k из {1,2,4,8,16,32}, при котором (точечные оценки)
    (1) IIA_DINO(k) ≥ 0,5·P   и   (2) IIA_DINO(k) − max по 3 случайным DINO IIA_rand(k) ≥ 0,3·P.
  Если такого k нет — «не сжимается до 32 направлений».

КОНТРОЛИ
  Случайные DINO (seed 0/1/2) — та же процедура (их проба ≈ угадывание; учитываемых прогонов меньше).
  У каждого обучения DAS своё начальное Q (seed = 10000·0 + 1000·[B] + 500·[перемешанные] + k).
  Перемешанные метки (только DINO, все k): DAS учится на случайно переставленных целевых цветах;
  IIA меряется по настоящим. Ставка Ерназара (A): ошибся, если > 20%.
  Две части направления (Makelov et al.): Q проецируется на подпространство главных компонент активаций
  области (99% дисперсии, обучающие сцены) → IIA «активной» части рядом с IIA целого Q (справка).
  Позиция: доля Q в подпространстве 32 главных компонент позиционных эмбеддингов DINO (справка;
  случайное k-мерное подпространство дало бы ≈ 32/768 = 4,2%).
  Совпадение A и B: доля Q_B в подпространстве Q_A (при одинаковом k; случайное ≈ k/768) (справка).

ВСТРОЕННЫЕ ПРОВЕРКИ
  1. Признаки дальнего фона L12 в torch (float32) ≈ способ probe_v2 (float16 → среднее): разница мала;
     точность фиксированной пробы на отложенном фолде через torch = через sklearn (одинаковые ответы).
  2. Q = единичная 768×768 → прогон = полная подмена токенов области (разница ~0).
  3. Q = 0 направлений → прогон = чистый прогон базы (разница ~0).
  4. QᵀQ = I для выученных Q (разница < 1e-4).
  5. Кэш активаций на слое подмены = прямой прогон (разница < 1e-4).
  Сходимость: для каждого обучения — падение средней потери последних 10% шагов против предыдущих 10%;
  < 2% = плато; не вышедшие на плато перечисляются (ограничение).

СТАВКА ЕРНАЗАРА (реестр 75–76, 9.11, до кода)
  (A) k = 4 (35%), IIA(k=1) ≈ 30% (засчёт 20–40); (B) k = 8 (30%), IIA(k=1) ≈ 5% (засчёт 2–15).

ЗАПУСК (на Маке, из папки binding-vit; зарядка, крышка открыта)
  0) замер скорости (не результат):  caffeinate -dims python das_binding.py --bench 2>&1 | tee das_bench_log.txt
  1) пилот сходимости (не результат; фолд 0 не загружается):
       caffeinate -dims python das_binding.py --pilot --steps N --batch B 2>&1 | tee das_pilot_log.txt
  2) основной (после регистрации на OSF; 400 шагов и пачка 16 — значения по умолчанию, ~7 ч):
       caffeinate -dims python das_binding.py 2>&1 | tee das_binding_log.txt
  Результаты сохраняются после каждого k в ~/binding-vit-data/das_binding/; при обрыве та же команда
  продолжает с места. Временный кэш активаций (≈ 4,8 ГБ на сеть) там же, удаляется после каждой сети.

ЧЕГО ЭТОТ ОПЫТ НЕ ПОКАЖЕТ
  Найденное подпространство переносит то, что читает ОДНА фиксированная линейная проба; это не поведение
  сети. Одна модель, одно разбиение, один seed обучения DAS. Высокий IIA сам по себе не доказывает
  осмысленность (Geiger et al. §4.4) — поэтому решает сравнение со случайными сетями и с уровнем P.
"""
import argparse
import csv
import json
import os
import shutil
import time
import warnings

import numpy as np
import torch
from PIL import Image

import ablate_blocks as ab
import probe_v2 as pv
from extract import load_model, fingerprint, pick_device
from patch_twins import scene_tokens, norm_batch

DATA = ab.DATA
KS = [1, 2, 4, 8, 16, 32]
REGIONS = {"A": ("twins", 3), "B": ("far_bg", 8)}     # область токенов, слой подмены (выход блока L)
STEPS, BATCH_SC, LR, SEED = 400, 16, 0.01, 0        # выбраны ДО регистрации: пачка — по замеру, шаги — по пилоту (10.11)
LR_DECAY = True                                     # линейное убывание шага обучения до 0: выбрано пилотом 2 по правилу (10.11)
TEST_FOLD = 0
VAL_LIMIT = 0                                       # только пилот: сколько сцен проверочного фолда брать (0 = все)
N_BOOT = 10000
N_POS_PC = 32
EXPECTED_FP = ab.EXPECTED_FP
BETS = {"Ерназар": {"A": (4, (20.0, 40.0)), "B": (8, (2.0, 15.0))}}
HC = {}                                             # кэш активаций на слое L: {L: memmap [n_img,197,768] float32}


# ------------------------------------------------------------------
# 1. Прогон с подменой подпространства
# ------------------------------------------------------------------
def to_layer(model, x, L):
    h = model.prepare_tokens(x)
    for blk in model.blocks[:L]:
        h = blk(h)
    return h


def from_layer(model, h, L):
    for blk in model.blocks[L:]:
        h = blk(h)
    return h


def intervene(h, M, Q):
    """h: [B,197,768] прогон пары (строки 2i и 2i+1 — S и Sp одной сцены); M: [B,197] маска области;
    Q: [768,k] (или None = полная подмена). Источник строки b — строка b^1."""
    src = h[torch.arange(h.shape[0], device=h.device) ^ 1]
    d = src - h
    if Q is not None:
        d = (d @ Q) @ Q.T
    return h + M.unsqueeze(-1) * d


def far_mean(h, F):
    """F: [B,197] маска дальнего фона, нормированная на число токенов -> [B,768]."""
    return torch.einsum("bt,btd->bd", F, h)


class TorchProbe:
    """StandardScaler + LogisticRegression (sklearn) -> линейный слой torch."""
    def __init__(self, clf, device):
        sc, lr = clf[0], clf[1]
        self.classes = list(lr.classes_)
        W = lr.coef_ / sc.scale_[None, :]
        b = lr.intercept_ - (lr.coef_ * (sc.mean_ / sc.scale_)[None, :]).sum(1)
        self.W = torch.tensor(W.T, dtype=torch.float32, device=device)
        self.b = torch.tensor(b, dtype=torch.float32, device=device)

    def __call__(self, f):
        return f @ self.W + self.b


def make_Q(X, device=None):
    """Ортонормирование (QR) всегда на CPU: на MPS QR идёт ~2,6 с на шаг (замер das_profile 10.11), на CPU — миллисекунды.
    Градиент проходит через .cpu()/.to(). Результат кладётся на device (по умолчанию — туда, где был X)."""
    Q, _ = torch.linalg.qr(X.cpu())
    return Q.to(device if device is not None else X.device)


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


# ------------------------------------------------------------------
# 2. Данные
# ------------------------------------------------------------------
def load_data(limit, drop_fold=None):
    meta_all = [json.loads(l) for l in open(f"{DATA}/meta.jsonl")]
    colors = sorted({o[k] for r in meta_all for o in r["objects"] for k in ("color", "color_Sp")})
    folds_all = np.random.default_rng(0).permutation(len(meta_all)) % pv.N_FOLDS
    if drop_fold is not None:                        # пилот: фолд 0 убирается целиком, его не видим
        keep = folds_all != drop_fold
        meta_all = [r for r, k in zip(meta_all, keep) if k]
        folds_all = folds_all[keep]
    meta = meta_all[:limit] if limit else meta_all
    pv.QUERY = "left"
    pv.GROUPS[3] = "left_twin"
    n = len(meta)
    Mtw = np.zeros((2 * n, 197), np.float32)
    Mfar = np.zeros((2 * n, 197), np.float32)
    ginfo, y = [], []
    for i, rec in enumerate(meta):
        groups, lab = pv.scene_info(rec, colors)
        tk = scene_tokens(rec, groups["far_bg"], groups["unchanged"])
        ginfo.append({"far_bg": groups["far_bg"], "left_twin": groups["left_twin"], "upper_twin": []})
        for k, v in enumerate(("S", "Sp")):
            Mtw[2 * i + k, tk["twins"]] = 1
            Mfar[2 * i + k, groups["far_bg"]] = 1
            y.append(lab[v][0])
    imgs = np.zeros((2 * n, 224, 224, 3), np.uint8)
    for i, rec in enumerate(meta):
        for k, v in enumerate(("S", "Sp")):
            imgs[2 * i + k] = np.asarray(Image.open(f"{DATA}/images/{rec['id']}_{v}.png").convert("RGB"))
    y = np.array(y)                                   # цвет левого близнеца картинки (номер в colors, как probe_v2)
    return dict(meta=meta, colors=colors, folds=folds_all[:n], imgs=imgs, Mtw=Mtw, Mfar=Mfar, ginfo=ginfo, y=y)


# ------------------------------------------------------------------
# 3. Кэш активаций на слое подмены (сеть заморожена -> начало сети считается один раз)
# ------------------------------------------------------------------
@torch.no_grad()
def build_hcache(model, D, L, device, path):
    if os.path.exists(path):
        mm = np.load(path, mmap_mode="r")
        if mm.shape == (len(D["imgs"]), 197, 768):
            return mm
    n_img = len(D["imgs"])
    tmp = path + ".tmp.npy"
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n_img, 197, 768))
    bs = 2 * BATCH_SC
    for s in range(0, n_img, bs):
        rows = list(range(s, min(n_img, s + bs)))
        mm[rows] = to_layer(model, norm_batch(D["imgs"][rows]).to(device), L).float().cpu().numpy()
    mm.flush()
    del mm
    os.replace(tmp, path)
    return np.load(path, mmap_mode="r")


def h_at(model, D, rows, L, device):
    """Активации на выходе блока L для картинок rows: из кэша, если он есть, иначе прямой прогон."""
    mm = HC.get(L)
    if mm is not None:
        return torch.from_numpy(np.ascontiguousarray(mm[rows])).to(device)
    with torch.no_grad():
        return to_layer(model, norm_batch(D["imgs"][rows]).to(device), L)


def need_bytes(D):
    return len(D["imgs"]) * 197 * 768 * 4 * len({L for _, L in REGIONS.values()})


# ------------------------------------------------------------------
# 4. Одна сеть
# ------------------------------------------------------------------
@torch.no_grad()
def clean_far_L12(model, D, device):
    """Признаки дальнего фона L12: (a) как probe_v2 (float16 → среднее), (b) torch float32."""
    n_img = len(D["imgs"])
    f16 = np.zeros((n_img, 768), np.float32)
    f32 = np.zeros((n_img, 768), np.float32)
    bs = 2 * BATCH_SC
    for s in range(0, n_img, bs):
        rows = list(range(s, min(n_img, s + bs)))
        x = norm_batch(D["imgs"][rows]).to(device)
        h = from_layer(model, to_layer(model, x, 0), 0)
        f16[rows] = ab.group_feats(h, rows, D["ginfo"])[:, 0]
        F = torch.tensor(D["Mfar"][rows], device=device)
        f32[rows] = far_mean(h, F / F.sum(1, keepdim=True)).cpu().numpy()
    return f16, f32


def fit_probe(f16, D):
    from sklearn.exceptions import ConvergenceWarning
    img_fold = np.repeat(D["folds"], 2)
    tr = img_fold != TEST_FOLD
    clf = pv.make_model("logreg")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(f16[tr], D["y"][tr])
        nw = sum(issubclass(x.category, ConvergenceWarning) for x in w)
    return clf, nw


def batches(scenes, rng=None):
    sc = np.array(scenes)
    if rng is not None:
        sc = sc[rng.permutation(len(sc))]
    for s in range(0, len(sc), BATCH_SC):
        yield [r for i in sc[s:s + BATCH_SC] for r in (2 * i, 2 * i + 1)]


def masks(D, rows, region, device):
    M = torch.tensor((D["Mtw"] if region == "twins" else D["Mfar"])[rows], device=device)
    F = torch.tensor(D["Mfar"][rows], device=device)
    return M, F / F.sum(1, keepdim=True)


def train_sc_of(D):
    return [i for i in range(len(D["meta"])) if D["folds"][i] != TEST_FOLD]


def train_das(model, probe, D, region, L, k, device, shuffle=False, steps=None, val_at=None, val_fn=None, seed=None, decay=None):
    """STEPS шагов Adam по пачкам сцен обучающих фолдов (проходы с перемешиванием, пока не наберётся STEPS).
    Возвращает Q, потери по шагам и (только пилот) кривую проверочного IIA."""
    steps = steps or STEPS
    # своё начальное Q для каждой части/k/условия (иначе A и B стартуют с одной матрицы и «совпадают» искусственно)
    run_seed = (SEED if seed is None else seed) * 10000 + (1000 if region == "far_bg" else 0) + (500 if shuffle else 0) + k
    torch.manual_seed(run_seed)
    rng = np.random.default_rng(run_seed)
    A = torch.nn.Parameter(torch.randn(768, k) / 768 ** 0.5)      # на CPU: одинаковое начало на любом компьютере
    opt = torch.optim.Adam([A], lr=LR)
    decay = LR_DECAY if decay is None else decay
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s_: 1.0 - s_ / steps) if decay else None
    train_sc = train_sc_of(D)
    pmap = torch.tensor([probe.classes.index(c) for c in range(len(D["colors"]))], device=device)
    target_all = D["y"][np.arange(len(D["y"])) ^ 1]               # цвет левого у источника
    if shuffle:
        target_all = rng.permutation(target_all)
    losses, curve, step = [], [], 0
    while step < steps:
        for rows in batches(train_sc, rng):
            h = h_at(model, D, rows, L, device)
            M, F = masks(D, rows, region, device)
            out = from_layer(model, intervene(h, M, make_Q(A, device)), L)
            logits = probe(far_mean(out, F))
            tgt = pmap[torch.tensor(target_all[rows], device=device)]
            loss = torch.nn.functional.cross_entropy(logits, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            losses.append(float(loss.item()))
            step += 1
            if val_fn is not None and step in val_at:
                curve.append((step, val_fn(make_Q(A.detach(), device))))
            if step >= steps:
                break
    return make_Q(A.detach(), device), losses, curve


def convergence(losses):
    """Сходимость: средняя потеря последних 10% шагов против предыдущих 10%; плато — если упала меньше чем на 2%."""
    n = max(1, len(losses) // 10)
    a, b = float(np.mean(losses[-n:])), float(np.mean(losses[-2 * n:-n] or losses[-n:]))
    rel = (b - a) / b if b else 0.0
    return rel, bool(rel < 0.02)


@torch.no_grad()
def src_pref(model, probe, D, region, L, Q, device, full=False):
    """Для каждой картинки отложенного фолда: 1 — проба после подмены ставит цвет источника выше цвета базы."""
    test_sc = [i for i in range(len(D["meta"])) if D["folds"][i] == TEST_FOLD]
    if VAL_LIMIT:
        test_sc = test_sc[:VAL_LIMIT]
    pmap = [probe.classes.index(c) for c in range(len(D["colors"]))]
    out_rows, out_v = [], []
    for rows in batches(test_sc):
        h = h_at(model, D, rows, L, device)
        M, F = masks(D, rows, region, device)
        o = from_layer(model, intervene(h, M, None if full else Q), L)
        p = probe(far_mean(o, F)).cpu().numpy()
        for b, r in enumerate(rows):
            pt, pr = p[b, pmap[D["y"][r]]], p[b, pmap[D["y"][r ^ 1]]]
            out_rows.append(r)
            out_v.append(1.0 if pr > pt else (0.5 if pr == pt else 0.0))
    return np.array(out_rows), np.array(out_v)


def boot(per_scene_vals):
    v = np.asarray(per_scene_vals)
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(0)
    b = v[rng.integers(0, len(v), size=(N_BOOT, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def per_scene(rows, vals, ok):
    keep = ok[rows]
    r, v = rows[keep], vals[keep]
    sc = r // 2
    u, inv = np.unique(sc, return_inverse=True)
    return np.bincount(inv, weights=v) / np.bincount(inv)


@torch.no_grad()
def pca_basis(model, D, region, L, device, var=0.99, max_imgs=800):
    """Главные компоненты активаций области на слое L (обучающие сцены): базис, покрывающий 99% дисперсии."""
    train_sc = train_sc_of(D)[: max_imgs // 2]
    acc = []
    for rows in batches(train_sc):
        h = h_at(model, D, rows, L, device)
        M, _ = masks(D, rows, region, device)
        acc.append(h[M.bool()].float().cpu())
    X = torch.cat(acc)
    X = X - X.mean(0)
    _, S, Vh = torch.linalg.svd(X, full_matrices=False)
    c = torch.cumsum(S ** 2, 0) / (S ** 2).sum()
    d = int((c < var).sum()) + 1
    return Vh[:d].T.to(device), d


def frac_in(U, Q):
    """Доля Q (k направлений) в подпространстве с ортонормированным базисом U: ||UᵀQ||²/k."""
    return float(((U.T @ Q) ** 2).sum() / Q.shape[1])


def loss_summary(losses, every=10):
    return [round(float(np.mean(losses[i:i + every])), 4) for i in range(0, len(losses), every)]


def run_net(name, model, D, device, cache, tag, ok):
    """Результаты сохраняются после КАЖДОГО k: при обрыве та же команда продолжает с места."""
    path = f"{cache}/{tag}{name}.json"
    part = f"{cache}/{tag}{name}.partial.json"
    if os.path.exists(path):
        print(f"  результаты взяты из кэша {path}")
        return json.load(open(path))
    os.makedirs(cache, exist_ok=True)
    res = json.load(open(part)) if os.path.exists(part) else {}
    if res:
        print(f"  продолжаю с места: {part}")
    is_main = name == "DINO"
    probe = TorchProbe(D["_probe"], device)
    pos = model.pos_embed[0, 1:].detach().float().cpu()
    pos = pos - pos.mean(0)
    Upos = torch.linalg.svd(pos, full_matrices=False)[2][:N_POS_PC].T.to(device)
    for reg, (region, L) in REGIONS.items():
        t0 = time.time()
        r = res.get(reg)
        if r is None:
            rows, v = src_pref(model, probe, D, region, L, None, device, full=True)
            r = {"L": L, "P": boot(per_scene(rows, v, ok)), "n_runs": int(ok[rows].sum()), "k": {}, "shuffled": {},
                 "active": {}, "pos": {}, "conv": {}, "conv_shuffled": {}, "loss": {}, "_Q": {}}
            res[reg] = r
        Upca, d = (pca_basis(model, D, region, L, device) if is_main else (None, None))
        r["pca_dim"] = d
        for k in KS:
            ks = str(k)
            if ks in r["k"]:
                continue
            tk = time.time()
            Q, losses, _ = train_das(model, probe, D, region, L, k, device)
            err = float((Q.T @ Q - torch.eye(k, device=device)).abs().max())
            rows, v = src_pref(model, probe, D, region, L, Q, device)
            r["conv"][ks] = convergence(losses)
            r["loss"][ks] = loss_summary(losses)
            if is_main:
                Qa = make_Q(Upca @ (Upca.T @ Q))
                rows_a, v_a = src_pref(model, probe, D, region, L, Qa, device)
                r["active"][ks] = boot(per_scene(rows_a, v_a, ok)) + (frac_in(Upca, Q),)
                r["pos"][ks] = frac_in(Upos, Q)
                Qsh, l_sh, _ = train_das(model, probe, D, region, L, k, device, shuffle=True)
                rows_s, v_s = src_pref(model, probe, D, region, L, Qsh, device)
                r["shuffled"][ks] = boot(per_scene(rows_s, v_s, ok))
                r["conv_shuffled"][ks] = convergence(l_sh)
                r["_Q"][ks] = Q.cpu().numpy().tolist()
            r["k"][ks] = boot(per_scene(rows, v, ok)) + (err,)          # последним: «k готово»
            json.dump(res, open(part + ".tmp", "w"))
            os.replace(part + ".tmp", part)
            c = r["conv"][ks]
            print(f"    {name} {reg} k={k:2d}: IIA {r['k'][ks][0] * 100:5.1f}%   потеря: {'плато' if c[1] else '⚠️ ещё падает'} "
                  f"({c[0] * 100:+.1f}%)   {time.time() - tk:.0f} с (с начала части {time.time() - t0:.0f} с)", flush=True)
    json.dump(res, open(path, "w"))
    os.remove(part)
    return res


# ------------------------------------------------------------------
# 5. Проверки
# ------------------------------------------------------------------
@torch.no_grad()
def run_checks(model, D, f16, f32, clf, device):
    d1 = float(np.abs(f16 - f32).max())
    probe = TorchProbe(clf, device)
    test = np.repeat(D["folds"], 2) == TEST_FOLD
    a = clf.predict(f32[test])
    b = np.array(probe.classes)[probe(torch.tensor(f32[test], device=device)).argmax(1).cpu().numpy()]
    same = float((a == b).mean())
    print(f"Проверка 1: признаки torch ≈ probe_v2 (макс. разница {d1:.3f}, округление float16); "
          f"ответы пробы torch = sklearn на отложенном фолде: {same * 100:.2f}%  {'✅' if same > 0.999 and d1 < 0.05 else '⚠️ проверь весь вывод'}")
    rows = [r for i in range(2) for r in (2 * i, 2 * i + 1)]
    for region, L in REGIONS.values():
        M, F = masks(D, rows, region, device)
        h = to_layer(model, norm_batch(D["imgs"][rows]).to(device), L)
        full = from_layer(model, intervene(h, M, None), L)
        eye = from_layer(model, intervene(h, M, torch.eye(768, device=device)), L)
        zero = from_layer(model, intervene(h, M, torch.zeros(768, 1, device=device)), L)
        base = from_layer(model, h, L)
        d2, d3 = float((full - eye).abs().max()), float((zero - base).abs().max())
        print(f"Проверка 2 ({region}, L{L:02d}): Q = единичная → полная подмена, разница {d2:.2e}  {'✅' if d2 < 1e-2 else '⚠️'};  "
              f"Проверка 3: Q = 0 → чистая база, разница {d3:.2e}  {'✅' if d3 < 1e-4 else '⚠️'}")
        if HC.get(L) is not None:
            d5 = float((h_at(model, D, rows, L, device) - h).abs().max())
            print(f"Проверка 5 (L{L:02d}): кэш активаций = прямой прогон, разница {d5:.2e}  {'✅' if d5 < 1e-4 else '⚠️ проверь весь вывод'}")


def prepare_net(name, seed, D, device):
    model, weights, _ = load_model("dino", seed)
    fp = fingerprint(model)
    extra = ""
    if name.startswith("rand") and seed in EXPECTED_FP:
        extra = "✅ совпадает с extract.py" if abs(fp - EXPECTED_FP[seed]) < 0.01 else "⚠️ НЕ совпадает — проверь весь вывод"
    print(f"\n### {name}: {weights}   отпечаток весов {fp:.4f}  {extra}  устройство: {device}", flush=True)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    f16, f32 = clean_far_L12(model, D, device)
    clf, nw = fit_probe(f16, D)
    D["_probe"] = clf
    test = np.repeat(D["folds"], 2) == TEST_FOLD
    probe = TorchProbe(clf, device)
    p = probe(torch.tensor(f32, device=device)).detach().cpu().numpy()
    pm = [probe.classes.index(c) for c in range(len(D["colors"]))]
    y = D["y"]
    right = np.array([p[i, pm[y[i]]] > p[i, pm[y[i ^ 1]]] for i in range(len(y))])
    ok = right & right[np.arange(len(y)) ^ 1] & test
    print(f"  проба «левый» (дальний фон L12, обучающие фолды) на {'проверочном' if TEST_FOLD else 'отложенном'} фолде {TEST_FOLD}: "
          f"{right[test].mean() * 100:.1f}%; учитываемых прогонов (права и на базе, и на источнике): {int(ok.sum())}; «не сошлось»: {nw}")
    return model, f16, f32, clf, nw, ok


def open_hcache(model, D, device, cache, tag, name, use):
    HC.clear()
    if not use:
        return []
    os.makedirs(cache, exist_ok=True)
    free = shutil.disk_usage(cache).free
    if free < need_bytes(D) + 2e9:
        print(f"  ⚠️ на диске {free / 1e9:.1f} ГБ, кэшу нужно {need_bytes(D) / 1e9:.1f} ГБ + 2 ГБ запаса — считаю без кэша (медленнее)")
        return []
    paths = []
    t = time.time()
    for L in sorted({L for _, L in REGIONS.values()}):
        path = f"{cache}/h_{tag}{name}_L{L:02d}_{len(D['imgs'])}.npy"
        HC[L] = build_hcache(model, D, L, device, path)
        paths.append(path)
    print(f"  кэш активаций (L{', L'.join(f'{L:02d}' for L in HC)}; {need_bytes(D) / 1e9:.1f} ГБ): {time.time() - t:.0f} с", flush=True)
    return paths


def drop_hcache(paths, keep):
    HC.clear()
    if not keep:
        for p in paths:
            if os.path.exists(p):
                os.remove(p)                         # временный кэш самого скрипта (пересоздаётся при надобности)


def ram_gb():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return float("nan")


def mps_gb(device):
    try:
        return torch.mps.driver_allocated_memory() / 1e9 if device.type == "mps" else float("nan")
    except Exception:
        return float("nan")


# ------------------------------------------------------------------
# Режим --bench: сколько секунд на шаг (и сколько памяти) — чтобы выбрать пачку и шаги ДО регистрации
# ------------------------------------------------------------------
def bench(D, device, cache, seed=None):
    global BATCH_SC
    print(f"ЗАМЕР СКОРОСТИ (не результат). Память компьютера: {ram_gb():.0f} ГБ; сцен в замере: {len(D['meta'])}")
    model, f16, f32, clf, nw, ok = prepare_net("DINO", seed, D, device)
    probe = TorchProbe(clf, device)
    n_test_full, n_img_full, n_nets = 400, 4000, 4
    out = {}
    for bsz in (16, 8):
        BATCH_SC = bsz
        t = time.time()
        paths = open_hcache(model, D, device, cache, f"bench_b{bsz}_", "DINO", True)
        t_cache = (time.time() - t) / len(D["imgs"])        # с на картинку (оба слоя)
        out[bsz] = {"cache_s_per_img": t_cache}
        for reg, (region, L) in REGIONS.items():
            if device.type == "mps":
                torch.mps.empty_cache()
            train_das(model, probe, D, region, L, 32, device, steps=2)             # разогрев
            sync(device)
            t = time.time()
            train_das(model, probe, D, region, L, 32, device, steps=8)
            sync(device)
            s_step = (time.time() - t) / 8
            mem = mps_gb(device)
            n_sc = int((D["folds"] == TEST_FOLD).sum())
            t = time.time()
            src_pref(model, probe, D, region, L, torch.eye(768, device=device)[:, :32], device)
            sync(device)
            s_eval_sc = (time.time() - t) / max(1, n_sc)
            out[bsz][reg] = (s_step, s_eval_sc)
            print(f"  пачка {bsz:2d} сцен, ({reg}) {region} L{L:02d}: {s_step:5.2f} с/шаг ({s_step / bsz:.3f} с/сцену),  "
                  f"оценка {s_eval_sc:.3f} с/сцену,  память видеокарты {mem:.1f} ГБ", flush=True)
        drop_hcache(paths, False)
    print("\nОЦЕНКА ПОЛНОГО ЗАПУСКА (DINO + 3 случайные; у DINO ещё перемешанные метки и активная часть; 400 отложенных сцен):")
    for bsz, o in out.items():
        for steps in (100, 150, 200, 300, 400):
            tot = 0.0
            for reg in REGIONS:
                s_step, s_ev = o[reg]
                n_train = len(KS) * (2 + (n_nets - 1))        # DINO: обычное + перемешанные; случайные: обычное
                n_eval = len(KS) * (3 + (n_nets - 1)) + n_nets  # DINO: 3 оценки на k; случайные: 1; + P у каждой сети
                tot += n_train * steps * s_step + n_eval * n_test_full * s_ev
            tot += 2 * n_nets * n_img_full * o["cache_s_per_img"]     # кэш + чистый прогон для пробы
            print(f"  пачка {bsz:2d}, шагов {steps:3d} ({steps * bsz / 1600:.1f} прохода по 1600 сценам): ≈ {tot / 3600:5.1f} ч")
    print("\nПо этому выводу выбираются пачка и шаги для пилота.")


# ------------------------------------------------------------------
# Режим --pilot: на каком шаге обучение выходит на плато. Фолд 0 убран ЦЕЛИКОМ; проверка — на фолде 1
# ------------------------------------------------------------------
def pilot(D, device, cache, max_steps, seed=None):
    global TEST_FOLD, VAL_LIMIT
    TEST_FOLD, VAL_LIMIT = 1, 200
    print(f"ПИЛОТ (не результат): фолд 0 не загружен; проба и DAS учатся на фолдах 2–4, IIA — на {VAL_LIMIT} сценах фолда 1. "
          f"Пачка {BATCH_SC}, до {max_steps} шагов.")
    model, f16, f32, clf, nw, ok = prepare_net("DINO", seed, D, device)
    probe = TorchProbe(clf, device)
    paths = open_hcache(model, D, device, cache, f"pilot_b{BATCH_SC}_", "DINO", True)
    val_at = sorted({s for s in (10, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800) if s <= max_steps} | {max_steps})
    suggest = []
    for reg, k in (("B", 8), ("B", 32), ("A", 8), ("A", 32)):
        region, L = REGIONS[reg]
        rows, v = src_pref(model, probe, D, region, L, None, device, full=True)
        P = boot(per_scene(rows, v, ok))[0]

        def val_fn(Q, region=region, L=L):
            r_, v_ = src_pref(model, probe, D, region, L, Q, device)
            return boot(per_scene(r_, v_, ok))[0]
        t = time.time()
        Q, losses, curve = train_das(model, probe, D, region, L, k, device, steps=max_steps, val_at=val_at, val_fn=val_fn)
        fin = curve[-1][1]
        s_ok = next((s for i, (s, v_) in enumerate(curve) if all(v2 >= fin - 0.03 for _, v2 in curve[i:])), curve[-1][0])
        suggest.append(s_ok)
        sm = loss_summary(losses, 25)
        print(f"\n  ({reg}) {region} L{L:02d}, k={k}: уровень полной подмены P = {P * 100:.1f}%;  {time.time() - t:.0f} с")
        print("    шаг:  " + "  ".join(f"{s:>5d}" for s, _ in curve))
        print("    IIA:  " + "  ".join(f"{v_ * 100:5.1f}" for _, v_ in curve))
        print(f"    потеря (среднее по 25 шагов): {sm}")
        c = convergence(losses)
        print(f"    в пределах 3 п.п. от последнего значения с шага {s_ok};  потеря последних 10% шагов: {c[0] * 100:+.1f}% "
              f"({'плато' if c[1] else 'ещё падает'})", flush=True)
    drop_hcache(paths, False)
    print(f"\nПРЕДЛОЖЕНИЕ: шагов не меньше {max(suggest)} (наибольшее по 4 прогонам; пилот — не результат).")


# ------------------------------------------------------------------
# Режим --pilot2: устойчивость последнего шага. Постоянный шаг обучения против линейного убывания до 0.
# Фолд 0 убран ЦЕЛИКОМ; проверка — на всех сценах фолда 1. Правило выбора записано здесь ДО запуска пилота 2:
#   W = среднее по прогонам |IIA(шаг 300) − IIA(шаг 400)|  («дрожание» последнего снимка),
#   M = среднее по прогонам IIA(шаг 400).
#   Убывание принимается, если W_убыв < W_пост  и  M_убыв ≥ M_пост − 2 п.п.; иначе остаётся постоянный шаг.
# ------------------------------------------------------------------
PILOT2_RUNS = [("B", 32, 0), ("B", 32, 1), ("B", 32, 2), ("B", 2, 0), ("A", 32, 0)]


def pilot2(D, device, cache, seed=None):
    global TEST_FOLD, VAL_LIMIT
    TEST_FOLD, VAL_LIMIT = 1, 0
    print(f"ПИЛОТ 2 (не результат): фолд 0 не загружен; проба и DAS учатся на фолдах 2–4, IIA — на всех сценах фолда 1. "
          f"Пачка {BATCH_SC}, {STEPS} шагов; постоянный шаг обучения против линейного убывания до 0.")
    model, f16, f32, clf, nw, ok = prepare_net("DINO", seed, D, device)
    probe = TorchProbe(clf, device)
    paths = open_hcache(model, D, device, cache, f"pilot_b{BATCH_SC}_", "DINO", True)
    val_at = sorted({int(STEPS * 0.75), int(STEPS * 0.875), STEPS})
    out = {False: [], True: []}
    for reg, k, sd in PILOT2_RUNS:
        region, L = REGIONS[reg]

        def val_fn(Q, region=region, L=L):
            r_, v_ = src_pref(model, probe, D, region, L, Q, device)
            return boot(per_scene(r_, v_, ok))[0]
        for decay in (False, True):
            t = time.time()
            _, losses, curve = train_das(model, probe, D, region, L, k, device, steps=STEPS, val_at=val_at, val_fn=val_fn,
                                         seed=sd, decay=decay)
            v = [x * 100 for _, x in curve]
            out[decay].append((reg, k, sd, v))
            c = convergence(losses)
            print(f"  ({reg}) k={k:2d} seed {sd}  {'убывание  ' if decay else 'постоянный'}:  IIA на шагах {[s_ for s_, _ in curve]} = "
                  + " / ".join(f"{x:5.1f}" for x in v) + f"   потеря последних 10%: {c[0] * 100:+.1f}%   {time.time() - t:.0f} с", flush=True)
    drop_hcache(paths, False)
    W = {d: float(np.mean([abs(v[0] - v[-1]) for *_, v in out[d]])) for d in out}
    M = {d: float(np.mean([v[-1] for *_, v in out[d]])) for d in out}
    for d in (False, True):
        b32 = [v[-1] for reg, k, sd, v in out[d] if (reg, k) == ("B", 32)]
        print(f"\n{'ЛИНЕЙНОЕ УБЫВАНИЕ' if d else 'ПОСТОЯННЫЙ ШАГ'}: дрожание W = {W[d]:.1f} п.п.; средний IIA на последнем шаге M = {M[d]:.1f}%; "
              f"(B) k=32 по трём seed: {' / '.join(f'{x:.1f}' for x in b32)} (разброс {max(b32) - min(b32):.1f} п.п., ст. откл. {np.std(b32, ddof=1):.1f})")
    adopt = W[True] < W[False] and M[True] >= M[False] - 2.0
    print(f"\nПО ПРАВИЛУ (записано в коде до пилота 2): {'ЛИНЕЙНОЕ УБЫВАНИЕ шага обучения' if adopt else 'ПОСТОЯННЫЙ шаг обучения'} "
          f"(пилот — не результат).")


def main():
    global STEPS, BATCH_SC
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true", help="замер скорости и памяти (~5–10 мин), не результат")
    ap.add_argument("--pilot", action="store_true", help="пилот сходимости на фолдах 1–4 (фолд 0 не трогается), не результат")
    ap.add_argument("--pilot2", action="store_true", help="пилот 2: постоянный шаг обучения против убывания (фолд 0 не трогается)")
    ap.add_argument("--limit", type=int, default=0, help="только для проверки: первые N сцен")
    ap.add_argument("--main-seed", type=int, default=None, help="только для проверки в облаке: случайные веса вместо DINO")
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--steps", type=int, default=None, help=f"шагов обучения DAS (по умолчанию {STEPS})")
    ap.add_argument("--batch", type=int, default=None, help=f"сцен в пачке (по умолчанию {BATCH_SC})")
    ap.add_argument("--no-hcache", action="store_true", help="не кэшировать активации (медленнее; для проверки)")
    ap.add_argument("--keep-hcache", action="store_true", help="не удалять кэш активаций после сети")
    ap.add_argument("--cache", default=os.path.expanduser("~/binding-vit-data/das_binding"))
    args = ap.parse_args()
    STEPS = args.steps or STEPS
    BATCH_SC = args.batch or BATCH_SC
    t_start = time.time()
    device = pick_device()
    if args.bench:
        bench(load_data(args.limit or 300), device, args.cache, args.main_seed)
        print(f"\nВсего: {time.time() - t_start:.0f} с")
        return
    if args.pilot2:
        pilot2(load_data(args.limit, drop_fold=0), device, args.cache, args.main_seed)
        print(f"\nВсего: {time.time() - t_start:.0f} с")
        return
    if args.pilot:
        pilot(load_data(args.limit, drop_fold=0), device, args.cache, STEPS, args.main_seed)
        print(f"\nВсего: {time.time() - t_start:.0f} с")
        return
    stem = "das_binding" + (f"_test{args.limit}" if args.limit else "")
    tag = (f"test{args.limit}_" if args.limit else "") + f"s{STEPS}_b{BATCH_SC}_" + \
          (f"mainseed{args.main_seed}_" if args.main_seed is not None else "")
    D = load_data(args.limit)
    if args.limit:
        print(f"⚠️ ПРОВЕРОЧНЫЙ прогон: {len(D['meta'])} сцен. Для настоящих чисел запускай без --limit.")
    n_test = int((D["folds"] == TEST_FOLD).sum())
    print(f"Сцен: {len(D['meta'])}; отложенный фолд {TEST_FOLD}: {n_test} сцен; обучение DAS: {len(D['meta']) - n_test} сцен")
    print(f"k = {KS}; (A) близнецы L{REGIONS['A'][1]:02d}, (B) дальний фон L{REGIONS['B'][1]:02d}; "
          f"шагов {STEPS}, пачка {BATCH_SC} сцен, lr {LR}, seed {SEED}")
    nets = [("DINO", args.main_seed)] + [(f"rand{s}", s) for s in range(args.n_random)]
    res, oks, warn = {}, {}, 0
    for name, seed in nets:
        if os.path.exists(f"{args.cache}/{tag}{name}.json"):
            res[name] = json.load(open(f"{args.cache}/{tag}{name}.json"))
            oks[name] = res[name]["_ok"]
            warn += res[name]["_warn"]
            print(f"\n### {name}: результаты взяты из кэша")
            continue
        model, f16, f32, clf, nw, ok = prepare_net(name, seed, D, device)
        warn += nw
        paths = open_hcache(model, D, device, args.cache, tag, name, not args.no_hcache)
        if name == "DINO":
            run_checks(model, D, f16, f32, clf, device)
        r = run_net(name, model, D, device, args.cache, tag, ok)
        r["_ok"], r["_warn"] = int(ok.sum()), nw
        json.dump(r, open(f"{args.cache}/{tag}{name}.json", "w"))
        res[name] = r
        oks[name] = int(ok.sum())
        drop_hcache(paths, args.keep_hcache)
        del model
    report(res, oks, warn, stem)
    print(f"\nВсего: {time.time() - t_start:.0f} с")


def verdict(res, reg):
    P = res["DINO"][reg]["P"][0]
    if P is None or np.isnan(P):
        return np.nan, None, "нет учитываемых прогонов (только в проверочном прогоне)"
    if P < 0.20:
        return P, None, "тест не применим (P < 20%)"
    for k in KS:
        iia = res["DINO"][reg]["k"][str(k) if str(k) in res["DINO"][reg]["k"] else k][0]
        rmax = max(r[reg]["k"][str(k) if str(k) in r[reg]["k"] else k][0] for n, r in res.items() if n != "DINO")
        if iia >= 0.5 * P and iia - rmax >= 0.3 * P:
            return P, k, f"наименьшее k = {k}"
    return P, None, "не сжимается до 32 направлений"


def g(d, k):
    return d[str(k)] if str(k) in d else d[k]


def report(res, oks, warn, stem):
    print("\n" + "=" * 110)
    rows_csv = []
    for reg, (region, L) in REGIONS.items():
        rD = res["DINO"][reg]
        P = rD["P"]
        print(f"\n({reg}) {region}, L{L:02d}.  Уровень полной подмены P (вся область): {P[0] * 100:.1f}% [{P[1] * 100:.1f}–{P[2] * 100:.1f}]"
              + "".join(f"   {n}: {r[reg]['P'][0] * 100:.1f}%" for n, r in res.items() if n != "DINO"))
        print(f"   k    IIA DINO, % [95%]        " + "  ".join(f"{n:>6s}" for n in res if n != "DINO")
              + "   перемеш.   активная часть (доля Q)   позиция   QᵀQ−I")
        for k in KS:
            m = g(rD["k"], k)
            line = f"  {k:2d}   {m[0] * 100:5.1f} [{m[1] * 100:5.1f}–{m[2] * 100:5.1f}]       "
            line += "  ".join(f"{g(r[reg]['k'], k)[0] * 100:6.1f}" for n, r in res.items() if n != "DINO")
            sh, ac = g(rD["shuffled"], k), g(rD["active"], k)
            line += f"     {sh[0] * 100:5.1f}       {ac[0] * 100:5.1f} ({ac[3] * 100:4.1f}%)          {g(rD['pos'], k) * 100:4.1f}%   {m[3]:.0e}"
            print(line)
            rows_csv.append([reg, region, L, k, "DINO", *[f"{v:.5f}" for v in m[:3]]])
            for n, r in res.items():
                if n != "DINO":
                    rows_csv.append([reg, region, L, k, n, *[f"{v:.5f}" for v in g(r[reg]["k"], k)[:3]]])
            rows_csv.append([reg, region, L, k, "DINO_shuffled", *[f"{v:.5f}" for v in sh[:3]]])
            rows_csv.append([reg, region, L, k, "DINO_active_part", *[f"{v:.5f}" for v in ac[:3]]])
        print(f"   главных компонент активаций области (99% дисперсии): {rD['pca_dim']};  позиция: {N_POS_PC} ГК позиционных эмбеддингов "
              f"(случайное подпространство ≈ {N_POS_PC / 768 * 100:.1f}%)")
    QA, QB = res["DINO"]["A"]["_Q"], res["DINO"]["B"]["_Q"]
    print("\nСовпадение подпространств B и A (доля Q_B в Q_A при одинаковом k; случайное ≈ k/768):")
    print("   " + "   ".join(f"k={k}: {frac_in(torch.tensor(g(QA, k)), torch.tensor(g(QB, k))) * 100:.1f}% ({k / 768 * 100:.1f}%)" for k in KS))
    print(f"\nУчитываемых прогонов: " + ", ".join(f"{n} {v}" for n, v in oks.items()))
    print(f"Предупреждений «обучение пробы не сошлось»: {warn}  {'✅' if warn == 0 else '⚠️ проверь весь вывод'}")
    errs = [g(r[reg]["k"], k)[3] for r in res.values() for reg in REGIONS for k in KS]
    print(f"Проверка 4 (QᵀQ = I): макс. отклонение {max(errs):.1e}  {'✅' if max(errs) < 1e-4 else '⚠️ проверь весь вывод'}")
    conv = [(n, reg, key, k, c) for n, r in res.items() for reg in REGIONS for key in ("conv", "conv_shuffled")
            for k, c in r[reg].get(key, {}).items()]
    bad = [f"{n} {reg} k={k}{' (перемеш.)' if key == 'conv_shuffled' else ''}" for n, reg, key, k, c in conv if not c[1]]
    print(f"Сходимость (потеря последних 10% шагов упала меньше чем на 2%): {len(conv) - len(bad)} из {len(conv)} обучений"
          + ("  ✅" if not bad else f"  ⚠️ ещё падает: {', '.join(bad)} — это ограничение, записать в статью"))

    print("\nВЫВОД ПО ПРАВИЛУ (реестр 76, записано до запуска):")
    vs = {}
    for reg in REGIONS:
        P, k, txt = verdict(res, reg)
        vs[reg] = k
        print(f"  ({reg}) P = {P * 100:.1f}%  ->  {txt}")
    print("  Оговорки: одна модель, одно разбиение (фолд 0 отложен), один seed DAS; переносится то, что читает одна "
          "фиксированная линейная проба, — не поведение сети.")
    print("\nСТАВКА ЕРНАЗАРА (реестр 75–76, 9.11, до кода):")
    for who, b in BETS.items():
        for reg in REGIONS:
            kb, rng_ = b[reg]
            k1 = round(g(res["DINO"][reg]["k"], 1)[0] * 100, 1)
            s = f"  {who} ({reg}): k = {kb} -> вышло {vs[reg] if vs[reg] else 'нет такого'}   {'ПОПАЛ ✅' if vs[reg] == kb else 'НЕ ПОПАЛ ❌'}"
            if rng_:
                s += f";  IIA(k=1) {rng_[0]}–{rng_[1]}% -> {k1:.1f}%   {'ПОПАЛ ✅' if rng_[0] <= k1 <= rng_[1] else 'НЕ ПОПАЛ ❌'}"
            print(s)
    with open(f"{stem}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["part", "region", "layer", "k", "net", "IIA", "IIA_lo", "IIA_hi"])
        w.writerows(rows_csv)
    plot(res, stem)
    print(f"\nГрафик: {stem}.png   Числа: {stem}.csv")


def plot(res, stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (reg, (region, L)) in zip(axes, REGIONS.items()):
        rD = res["DINO"][reg]
        m = [g(rD["k"], k)[0] * 100 for k in KS]
        lo = [g(rD["k"], k)[1] * 100 for k in KS]
        hi = [g(rD["k"], k)[2] * 100 for k in KS]
        ax.plot(KS, m, "o-", color="C0", label="DINO")
        ax.fill_between(KS, lo, hi, color="C0", alpha=0.2)
        ax.plot(KS, [g(rD["shuffled"], k)[0] * 100 for k in KS], "s--", color="C3", lw=1, label="DINO, перемешанные метки")
        for n, r in res.items():
            if n != "DINO":
                ax.plot(KS, [g(r[reg]["k"], k)[0] * 100 for k in KS], ":", color="gray", lw=1)
        ax.axhline(rD["P"][0] * 100, color="C0", ls="--", lw=0.8, label="P — полная подмена области")
        ax.axhline(rD["P"][0] * 50, color="k", ls=":", lw=0.8, label="0,5·P")
        ax.set_xscale("log", base=2)
        ax.set_xticks(KS, [str(k) for k in KS])
        ax.set_xlabel("k (число направлений)")
        ax.set_title(f"({reg}) {'близнецы' if region == 'twins' else 'дальний фон'}, L{L:02d}", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("IIA, % (отложенный фолд)")
    axes[0].legend(fontsize=7)
    fig.suptitle("DINO: DAS — подмена k направлений, чтение фиксированной пробой «левый» (дальний фон L12); серые — случайные DINO", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=150)


if __name__ == "__main__":
    main()

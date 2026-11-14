"""
photos_explore.py — опыт C (OSF dprnk): РАЗВЕДОЧНЫЙ разбор C2 после запуска. Не правило, не регистрировалось.
Добавлено 15.11 (реестр 167).

ЗАЧЕМ:
  На фото-фоне обычная проба дальнего фона дала ≈ 50% и на «левый», и на «верхний» у всех 4 сетей (C2).
  Вопрос: сведений о близнецах в дальнем фоне нет совсем — или они есть, но тонут в содержимом фото,
  которое меняется от сцены к сцене сильнее, чем след близнецов?
  S и Sp лежат на одном и том же фото, поэтому в разности Δ = h(S) − h(Sp) (дальний фон, последний слой)
  содержимое фото сокращается, а след привязки остаётся (и меняет знак при обмене цветов).

ЧТО СЧИТАЕТСЯ (все — по кэшу photos.py --c2, те же 2000 сцен и 5 фолдов по сценам):
  (1) обычная проба на CLS (фото и серый фон) — справка, как в C2;
  (2) d(S, Sp) / d(разные сцены): косинусное расстояние дальнего фона S и Sp относительно разброса между сценами;
  (3) проба на Δ: StandardScaler + логрег C = 0,1; метка — какой из двух цветов близнецов (по постоянному порядку цветов)
      отвечает на вопрос «левый» / «верхний» в S; доля большего класса печатается (≈ 52%); 95% интервал — бутстрэп
      по сценам; разрыв = «верхний» − «левый». Контроль: те же числа у 2 случайных сетей каждой архитектуры и на L00
      (там Δ = 0, проба угадывает большинство).
  ОГРАНИЧЕНИЕ: проба на Δ видит пару (S, Sp) — как несимметричная проба опыта «A + D»: она показывает, что сведения
  восстанавливаются из пары, а не что сеть хранит их читаемо в одной картинке.

ЗАПУСК (из папки binding-vit):  python photos_explore.py 2>&1 | tee photos_explore_log.txt   (~5–10 мин)
"""
import argparse
import json
import os
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import flip_test as ft
import mirror_decomp as md
import probe_big as pb

DATA = "data_v2"
NETS = [("DINO B/16", "DINO_B_16", 16), ("CLIP-OpenAI B/16", "CLIP-OpenAI_B_16", 16),
        ("DeiT III B/16", "DeiT_III_B_16", 16), ("DINOv2-reg B/14", "DINOv2-reg_B_14", 14)]
TAGS = ["photo", "rand0_photo", "rand1_photo", "gray"]


def cv_bin(X, y, folds):
    acc = np.zeros(len(y))
    for f in range(md.N_FOLDS):
        te = folds == f
        c = make_pipeline(StandardScaler(), LogisticRegression(C=md.C_REG, max_iter=3000)).fit(X[~te], y[~te])
        acc[te] = c.predict(X[te]) == y[te]
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.expanduser("~/binding-vit-data/photos/cache"))
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    meta = [json.loads(line) for line in open(f"{DATA}/meta.jsonl")]
    colors = sorted({ob[k] for r in meta for ob in r["objects"] for k in ("color", "color_Sp")})
    folds = np.random.default_rng(0).permutation(len(meta)) % md.N_FOLDS
    info = {}
    for P in (16, 14):
        pb.FAR = 3 if P == 16 else 4
        info[P] = [md.scene_info(r, colors, P, 224 // P) for r in meta]
    print("Опыт C, РАЗВЕДОЧНО (после запуска, не правило регистрации dprnk): дальний фон, последний слой\n")
    for name, safe, P in NETS:
        inf = info[P]
        lab = {q: np.array([[inf[i][1][q][1][v] for v in ("S", "Sp")] for i in range(len(meta))]) for q in ("left", "upper")}
        ys = {q: (lab[q][:, 0, 0] < lab[q][:, 0, 1]).astype(int) for q in lab}
        base = {q: max(ys[q].mean(), 1 - ys[q].mean()) for q in ys}
        print(f"### {name}: доля большего класса «левый» {base['left'] * 100:.1f}%, «верхний» {base['upper'] * 100:.1f}%")
        for tag in TAGS:
            F = np.load(os.path.join(args.cache, f"c2_{safe}_{tag}.npz"))["F"].astype(np.float32)
            far = F[1, :, :, 1]
            msg = f"  {tag:12s}:"
            if not tag.startswith("rand"):
                acc = {q: md.probe_plain(F[1, :, :, 0], lab[q][:, :, 0], lab[q][:, :, 1], folds)[0].mean() for q in lab}
                dp = ft.cos_dist(far[:, 0], far[:, 1]).mean()
                ds = ft.cos_dist(far[:, 0], far[np.random.default_rng(0).permutation(len(meta)), 0]).mean()
                msg += f" CLS «левый» {acc['left'] * 100:.1f} «верхний» {acc['upper'] * 100:.1f};  d(S,Sp)/d(сцены) {dp / ds:.4f};"
            res = {}
            for li in (0, 1):
                dlt = F[li, :, 0, 1] - F[li, :, 1, 1]
                res[li] = {q: cv_bin(dlt, ys[q], folds) for q in ys}
            a = res[1]
            bl, bu = md.boot(a["left"], np.random.default_rng(0)), md.boot(a["upper"], np.random.default_rng(0))
            g = md.boot(a["upper"] - a["left"], np.random.default_rng(0))
            msg += (f"\n      проба на Δ: «левый» {bl[0] * 100:.1f} [{bl[1] * 100:.1f}; {bl[2] * 100:.1f}], «верхний» {bu[0] * 100:.1f} "
                    f"[{bu[1] * 100:.1f}; {bu[2] * 100:.1f}], разрыв {g[0] * 100:.1f} [{g[1] * 100:.1f}; {g[2] * 100:.1f}] п.п.  "
                    f"(L00: «левый» {res[0]['left'].mean() * 100:.1f}, «верхний» {res[0]['upper'].mean() * 100:.1f})")
            print(msg, flush=True)
    print("\nРазведочно, после запуска; не правило регистрации dprnk. Проба на Δ видит пару (S, Sp).")


if __name__ == "__main__":
    main()

"""
probe_sanity.py — проверка, что весь конвейер собран правильно.

ИДЕЯ:
  Прежде чем задавать модели трудные вопросы (про привязку), задаём лёгкий:
  "какого цвета этот объект?" и "какой он формы?".
  Если активации сохранены правильно и патчи сопоставлены объектам правильно,
  простой линейный классификатор ("проба") должен угадывать цвет почти на 100%.
  Если получится около 17% (1 из 6, угадывание наугад) — где-то ошибка
  в индексах, и дальше идти нельзя.

КАК УСТРОЕНО:
  1. Для каждого объекта на каждой картинке берём патчи, которые он покрывает
     (список patches16 из meta.jsonl), и усредняем их векторы -> один вектор
     768 чисел на объект. Это "представление объекта" на данном слое.
  2. Ответ (метка) — цвет объекта. В S' берём цвет из color_Sp: у двух
     переставленных объектов он другой.
  3. Делим СЦЕНЫ на обучение (75%) и проверку (25%). Важно: S и S' одной
     сцены всегда попадают в одну и ту же часть. Иначе проба "подсмотрит"
     почти ту же картинку и точность будет завышена.
  4. Обучаем логистическую регрессию на каждом из 13 слоёв, меряем точность
     на проверочных сценах. То же для формы.

КОНТРОЛЬ:
  На слое L00 (до всякого смешивания) берём ФОНОВЫЕ патчи, соседние с объектом, но
  не входящие в него. Они не видят объект -> точность цвета должна быть около
  "наивного" уровня (всегда называть самый частый цвет). Если она высокая — патчи перепутаны местами (ошибка в индексах).

ЗАПУСК (после extract.py):
  python probe_sanity.py
  -> печатает таблицу и сохраняет probe_sanity.png
"""

import csv
import json
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA, ACTS = "data", "acts"
G = 14                                   # сетка 14x14 патчей


def load_samples():
    """Список объектов: (строка в acts, номера токенов, цвет, форма, id сцены)."""
    meta = {r["id"]: r for r in map(json.loads, open(f"{DATA}/meta.jsonl"))}
    samples, skipped = [], 0
    for row in csv.DictReader(open(f"{ACTS}/index.csv")):
        rec = meta[row["id"]]
        for o in rec["objects"]:
            if not o["patches16"]:
                skipped += 1
                continue
            color = o["color"] if row["version"] == "S" else o["color_Sp"]
            tokens = [p + 1 for p in o["patches16"]]          # +1: токен 0 — это CLS
            samples.append(dict(row=int(row["row"]), tokens=tokens, color=color,
                                shape=o["shape"], sid=row["id"]))
    return samples, meta, skipped


def background_neighbours(rec, obj_patches):
    """Патчи рядом с объектом, в которых нет НИ ОДНОГО объекта (по bbox)."""
    occupied = set()
    for o in rec["objects"]:
        x0, y0, x1, y1 = o["bbox"]
        for py in range(max(0, y0 // 16), min(G, y1 // 16 + 1)):
            for px in range(max(0, x0 // 16), min(G, x1 // 16 + 1)):
                occupied.add(py * G + px)
    out = set()
    for p in obj_patches:
        py, px = divmod(p, G)
        for dy in (-2, -1, 0, 1, 2):
            for dx in (-2, -1, 0, 1, 2):
                qy, qx = py + dy, px + dx
                if 0 <= qy < G and 0 <= qx < G and qy * G + qx not in occupied:
                    out.add(qy * G + qx)
    return sorted(out)


def features(arr, samples, key="tokens"):
    X = np.zeros((len(samples), arr.shape[2]), dtype=np.float32)
    for i, s in enumerate(samples):
        X[i] = np.asarray(arr[s["row"], s[key]], dtype=np.float32).mean(0)
    return X


def probe(X, y, train, test):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))
    clf.fit(X[train], y[train])
    return (clf.predict(X[test]) == y[test]).mean()


def main():
    samples, meta, skipped = load_samples()
    info = json.load(open(f"{ACTS}/info.json"))
    n_layers = 13

    # --- разбиение по сценам, фиксированный seed
    sids = sorted({s["sid"] for s in samples})
    random.Random(0).shuffle(sids)
    test_sids = set(sids[: len(sids) // 4])
    test = np.array([s["sid"] in test_sids for s in samples])
    train = ~test

    y_color = np.array([s["color"] for s in samples])
    y_shape = np.array([s["shape"] for s in samples])
    print(f"Объектов: {len(samples)} (обучение {train.sum()}, проверка {test.sum()}), "
          f"без патчей пропущено: {skipped}")
    print(f"Случайное угадывание: цвет ~{1/6:.0%}, форма ~{1/5:.0%}\n")

    acc_c, acc_s = [], []
    print("слой   цвет    форма")
    for l in range(n_layers):
        arr = np.load(f"{ACTS}/L{l:02d}.npy", mmap_mode="r")
        X = features(arr, samples)
        acc_c.append(probe(X, y_color, train, test))
        acc_s.append(probe(X, y_shape, train, test))
        print(f"L{l:02d}   {acc_c[-1]:6.1%}  {acc_s[-1]:6.1%}")

    # --- контроль: фоновые патчи на L00
    bg = []
    for s in samples:
        rec = meta[s["sid"]]
        obj = [t - 1 for t in s["tokens"]]
        s["bg"] = [p + 1 for p in background_neighbours(rec, obj)]
    keep = np.array([len(s["bg"]) > 0 for s in samples])
    arr0 = np.load(f"{ACTS}/L00.npy", mmap_mode="r")
    sub = [s for s, k in zip(samples, keep) if k]
    Xbg = features(arr0, sub, key="bg")
    acc_bg = probe(Xbg, y_color[keep], train[keep], test[keep])
    # "наивный" уровень: всегда называть самый частый цвет в обучающей части
    vals, cnt = np.unique(y_color[train], return_counts=True)
    majority = (y_color[test] == vals[cnt.argmax()]).mean()
    print(f"\nКонтроль (фон рядом с объектом, L00): цвет {acc_bg:.1%}")
    print(f"  для сравнения, 'всегда самый частый цвет': {majority:.1%}  <- контроль должен быть близко к этому")

    # --- вердикт
    ok = min(acc_c) > 0.9 and acc_bg < majority + 0.10
    print("\nВЕРДИКТ:", "✅ конвейер работает, можно двигаться дальше" if ok
          else "⚠️ что-то не так — пришли Claude весь вывод")

    # --- график
    fig, ax = plt.subplots(figsize=(7, 4))
    L = np.arange(n_layers)
    ax.plot(L, acc_c, "o-", label="цвет объекта")
    ax.plot(L, acc_s, "s-", label="форма объекта")
    ax.axhline(1 / 6, ls="--", c="gray", lw=1, label="наугад (цвет)")
    ax.scatter([0], [acc_bg], marker="x", c="red", s=70, zorder=5, label="контроль: фон, L00")
    ax.set_xticks(L, [f"L{l:02d}" for l in L], fontsize=8)
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("точность на проверочных сценах")
    ax.set_title(f"Sanity check · {info['model']} · {info['n_scenes']} сцен")
    ax.legend(loc="center right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("probe_sanity.png", dpi=150)
    print("График: probe_sanity.png")


if __name__ == "__main__":
    main()

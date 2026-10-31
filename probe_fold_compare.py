"""probe_fold_compare.py — сравнение DINO и CLIP в опыте «складывание» (6.10; разбор ПОСЛЕ запуска, не был записан до запуска).
Ничего не обучает: берёт баллы сцен из probe_fold_scores.npz (DINO) и probe_fold_clip_scores.npz (CLIP).
D = (верхний − левый) − (ближе к середине ↑↓ − ближе к середине ←→): та часть потери горизонтали, которая касается
только СТОРОНЫ (лево/право), за вычетом общей слабости горизонтали. Сцены у обеих сетей одни и те же (1043 сцены
общего подмножества), поэтому разницу D(DINO) − D(CLIP) оцениваем парным бутстрэпом по сценам (1000 повторов).
Запуск (из binding-vit):  python probe_fold_compare.py"""
import numpy as np
A, B = np.load("probe_fold_scores.npz"), np.load("probe_fold_clip_scores.npz")
SUB = A["sub"]
assert (SUB == B["sub"]).all(), "подмножества сцен у DINO и CLIP разные"
idx = np.where(SUB)[0]
rng = np.random.default_rng(0)
BOOT = [rng.choice(len(idx), len(idx)) for _ in range(1000)]
def dvec(Z, net, g, L):
    s = {q: Z[f"{net}|{q}|{g}|L{L:02d}"][idx] for q in ("left", "upper", "cx", "cy")}
    return (s["upper"] - s["left"]) - (s["cy"] - s["cx"])
def ci(v):
    b = [np.nanmean(v[k]) for k in BOOT]
    return f"{np.nanmean(v)*100:5.1f} [{np.percentile(b, 2.5)*100:5.1f}…{np.percentile(b, 97.5)*100:5.1f}]"
print(f"Сцен в подмножестве: {len(idx)}. D в п.п., [95% ДИ]")
for g, name in (("cls", "CLS"), ("far_bg", "дальний фон")):
    print(f"=== {name}\nслой   D DINO              D CLIP              D DINO − D CLIP")
    for L in range(1, 13):
        a, b = dvec(A, "acts_v2", g, L), dvec(B, "acts_v2_clip", g, L)
        print(f"L{L:02d}   {ci(a)}   {ci(b)}   {ci(a - b)}")

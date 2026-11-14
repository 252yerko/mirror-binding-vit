"""
photos.py — опыт C: настоящие фото. Добавлено 14.11 (реестр 153–154, ставка Ерназара — до кода).

ВОПРОС ОПЫТА:
  Все прежние опыты — на синтетических сценах с серым фоном. Переносятся ли (C1) зеркальная слепота и
  (C2) разрыв «верхний − левый» на настоящие фото?

ЛОВУШКА, РАДИ КОТОРОЙ ТАКОЙ ДИЗАЙН:
  на обычном фото отражение сверху-вниз даёт неестественную картинку (небо внизу), d(↑↓) раздувается,
  и отношение падает без всякой слепоты. Поэтому главная выборка C1 — аэрофото (виды сверху), где оба
  отражения естественны; обычные фото (COCO) — справка и ставка 154e.

ДАННЫЕ (готовятся командой --prepare на Маке; выбор — до регистрации, по seed 0):
  Аэрофото: UC Merced Land Use (USGS, CC0), Hugging Face torchgeo/ucmerced, ревизия
      9b6b32fa404fe95240f37bcc80bac363568d8437, файл UCMerced_LandUse.zip; 21 класс × 100 снимков 256×256;
      берём по 50 снимков из класса (np.random.default_rng(0), классы и файлы по алфавиту) = 1050.
  Обычные фото: COCO val2017 (http://images.cocodataset.org/zips/val2017.zip), 5000 фото; имена по алфавиту,
      перестановка np.random.default_rng(0): первые 2000 — фоны C2, следующие 1050 — C1 (наборы не пересекаются).
  Все фото: короткая сторона → 224 (бикубически), центральный кроп 224×224, RGB. Списки и md5 архивов —
      в photos_lists.json (md5 этого файла — в регистрации).

C1 — ЗЕРКАЛЬНАЯ СЛЕПОТА НА ФОТО (24 сети и 7 случайных из many_nets.py, без изменений):
  отношение = ср. d(←→) / ср. d(↑↓), косинусное расстояние среднего по патчам (без служебных токенов) на
  последнем слое (L12 у S/B, L24 у L) — та же мера, что в опыте B; отдельно на аэрофото и на обычных фото;
  95% интервал — бутстрэп по картинкам (1000, seed 0).
C2 — ПРИВЯЗКА НА ПРИРОДНОМ ФОНЕ (DINO B/16, CLIP-OpenAI B/16, DeiT III B/16, DINOv2-reg B/14 + по 2 случайные):
  сцены data_v2 (2000), фигуры — из data_v2 по маске (пиксель в пиксель), всё вне маски — фото-фон сцены
  (одно и то же фото для S и Sp, поэтому «мешок признаков» по-прежнему ровно 50%). Обычная проба (StandardScaler +
  логрег C = 0,1, те же 5 фолдов по сценам), дальний фон (зазор ≥ 32 px: 3 патча при патче 16, 4 при патче 14),
  последний слой; разрыв = «верхний» − «левый». Условие «серый фон» (вместо фото — серый 60) повторяет data_v2
  до пикселя и служит проверкой 9 (числа опыта B).

ПРАВИЛА (записаны ДО запуска; точечные значения):
  C1a: ρ Спирмена(отношение на синтетике из many_nets.csv, отношение на аэрофото) по 24 обученным сетям.
       «Слепота переносится на фото», если ρ > 0 и односторонний перестановочный p < 0,05 (10 000, seed 0);
       иначе «не подтверждена».
  C1b: DINO B/16 на аэрофото: верхняя граница отношения < 0,8 → «DINO слеп и на фото»;
       нижняя граница ≥ 0,8 → «DINO не слеп на фото»; иначе «не решено».
  C2:  для каждой из 4 сетей (фото-фон): нижняя граница разрыва > 0 → «разрыв есть на природном фоне»;
       верхняя граница < 0 → «обратный разрыв»; иначе «не показан».
  Справка (не правила): отношение на обычных фото и у случайных сетей; CLS (где есть); разрыв фото − серый
  (парный бутстрэп по сценам); точности «левый» / «верхний».
  Ставки Ерназара (реестр 154, 14.11 19:46): 154a отношение DINO B/16 на аэрофото 0,30–0,55 (округление 0,01);
      154b C1b «DINO слеп и на фото»; 154c C1a «слепота переносится на фото»; 154d ρ в C1a от +0,40 до +0,80
      (округление 0,01); 154e на обычных фото отношение ниже, чем на аэрофото, у ≥ 13 из 24 сетей;
      154f разрыв DINO 5–16 п.п. (округление 0,1); 154g DINO «разрыв есть на природном фоне»;
      154h разрыв CLIP 0–10 п.п. (округление 0,1); 154i CLIP «разрыв есть на природном фоне».

ВСТРОЕННЫЕ ПРОВЕРКИ (все должны пройти):
  1. C2, L00: проба (дальний фон и CLS, где есть) на фото-фоне — ровно 50,0%.
  2. C2: вне маски пиксели S и Sp равны фото; внутри маски — равны data_v2; «серый фон» = data_v2 до пикселя.
  3. Пошаговый прогон = forward_features самой модели (макс. разница < 1e-3).
  4. Пустая серая картинка не меняется при отражениях на L00 и на последнем слое (< 1e-4).
  5. C2: контроль «близнец-вопрос» > 90% на последнем слое (фото-фон) у обученных сетей.
  6. Случайная сеть отличается от обученной (отпечаток весов).
  7. Сетка, служебные токены, блоки — как ожидается (как в many_nets.py).
  8. Данные: 21 класс × 50 аэрофото; 2000 + 1050 фото COCO без пересечений; печатается md5 photos_lists.json
     (должен совпасть с записанным в регистрации).
  9. Прежние числа: (а) C1-код на 631 синтетической сцене даёт отношение опыта B ± 0,01 у DINO B/16, CLIP-OpenAI B/16,
     DeiT III B/16, DINOv2-reg B/14; (б) C2 «серый фон»: «левый» / «верхний» как в опыте B ± 0,5 п.п.
 10. Случайные сети в C2 (seed 0 и 1): |разрыв| < 3 п.п. или 95% интервал накрывает 0 (порог шире, чем в опыте B,
     где проверка 10 не прошла при |разрыв| < 2 п.п.).
 11. Предупреждений «обучение не сошлось» — 0.

ЗАПУСК (из папки binding-vit):
  python photos.py --prepare 2>&1 | tee photos_prepare_log.txt          # скачать и выбрать фото (~1,1 ГБ, 5–15 мин)
  caffeinate -i python photos.py --c1 --part 1 2>&1 | tee photos_c1_part1_log.txt   # ~1 ч; также --part 2, 3
  caffeinate -i python photos.py --c2 2>&1 | tee photos_c2_log.txt                  # ~40–60 мин
  python photos.py --summary 2>&1 | tee photos_summary_log.txt
  -> photos_c1.csv, photos_c2.csv, photos_rules.csv; кэш — ~/binding-vit-data/photos/.
"""

import argparse
import csv
import hashlib
import json
import os
import time
import zipfile

import numpy as np
import torch
from PIL import Image

import flip_test as ft
import probe_big as pb
import mirror_decomp as md
import many_nets as mn
from extract import run_layers, fingerprint, pick_device

DATA = "data_v2"
BG_GRAY = 60
BATCH = 16
ROOT = os.path.expanduser("~/binding-vit-data/photos")
LISTS = "photos_lists.json"
UCM_REPO, UCM_REV, UCM_FILE = "torchgeo/ucmerced", "9b6b32fa404fe95240f37bcc80bac363568d8437", "UCMerced_LandUse.zip"
COCO_URL = "http://images.cocodataset.org/zips/val2017.zip"
N_AER_PER_CLASS, N_C2, N_C1 = 50, 2000, 1050
C2_NETS = ["DINO B/16", "CLIP-OpenAI B/16", "DeiT III B/16", "DINOv2-reg B/14"]
C2_RANDOM_SEEDS = (0, 1)
OUT_C1, OUT_C2 = "photos_c1.csv", "photos_c2.csv"


# ------------------------------------------------------------------
# данные
# ------------------------------------------------------------------
def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def junk(path):
    """Служебные файлы архивов macOS (__MACOSX, ._имя) — не картинки."""
    return "__MACOSX" in path.split(os.sep) or os.path.basename(path).startswith("._")


def prep_photo(path):
    """Короткая сторона → 224 (бикубически), центральный кроп 224×224, RGB, uint8 [224, 224, 3]."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = 224 / min(w, h)
    nw, nh = max(224, round(w * s)), max(224, round(h * s))
    im = im.resize((nw, nh), Image.BICUBIC)
    l, t = (nw - 224) // 2, (nh - 224) // 2
    a = np.asarray(im.crop((l, t, l + 224, t + 224))).copy()
    assert a.shape == (224, 224, 3)
    return a


def prepare(args):
    os.makedirs(ROOT, exist_ok=True)
    # аэрофото
    from huggingface_hub import hf_hub_download
    zp = hf_hub_download(repo_id=UCM_REPO, repo_type="dataset", filename=UCM_FILE, revision=UCM_REV, local_dir=os.path.join(ROOT, "ucm_zip"))
    ucm_md5 = md5(zp)
    ucm_dir = os.path.join(ROOT, "ucm")
    if not os.path.isdir(ucm_dir):
        with zipfile.ZipFile(zp) as z:
            z.extractall(ucm_dir)
    imgs = [os.path.relpath(os.path.join(d, f), ucm_dir) for d, _, fs in os.walk(ucm_dir) for f in fs
            if f.lower().endswith((".tif", ".tiff", ".png", ".jpg")) and not junk(os.path.join(d, f))]
    classes = sorted({os.path.basename(os.path.dirname(p)) for p in imgs})
    rng = np.random.default_rng(0)
    aerial = []
    for c in classes:
        fs = sorted(p for p in imgs if os.path.basename(os.path.dirname(p)) == c)
        aerial += [fs[i] for i in sorted(rng.choice(len(fs), N_AER_PER_CLASS, replace=False))]
    # обычные фото
    cz = os.path.join(ROOT, "val2017.zip")
    if not os.path.exists(cz):
        import urllib.request
        print(f"Скачиваю {COCO_URL} (≈ 0,8 ГБ)…", flush=True)
        urllib.request.urlretrieve(COCO_URL, cz + ".part")
        os.replace(cz + ".part", cz)
    coco_md5 = md5(cz)
    coco_dir = os.path.join(ROOT, "coco")
    if not os.path.isdir(coco_dir):
        with zipfile.ZipFile(cz) as z:
            z.extractall(coco_dir)
    cfiles = sorted(os.path.relpath(os.path.join(d, f), coco_dir) for d, _, fs in os.walk(coco_dir) for f in fs
                    if f.lower().endswith(".jpg") and not junk(os.path.join(d, f)))
    perm = np.random.default_rng(0).permutation(len(cfiles))
    coco_c2 = [cfiles[i] for i in perm[:N_C2]]
    coco_c1 = [cfiles[i] for i in perm[N_C2:N_C2 + N_C1]]
    lists = dict(ucm_repo=UCM_REPO, ucm_revision=UCM_REV, ucm_zip_md5=ucm_md5, ucm_classes=classes, ucm_n_files=len(imgs),
                 coco_url=COCO_URL, coco_zip_md5=coco_md5, coco_n_files=len(cfiles),
                 aerial=aerial, coco_c1=coco_c1, coco_c2=coco_c2)
    with open(LISTS, "w") as fh:
        json.dump(lists, fh, indent=1, ensure_ascii=False)
    sizes = {}
    for p in aerial:
        sz = Image.open(os.path.join(ucm_dir, p)).size
        sizes[sz] = sizes.get(sz, 0) + 1
    print(f"UC Merced: файлов {len(imgs)}, классов {len(classes)}; md5 архива {ucm_md5}; размеры выбранных: {sizes}")
    print(f"COCO val2017: {len(cfiles)} фото; md5 архива {coco_md5}")
    print(f"Выбрано: аэрофото {len(aerial)}, COCO для C1 {len(coco_c1)}, COCO-фоны для C2 {len(coco_c2)}")
    print(f"{LISTS}: md5 {md5(LISTS)}")
    preview(lists)


LIST_PATH = [LISTS]


def load_lists(args):
    LIST_PATH[0] = args.lists
    L = json.load(open(args.lists))
    return L, args.ucm_dir or os.path.join(ROOT, "ucm"), args.coco_dir or os.path.join(ROOT, "coco")


def check_data(L, ok):
    cls = {}
    for p in L["aerial"]:
        c = os.path.basename(os.path.dirname(p))
        cls[c] = cls.get(c, 0) + 1
    good = (len(cls) == 21 and all(v == N_AER_PER_CLASS for v in cls.values()) and len(set(L["aerial"])) == len(L["aerial"])
            and len(L["coco_c2"]) == N_C2 and len(L["coco_c1"]) == N_C1 and not set(L["coco_c2"]) & set(L["coco_c1"])
            and len(set(L["coco_c2"])) == N_C2 and len(set(L["coco_c1"])) == N_C1)
    ok["8"] &= good
    print(f"  Проверка 8 (данные): классов {len(cls)}, по {sorted(set(cls.values()))} снимков; COCO {len(L['coco_c2'])} + "
          f"{len(L['coco_c1'])}, пересечений {len(set(L['coco_c2']) & set(L['coco_c1']))}  {'✅' if good else '⚠️ проверь весь вывод'};  "
          f"md5 {LISTS} = {md5(LIST_PATH[0])}")


def composite(rec, version, photo):
    """Фигуры data_v2 (по маске) поверх фото; photo = None → серый фон BG_GRAY (должно совпасть с data_v2)."""
    base = np.asarray(Image.open(f"{DATA}/images/{rec['id']}_{version}.png").convert("RGB"))
    mask = np.asarray(Image.open(f"{DATA}/masks/{rec['id']}.png"))
    bg = np.full_like(base, BG_GRAY) if photo is None else photo
    return np.where(mask[..., None] > 0, base, bg).astype(np.uint8), base, mask


def preview(L):
    ucm_dir, coco_dir = os.path.join(ROOT, "ucm"), os.path.join(ROOT, "coco")
    meta = [json.loads(line) for line in open(f"{DATA}/meta.jsonl")][:6]
    rows = []
    for i, rec in enumerate(meta):
        ph = prep_photo(os.path.join(coco_dir, L["coco_c2"][i]))
        rows.append(np.concatenate([composite(rec, v, ph)[0] for v in ("S", "Sp")], 1))
    a = [prep_photo(os.path.join(ucm_dir, p)) for p in L["aerial"][::175][:6]]
    rows2 = [np.concatenate([x, x[:, ::-1], x[::-1]], 1) for x in a]
    Image.fromarray(np.concatenate(rows, 0)).save("photos_c2_preview.png")
    Image.fromarray(np.concatenate(rows2, 0)).save("photos_c1_preview.png")
    print("Превью: photos_c2_preview.png (S | Sp на фото), photos_c1_preview.png (аэрофото | ←→ | ↑↓)")


def to_tensor(a, mean, std):
    x = torch.from_numpy(a).permute(2, 0, 1).float() / 255.0
    return (x - mean) / std


# ------------------------------------------------------------------
# общие проверки сети (те же, что в many_nets.run_net)
# ------------------------------------------------------------------
def net_checks(model, nb, mean, std, device, ok):
    meta0 = json.loads(open(f"{DATA}/meta.jsonl").readline())
    x2 = torch.stack([to_tensor(np.asarray(Image.open(f"{DATA}/images/{meta0['id']}_{v}.png").convert("RGB")), mean, std)
                      for v in ("S", "Sp")]).to(device)
    with torch.no_grad():
        ref = model.forward_features(x2)
        mine = model.norm(run_layers(model, x2)[nb])
    d3 = float((mine - ref).abs().max())
    ok["3"] &= d3 < 1e-3
    blank = x2[:1].clone()
    blank[:] = blank[0, :, :1, :1]
    with torch.no_grad():
        fb = [run_layers(model, b) for b in (blank, blank.flip(3), blank.flip(2))]
    z = max(float((fb[0][l] - fb[k][l]).abs().max()) for l in (0, nb) for k in (1, 2))
    ok["4"] &= z < 1e-4
    print(f"  Проверка 3: макс. разница {d3:.2e} {'✅' if d3 < 1e-3 else '⚠️'};  Проверка 4: {z:.2e} {'✅' if z < 1e-4 else '⚠️'}")


def open_net(name, timm_name, P, seed, device, ok):
    model, weights = mn.load(timm_name, seed)
    fp = fingerprint(model)
    tag = name if seed is None else f"{name} [случайная, seed {seed}]"
    print(f"\n### {tag}: {weights}   отпечаток весов {fp:.4f}", flush=True)
    ok8, G, n_pre, nb, mean, std = mn.setup(model, P)
    ok["7"] &= ok8
    model.to(device)
    net_checks(model, nb, mean, std, device, ok)
    return model, weights, fp, G, n_pre, nb, mean, std


# ------------------------------------------------------------------
# C1
# ------------------------------------------------------------------
@torch.no_grad()
def patch_means(model, arrays, device, n_pre, nb, mean, std):
    """arrays — список uint8 [224,224,3]. Возвращает PM[n, 3 (o, ←→, ↑↓), D] и CLS[n, 3, D] (последний слой, без norm)."""
    D = model.embed_dim
    PM = np.zeros((len(arrays), 3, D), dtype=np.float32)
    CL = np.zeros((len(arrays), 3, D), dtype=np.float32)
    jobs = [(i, k) for i in range(len(arrays)) for k in range(3)]
    t0 = time.time()
    for s in range(0, len(jobs), BATCH):
        part = jobs[s:s + BATCH]
        xs = []
        for i, k in part:
            x = to_tensor(arrays[i], mean, std)
            xs.append(x if k == 0 else (x.flip(2) if k == 1 else x.flip(1)))
        h = run_layers(model, torch.stack(xs).to(device))[nb].float().cpu().numpy()
        for b, (i, k) in enumerate(part):
            PM[i, k] = h[b, n_pre:].mean(0)
            if n_pre > 0:
                CL[i, k] = h[b, 0]
        if (s // BATCH) % 100 == 0:
            print(f"    картинок {min(s + BATCH, len(jobs))}/{len(jobs)}  ({time.time() - t0:.0f} с)", flush=True)
    return PM, CL


def ratio_of(PM):
    dh, dv = ft.cos_dist(PM[:, 0], PM[:, 1]), ft.cos_dist(PM[:, 0], PM[:, 2])
    return ft.boot_ratio(dh, dv, np.random.default_rng(0))


def run_c1(args):
    L, ucm_dir, coco_dir = load_lists(args)
    ok = {k: True for k in ("3", "4", "7", "8")}
    ok["6"], ok["9"] = True, True
    check_data(L, ok)
    lim = args.limit or None
    sets = {"aerial": [prep_photo(os.path.join(ucm_dir, p)) for p in L["aerial"][:lim]],
            "coco": [prep_photo(os.path.join(coco_dir, p)) for p in L["coco_c1"][:lim]]}
    meta = [json.loads(line) for line in open(f"{DATA}/meta.jsonl")]
    sym = [r for r in meta if all(o["shape"] in ft.SYM_SHAPES for o in r["objects"])][:lim]
    synth = [np.asarray(Image.open(f"{DATA}/images/{r['id']}_S.png").convert("RGB")) for r in sym]
    os.makedirs(args.out, exist_ok=True)
    device = pick_device()
    import timm
    print(f"Опыт C, C1, часть {args.part}   timm {timm.__version__}, torch {torch.__version__}; "
          f"аэрофото {len(sets['aerial'])}, COCO {len(sets['coco'])}")
    if args.limit:
        print("⚠️ ПРОВЕРОЧНЫЙ прогон (облако): числа не для статьи.")
    rows, fps = [], {}
    sel = [n for n in mn.NETS if n[5] == args.part and (not args.only or n[0] in args.only)]
    for name, tn, fam, size, P, _ in sel:
        seeds = ([] if args.random_only else [99 if args.test_random_main else None]) + ([0] if name in mn.RANDOM else [])
        for seed in seeds:
            t0 = time.time()
            model, weights, fp, G, n_pre, nb, mean, std = open_net(name, tn, P, seed, device, ok)
            if seed in (None, 99):
                fps[name] = fp
            elif name in fps:
                ok["6"] &= abs(fp - fps[name]) > 1e-3
            safe = name.replace("/", "_").replace(" ", "_") + ("" if seed is None else f"_rand{seed}") + (f"_test{args.limit}" if args.limit else "")
            cache = os.path.join(args.out, f"c1_{safe}.npz")
            if os.path.exists(cache):
                zz = np.load(cache)
                res = {k: zz[k].astype(np.float32) for k in zz.files}
                print(f"  взято из {cache}")
            else:
                res = {}
                for sname, arr in sets.items():
                    res[f"PM_{sname}"], res[f"CL_{sname}"] = patch_means(model, arr, device, n_pre, nb, mean, std)
                if name in mn.OLD and seed in (None, 99):
                    res["PM_synth"], res["CL_synth"] = patch_means(model, synth, device, n_pre, nb, mean, std)
                np.savez(cache, **{k: v.astype(np.float16) for k, v in res.items()})
                res = {k: v.astype(np.float16).astype(np.float32) for k, v in res.items()}
            del model
            if device.type == "mps":
                torch.mps.empty_cache()
            for sname in ("aerial", "coco"):
                r = ratio_of(res[f"PM_{sname}"])
                c = ratio_of(res[f"CL_{sname}"]) if n_pre > 0 else (float("nan"),) * 3
                rows.append(dict(name=name, family=fam, size=size, P=P, seed="" if seed is None else seed, weights=weights, fp=fp,
                                 set=sname, n=len(sets[sname]), ratio=r[0], ratio_lo=r[1], ratio_hi=r[2], cls_ratio=c[0], n_pre=n_pre, blocks=nb))
                print(f"  {sname:6s}: отношение {r[0]:.3f} [{r[1]:.3f}–{r[2]:.3f}]" + (f"; CLS {c[0]:.3f}" if n_pre > 0 else ""))
            if "PM_synth" in res:
                rs = ratio_of(res["PM_synth"])
                good = abs(rs[0] - mn.OLD[name][0]) <= 0.01
                if not args.limit:
                    ok["9"] &= good
                print(f"  Проверка 9а (синтетика, {len(synth)} сцен): отношение {rs[0]:.3f} / в опыте B {mn.OLD[name][0]:.3f}  "
                      f"{'✅' if good else ('(облако — не сравнивается)' if args.limit else '⚠️ проверь весь вывод')}")
            print(f"  ({time.time() - t0:.0f} с)", flush=True)
    out = OUT_C1 if not args.limit else f"photos_c1_test{args.limit}.csv"
    write_rows(out, rows, ["name", "family", "size", "P", "seed", "weights", "fp", "set", "n", "ratio", "ratio_lo", "ratio_hi",
                           "cls_ratio", "n_pre", "blocks"])
    for k, t in (("3", "прогон = forward_features"), ("4", "пустая картинка"), ("6", "случайные отличаются от обученных"),
                 ("7", "сетка / токены / блоки"), ("8", "данные"), ("9", "9а: отношение на синтетике = опыт B")):
        print(f"Проверка {k} ({t}): {'✅' if ok[k] else '⚠️ проверь весь вывод'}")
    print(f"\nЧисла дописаны в {out}")


def write_rows(out, rows, fields):
    exists = os.path.exists(out)
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)


# ------------------------------------------------------------------
# C2
# ------------------------------------------------------------------
@torch.no_grad()
def c2_extract(model, recs, infos, photos, device, n_pre, nb, mean, std, ok):
    """F[2 слоя (L00, последний), сцены, 2 версии, 2 группы (cls, far_bg), D]; T[сцены, 2 версии, 2 близнеца, D]."""
    D = model.embed_dim
    F = np.zeros((2, len(recs), 2, 2, D), dtype=np.float32)
    T = np.zeros((len(recs), 2, 2, D), dtype=np.float32)
    jobs = [(si, vi) for si in range(len(recs)) for vi in range(2)]
    t0 = time.time()
    for s in range(0, len(jobs), BATCH):
        part = jobs[s:s + BATCH]
        xs = []
        for si, vi in part:
            a, base, mask = composite(recs[si], ("S", "Sp")[vi], photos[si])
            # проверка 2
            out_m = mask == 0
            good = np.array_equal(a[mask > 0], base[mask > 0]) and (
                np.array_equal(a[out_m], base[out_m]) if photos[si] is None else np.array_equal(a[out_m], photos[si][out_m]))
            if photos[si] is None:
                good &= np.array_equal(a, base)
            ok["2"] &= bool(good)
            xs.append(to_tensor(a, mean, std))
        outs = run_layers(model, torch.stack(xs).to(device))
        hs = [outs[0].float().cpu().numpy(), outs[nb].float().cpu().numpy()]
        for b, (si, vi) in enumerate(part):
            g = infos[si][0]["o"]
            idx = [int(q) + n_pre for q in g["far_bg"]]
            for li, h in enumerate(hs):
                if n_pre > 0:
                    F[li, si, vi, 0] = h[b, 0]
                F[li, si, vi, 1] = h[b, idx].mean(0)
            for t in range(2):
                T[si, vi, t] = hs[1][b, [int(q) + n_pre for q in g[f"twin{t}"]]].mean(0)
        if (s // BATCH) % 100 == 0:
            print(f"    картинок {min(s + BATCH, len(jobs))}/{len(jobs)}  ({time.time() - t0:.0f} с)", flush=True)
    return F, T


def c2_far_pixels_check(recs, infos, photos, P, G, ok):
    """Проверка 2 (часть): пиксели патчей дальнего фона одинаковы в S и Sp на фото-фоне."""
    for si, rec in enumerate(recs):
        S = composite(rec, "S", photos[si])[0]
        Sp = composite(rec, "Sp", photos[si])[0]
        for q in infos[si][0]["o"]["far_bg"]:
            py, px = divmod(int(q), G)
            if not np.array_equal(S[py * P:(py + 1) * P, px * P:(px + 1) * P], Sp[py * P:(py + 1) * P, px * P:(px + 1) * P]):
                ok["2"] = False
                return


INFO = {}


def run_c2(args):
    L, ucm_dir, coco_dir = load_lists(args)
    meta = [json.loads(line) for line in open(f"{DATA}/meta.jsonl")]
    if args.limit:
        meta = meta[:args.limit]
    colors = sorted({ob[k] for r in meta for ob in r["objects"] for k in ("color", "color_Sp")})
    folds_all = np.random.default_rng(0).permutation(len(meta)) % md.N_FOLDS
    photos = [prep_photo(os.path.join(coco_dir, p)) for p in L["coco_c2"][:len(meta)]]
    ok = {k: True for k in ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10")}
    ok["warn"] = 0
    check_data(L, ok)
    os.makedirs(args.out, exist_ok=True)
    device = pick_device()
    import timm
    print(f"Опыт C, C2   timm {timm.__version__}, torch {torch.__version__}; сцен {len(meta)}")
    if args.limit:
        print("⚠️ ПРОВЕРОЧНЫЙ прогон (облако): числа не для статьи.")
    nets = {n[0]: n for n in mn.NETS}
    old = {r["name"]: r for r in csv.DictReader(open(mn.OUT_CSV, encoding="utf-8")) if r["seed"] == ""} if os.path.exists(mn.OUT_CSV) else {}
    rows, fps = [], {}
    sel = [n for n in C2_NETS if not args.only or n in args.only]
    for name in sel:
        _, tn, fam, size, P, _ = nets[name]
        seeds = ([] if args.random_only else [99 if args.test_random_main else None]) + list(C2_RANDOM_SEEDS)
        for seed in seeds:
            t0 = time.time()
            model, weights, fp, G, n_pre, nb, mean, std = open_net(name, tn, P, seed, device, ok)
            if seed in (None, 99):
                fps[name] = fp
            elif name in fps:
                ok["6"] &= abs(fp - fps[name]) > 1e-3
            pb.FAR = 3 if P == 16 else 4
            if P not in INFO:
                INFO[P] = [md.scene_info(r, colors, P, G) for r in meta]
                c2_far_pixels_check(meta, INFO[P], photos, P, G, ok)
            infos = INFO[P]
            trained = seed in (None, 99)
            conds = ["photo", "gray"] if trained else ["photo"]
            safe = name.replace("/", "_").replace(" ", "_") + ("" if seed is None else f"_rand{seed}") + (f"_test{args.limit}" if args.limit else "")
            res = {}
            for cond in conds:
                cache = os.path.join(args.out, f"c2_{safe}_{cond}.npz")
                if os.path.exists(cache):
                    zz = np.load(cache)
                    Fc, Tc = zz["F"].astype(np.float32), zz["T"].astype(np.float32)
                    print(f"  взято из {cache}")
                else:
                    ph = photos if cond == "photo" else [None] * len(meta)
                    Fc, Tc = c2_extract(model, meta, infos, ph, device, n_pre, nb, mean, std, ok)
                    np.savez(cache, F=Fc.astype(np.float16), T=Tc.astype(np.float16))
                    Fc, Tc = Fc.astype(np.float16).astype(np.float32), Tc.astype(np.float16).astype(np.float32)
                res[cond] = (Fc, Tc)
            del model
            if device.type == "mps":
                torch.mps.empty_cache()
            lab = {q: np.array([[infos[i][1][q][1][v] for v in ("S", "Sp")] for i in range(len(meta))]) for q in ("left", "upper")}
            out = {}
            for cond, (Fc, Tc) in res.items():
                acc, accc = {}, {}
                for q in ("left", "upper"):
                    acc[q], nw = md.probe_plain(Fc[1, :, :, 1], lab[q][:, :, 0], lab[q][:, :, 1], folds_all)
                    ok["warn"] += nw
                    if n_pre > 0:
                        accc[q], nw = md.probe_plain(Fc[1, :, :, 0], lab[q][:, :, 0], lab[q][:, :, 1], folds_all)
                        ok["warn"] += nw
                    if cond == "photo":
                        for gi in ([1] + ([0] if n_pre > 0 else [])):
                            a0, nw = md.probe_plain(Fc[0, :, :, gi], lab[q][:, :, 0], lab[q][:, :, 1], folds_all)
                            ok["warn"] += nw
                            ok["1"] &= bool(np.allclose(a0, 0.5))
                        if trained:
                            tq = np.array([infos[i][1][q][0] for i in range(len(meta))])
                            c_, _ = md.probe_plain(Tc[np.arange(len(meta)), :, tq], lab[q][:, :, 0], lab[q][:, :, 1], folds_all)
                            ok["5"] &= np.nanmean(c_) > 0.9
                out[cond] = (acc, accc, acc["upper"] - acc["left"])
            gp = md.boot(out["photo"][2], np.random.default_rng(0))
            row = dict(name=name, family=fam, seed="" if seed is None else seed, weights=weights, fp=fp, P=P, n=len(meta),
                       left=float(np.nanmean(out["photo"][0]["left"])), upper=float(np.nanmean(out["photo"][0]["upper"])),
                       gap=gp[0], gap_lo=gp[1], gap_hi=gp[2], cls_gap="", gray_left="", gray_upper="", gray_gap="", diff="", diff_lo="", diff_hi="")
            if n_pre > 0:
                cg = md.boot(out["photo"][1]["upper"] - out["photo"][1]["left"], np.random.default_rng(0))
                row["cls_gap"] = cg[0]
            msg = (f"  фото-фон: «левый» {row['left'] * 100:.2f}%, «верхний» {row['upper'] * 100:.2f}%; "
                   f"разрыв {gp[0] * 100:.1f} [{gp[1] * 100:.1f} … {gp[2] * 100:.1f}] п.п." + (f"; CLS {row['cls_gap'] * 100:.1f}" if n_pre > 0 else ""))
            if "gray" in out:
                gg = md.boot(out["gray"][2], np.random.default_rng(0))
                dd = md.boot(out["photo"][2] - out["gray"][2], np.random.default_rng(0))
                row.update(gray_left=float(np.nanmean(out["gray"][0]["left"])), gray_upper=float(np.nanmean(out["gray"][0]["upper"])),
                           gray_gap=gg[0], diff=dd[0], diff_lo=dd[1], diff_hi=dd[2])
                msg += (f"\n  серый фон: «левый» {row['gray_left'] * 100:.2f}%, «верхний» {row['gray_upper'] * 100:.2f}%; разрыв {gg[0] * 100:.1f} п.п.; "
                        f"фото − серый {dd[0] * 100:+.1f} [{dd[1] * 100:+.1f} … {dd[2] * 100:+.1f}] п.п.")
                if name in old and not args.limit:
                    o = old[name]
                    good = abs(row["gray_left"] - float(o["left"])) * 100 <= 0.5 and abs(row["gray_upper"] - float(o["upper"])) * 100 <= 0.5
                    ok["9"] &= good
                    msg += (f"\n  Проверка 9б: «левый» {row['gray_left'] * 100:.2f} / в опыте B {float(o['left']) * 100:.2f}; «верхний» "
                            f"{row['gray_upper'] * 100:.2f} / {float(o['upper']) * 100:.2f}  {'✅' if good else '⚠️ проверь весь вывод'}")
            if not trained:
                good10 = abs(gp[0]) < 0.03 or gp[1] <= 0 <= gp[2]
                ok["10"] &= good10
                msg += f"\n  Проверка 10 (случайная): {'✅' if good10 else '⚠️ проверь весь вывод'}"
            print(msg + f"   ({time.time() - t0:.0f} с)", flush=True)
            rows.append(row)
    out_csv = OUT_C2 if not args.limit else f"photos_c2_test{args.limit}.csv"
    write_rows(out_csv, rows, ["name", "family", "seed", "weights", "fp", "P", "n", "left", "upper", "gap", "gap_lo", "gap_hi", "cls_gap",
                               "gray_left", "gray_upper", "gray_gap", "diff", "diff_lo", "diff_hi"])
    for k, t in (("1", "L00 ровно 50,0%"), ("2", "картинки: фото вне маски, data_v2 внутри, серый = data_v2, дальний фон S = Sp"),
                 ("3", "прогон = forward_features"), ("4", "пустая картинка"), ("5", "контроль > 90%"),
                 ("6", "случайные отличаются от обученных"), ("7", "сетка / токены / блоки"), ("8", "данные"),
                 ("9", "9б: серый фон = опыт B"), ("10", "случайные: |разрыв| < 3 п.п. или интервал накрывает 0")):
        print(f"Проверка {k} ({t}): {'✅' if ok[k] else '⚠️ проверь весь вывод'}")
    print(f"Проверка 11 (предупреждений «не сошлось»: {ok['warn']}): {'✅' if ok['warn'] == 0 else '⚠️ проверь весь вывод'}")
    print(f"\nЧисла дописаны в {out_csv}")


# ------------------------------------------------------------------
# сводка и правила
# ------------------------------------------------------------------
def rules(c1, c2, synth):
    """c1: {имя: {set: (ratio, lo, hi)}} по обученным; c2: {имя: (gap, lo, hi)}; synth: {имя: отношение в опыте B}."""
    names = [n for n in synth if n in c1 and "aerial" in c1[n]]
    x = np.array([synth[n] for n in names])
    y = np.array([c1[n]["aerial"][0] for n in names])
    rho, p = mn.perm_p(x, y, +1) if len(names) >= 4 else (float("nan"), float("nan"))
    v1a = "слепота переносится на фото" if rho > 0 and p < 0.05 else "не подтверждена"
    d = c1.get("DINO B/16", {}).get("aerial")
    if d is None:
        v1b = "нет данных"
    else:
        v1b = "DINO слеп и на фото" if d[2] < 0.8 else ("DINO не слеп на фото" if d[1] >= 0.8 else "не решено")
    lower = sum(1 for n in names if "coco" in c1[n] and c1[n]["coco"][0] < c1[n]["aerial"][0])
    v2 = {}
    for n, (g, lo, hi) in c2.items():
        v2[n] = "разрыв есть на природном фоне" if lo > 0 else ("обратный разрыв" if hi < 0 else "не показан")
    return dict(names=names, rho=rho, p=p, v1a=v1a, v1b=v1b, dino=d, lower=lower, v2=v2)


def summary(args):
    f1 = OUT_C1 if not args.limit else f"photos_c1_test{args.limit}.csv"
    f2 = OUT_C2 if not args.limit else f"photos_c2_test{args.limit}.csv"
    main_seed = "" if not args.test_random_main else "99"
    c1, rnd = {}, []
    for r in csv.DictReader(open(f1, encoding="utf-8")):
        if r["seed"] == main_seed:
            c1.setdefault(r["name"], {})[r["set"]] = (float(r["ratio"]), float(r["ratio_lo"]), float(r["ratio_hi"]))
        else:
            rnd.append(r)
    c2, c2r, c2rows = {}, [], {}
    if os.path.exists(f2):
        for r in csv.DictReader(open(f2, encoding="utf-8")):
            if r["seed"] == main_seed:
                c2[r["name"]] = (float(r["gap"]), float(r["gap_lo"]), float(r["gap_hi"]))
                c2rows[r["name"]] = r
            else:
                c2r.append(r)
    synth = {r["name"]: float(r["ratio"]) for r in csv.DictReader(open(args.synth_csv, encoding="utf-8")) if r["seed"] == ""}
    synth = {n: v for n, v in synth.items() if n in {x[0] for x in mn.NETS}}
    R = rules(c1, c2, synth)
    print(f"Сетей в C1: {len(R['names'])} из {len(mn.NETS)}")
    for n in sorted(R["names"], key=lambda n: synth[n]):
        a, c = c1[n]["aerial"], c1[n].get("coco", (float("nan"),) * 3)
        print(f"  {n:18s} синтетика {synth[n]:.3f}   аэрофото {a[0]:.3f} [{a[1]:.3f}–{a[2]:.3f}]   COCO {c[0]:.3f} [{c[1]:.3f}–{c[2]:.3f}]")
    if rnd:
        print("Справка: случайные сети (асимметрия самих картинок):")
        for r in rnd:
            print(f"  {r['name']:18s} seed {r['seed']}  {r['set']:6s} отношение {float(r['ratio']):.3f} [{float(r['ratio_lo']):.3f}–{float(r['ratio_hi']):.3f}]")
    print(f"\nВЫВОД ПО ПРАВИЛАМ (реестр 153–154, записано до запуска):")
    print(f"  C1a: ρ(синтетика, аэрофото) = {R['rho']:+.3f}, p = {R['p']:.4f} (n = {len(R['names'])}) -> «{R['v1a']}»")
    if R["dino"] is not None:
        print(f"  C1b: DINO B/16 на аэрофото {R['dino'][0]:.3f} [{R['dino'][1]:.3f}–{R['dino'][2]:.3f}] -> «{R['v1b']}»")
    for n, v in R["v2"].items():
        g = c2[n]
        extra = c2rows[n]
        dif = f"; фото − серый {float(extra['diff']) * 100:+.1f} [{float(extra['diff_lo']) * 100:+.1f} … {float(extra['diff_hi']) * 100:+.1f}]" if extra["diff"] not in ("", None) else ""
        r1 = lambda v: round(v * 100, 1) + 0.0
        print(f"  C2 {n:18s}: разрыв {r1(g[0]):.1f} [{r1(g[1]):.1f} … {r1(g[2]):.1f}] п.п.{dif} -> «{v}»")
    for r in c2r:
        print(f"  справка, случайная {r['name']} seed {r['seed']}: разрыв {float(r['gap']) * 100:.1f} [{float(r['gap_lo']) * 100:.1f} … {float(r['gap_hi']) * 100:.1f}] п.п.")
    print(f"Справка: на COCO отношение ниже, чем на аэрофото, у {R['lower']} из {len(R['names'])} сетей")
    if not args.limit:
        hit = lambda b: "ПОПАЛ ✅" if b else "НЕ ПОПАЛ ❌"
        g = lambda n: round(c2.get(n, (float("nan"),) * 3)[0] * 100, 1) + 0.0
        print("\nСТАВКИ ЕРНАЗАРА (реестр 154, 14.11 19:46):")
        print(f"  154a отношение DINO B/16 на аэрофото 0,30–0,55 -> {R['dino'][0]:.2f}  {hit(0.30 <= round(R['dino'][0], 2) <= 0.55)}")
        print(f"  154b C1b «DINO слеп и на фото» -> «{R['v1b']}»  {hit(R['v1b'] == 'DINO слеп и на фото')}")
        print(f"  154c C1a «слепота переносится на фото» -> «{R['v1a']}»  {hit(R['v1a'] == 'слепота переносится на фото')}")
        print(f"  154d ρ от +0,40 до +0,80 -> {R['rho']:+.2f}  {hit(0.40 <= round(R['rho'], 2) <= 0.80)}")
        print(f"  154e на COCO ниже, чем на аэрофото, у ≥ 13 из 24 -> {R['lower']}  {hit(R['lower'] >= 13)}")
        print(f"  154f разрыв DINO 5–16 п.п. -> {g('DINO B/16'):.1f}  {hit(5.0 <= round(g('DINO B/16'), 1) <= 16.0)}")
        print(f"  154g DINO «разрыв есть на природном фоне» -> «{R['v2'].get('DINO B/16')}»  {hit(R['v2'].get('DINO B/16') == 'разрыв есть на природном фоне')}")
        print(f"  154h разрыв CLIP 0–10 п.п. -> {g('CLIP-OpenAI B/16'):.1f}  {hit(0.0 <= round(g('CLIP-OpenAI B/16'), 1) <= 10.0)}")
        print(f"  154i CLIP «разрыв есть на природном фоне» -> «{R['v2'].get('CLIP-OpenAI B/16')}»  "
              f"{hit(R['v2'].get('CLIP-OpenAI B/16') == 'разрыв есть на природном фоне')}")
    with open("photos_rules.csv" if not args.limit else f"photos_rules_test{args.limit}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rule", "value", "lo", "hi", "p", "n", "verdict"])
        w.writerow(["C1a", R["rho"], "", "", R["p"], len(R["names"]), R["v1a"]])
        if R["dino"] is not None:
            w.writerow(["C1b", R["dino"][0], R["dino"][1], R["dino"][2], "", "", R["v1b"]])
        for n, v in R["v2"].items():
            w.writerow([f"C2 {n}", c2[n][0], c2[n][1], c2[n][2], "", "", v])
        w.writerow(["coco_lower_than_aerial", R["lower"], "", "", "", len(R["names"]), ""])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--c1", action="store_true")
    ap.add_argument("--part", type=int, choices=[1, 2, 3])
    ap.add_argument("--c2", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "cache"))
    ap.add_argument("--lists", default=LISTS)
    ap.add_argument("--ucm-dir", default=None, help="только для проверки в облаке")
    ap.add_argument("--coco-dir", default=None, help="только для проверки в облаке")
    ap.add_argument("--synth-csv", default=mn.OUT_CSV)
    ap.add_argument("--limit", type=int, default=0, help="только для проверки в облаке")
    ap.add_argument("--test-random-main", action="store_true", help="только для проверки в облаке: вместо обученной — случайная seed 99")
    ap.add_argument("--only", nargs="*", default=None, help="только для проверки: имена сетей")
    ap.add_argument("--random-only", action="store_true", help="только для проверки")
    args = ap.parse_args()
    if args.prepare:
        prepare(args)
    elif args.c1:
        assert args.part, "нужно --part 1|2|3"
        run_c1(args)
    elif args.c2:
        run_c2(args)
    elif args.summary:
        summary(args)
    else:
        ap.error("нужно --prepare, --c1 --part N, --c2 или --summary")


if __name__ == "__main__":
    main()

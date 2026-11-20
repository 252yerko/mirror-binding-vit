"""
e_prep_kaggle.py — опыт E: однократная подготовка ImageNet-100 (список CMC) для обучения на Kaggle. Добавлено 19.11 (реестр 282).

ЗАЧЕМ: замер 19.11 (реестр 281) — чтение файлов набора на Kaggle 14 файлов/с на поток (44 карт./с на 4 процесса), а одно
  обучение E — 5,07 млн картинок. Поэтому все фото один раз читаются параллельно, уменьшаются и складываются в один большой файл
  на часть (train / val) + индекс. Сессия обучения читает большие файлы, а не 132 тыс. мелких.
ЧТО ДЕЛАЕТ С КАРТИНКАМИ: если короткая сторона больше 256 — уменьшение до 256 по короткой стороне (пропорции сохраняются;
  PIL, BICUBIC с антиалиасингом), затем JPEG качества 90; если короткая сторона ≤ 256 — исходные байты файла без изменений.
  Обрезки, отражений, смены цвета нет. Порядок: классы по возрастанию wnid, внутри класса — файлы по возрастанию имени.
ВЫХОД (--out, по умолчанию /kaggle/working/in100pack): train.bin, val.bin (склеенные байты JPEG), index.npz (для каждой
  части: смещения, длины, метки, исходные имена файлов, размеры до и после), classes.txt, MANIFEST.txt (md5 всех файлов),
  prep_log.txt (этот вывод).
ЗАПУСК (блокнот Kaggle БЕЗ GPU — квота GPU не тратится; данные t3s1addd/imagenet-100 добавлены как Input):
  python -u e_prep_kaggle.py --data /kaggle/input/datasets/t3s1addd/imagenet-100 2>&1 | tee /kaggle/working/prep_log.txt
"""
import argparse
import hashlib
import io
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

CMC100 = (
    "n02869837 n01749939 n02488291 n02107142 n13037406 n02091831 n04517823 n04589890 n03062245 n01773797 "
    "n01735189 n07831146 n07753275 n03085013 n04485082 n02105505 n01983481 n02788148 n03530642 n04435653 "
    "n02086910 n02859443 n13040303 n03594734 n02085620 n02099849 n01558993 n04493381 n02109047 n04111531 "
    "n02877765 n04429376 n02009229 n01978455 n02106550 n01820546 n01692333 n07714571 n02974003 n02114855 "
    "n03785016 n03764736 n03775546 n02087046 n07836838 n04099969 n04592741 n03891251 n02701002 n03379051 "
    "n02259212 n07715103 n03947888 n04026417 n02326432 n03637318 n01980166 n02113799 n02086240 n03903868 "
    "n02483362 n04127249 n02089973 n03017168 n02093428 n02804414 n02396427 n04418357 n02172182 n01729322 "
    "n02113978 n03787032 n02089867 n02119022 n03777754 n04238763 n02231487 n03032252 n02138441 n02104029 "
    "n03837869 n03494278 n04136333 n03794056 n03492542 n02018207 n04067472 n03930630 n03584829 n02123045 "
    "n04229816 n02100583 n03642806 n04336792 n03259280 n02116738 n02108089 n03424325 n01855672 n02090622"
).split()
assert len(CMC100) == 100 and len(set(CMC100)) == 100
WNID = re.compile(r"n\d{8}$")
EXT = (".jpeg", ".jpg", ".png")
SHORT = 256
QUALITY = 90


def scan(root, split):
    found = {}
    for d, _, files in os.walk(root, followlinks=True):
        name = os.path.basename(d)
        if not WNID.match(name):
            continue
        parts = os.path.relpath(d, root).lower().split(os.sep)[:-1]
        if not any(p.startswith(split) for p in parts):
            continue
        imgs = sorted(os.path.join(d, f) for f in files if f.lower().endswith(EXT))
        if imgs:
            found.setdefault(name, []).extend(imgs)
    return found


def process(path):
    """-> (байты JPEG, (w0, h0), (w1, h1), перекодировано ли)."""
    with open(path, "rb") as f:
        raw = f.read()
    with Image.open(io.BytesIO(raw)) as im:
        w0, h0 = im.size
        fmt = im.format
        if min(w0, h0) <= SHORT and fmt == "JPEG" and im.mode == "RGB":
            return raw, (w0, h0), (w0, h0), False
        im = im.convert("RGB")
        if min(w0, h0) > SHORT:
            if w0 <= h0:
                w1, h1 = SHORT, max(1, round(h0 * SHORT / w0))
            else:
                w1, h1 = max(1, round(w0 * SHORT / h0)), SHORT
            im = im.resize((w1, h1), Image.BICUBIC, reducing_gap=None)
        else:
            w1, h1 = w0, h0
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=QUALITY)
        return buf.getvalue(), (w0, h0), (w1, h1), True


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="/kaggle/working/in100pack")
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0, help="только для проверки: столько файлов на часть")
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    print(f"=== e_prep_kaggle.py  {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC; md5 скрипта "
          f"{hashlib.md5(open(__file__, 'rb').read()).hexdigest()}; потоков {args.threads}; ядер {os.cpu_count()}", flush=True)
    if args.limit:
        print("⚠️ ПРОВЕРОЧНЫЙ прогон: не для обучения.", flush=True)
    index = {}
    classes = None
    for split in ("train", "val"):
        found = scan(args.data, split)
        got, want = set(found), set(CMC100)
        if got != want:
            sys.exit(f"⚠️ классы {split}: найдено {len(got)}; нет {sorted(want - got)[:10]}; лишние {sorted(got - want)[:10]} — остановка")
        cls = sorted(got)
        if classes is None:
            classes = cls
        assert cls == classes
        items = [(p, i) for i, c in enumerate(cls) for p in found[c]]
        if args.limit:
            items = items[:: max(1, len(items) // args.limit)][: args.limit]
        n = len(items)
        print(f"{split}: {n} файлов, классов {len(cls)} ({time.time() - t0:.0f} с)", flush=True)
        offs = np.zeros(n, np.int64)
        lens = np.zeros(n, np.int64)
        size0 = np.zeros((n, 2), np.int32)
        size1 = np.zeros((n, 2), np.int32)
        rec = np.zeros(n, np.bool_)
        pos, t1 = 0, time.time()
        binp = os.path.join(args.out, f"{split}.bin")
        with open(binp, "wb") as fo, ThreadPoolExecutor(args.threads) as ex:
            for k, (b, s0, s1, r) in enumerate(ex.map(process, (p for p, _ in items), chunksize=16)):
                fo.write(b)
                offs[k], lens[k], size0[k], size1[k], rec[k] = pos, len(b), s0, s1, r
                pos += len(b)
                if (k + 1) % 5000 == 0 or k + 1 == n:
                    el = time.time() - t1
                    print(f"  {split}: {k + 1}/{n}, {(k + 1) / el:.0f} файлов/с, {pos / 2 ** 30:.2f} ГБ, осталось ≈ {(n - k - 1) / ((k + 1) / el) / 60:.1f} мин", flush=True)
        index[f"{split}_off"], index[f"{split}_len"] = offs, lens
        index[f"{split}_label"] = np.array([y for _, y in items], np.int16)
        index[f"{split}_name"] = np.array([os.path.relpath(p, args.data) for p, _ in items])
        index[f"{split}_size0"], index[f"{split}_size1"], index[f"{split}_recoded"] = size0, size1, rec
        print(f"{split}: готово, {pos / 2 ** 30:.2f} ГБ, перекодировано {int(rec.sum())} из {n}; короткая сторона после: "
              f"мин {int(size1.min(1).min())}, макс {int(size1.min(1).max())}; средний файл {pos / n / 1e3:.1f} КБ", flush=True)
    index["classes"] = np.array(classes)
    index["short"], index["quality"] = np.int32(SHORT), np.int32(QUALITY)
    np.savez(os.path.join(args.out, "index.npz"), **index)
    with open(os.path.join(args.out, "classes.txt"), "w") as f:
        f.write("\n".join(classes) + "\n")
    print(f"md5 списка классов {hashlib.md5(' '.join(classes).encode()).hexdigest()}", flush=True)
    # проверка: 20 случайных записей каждой части читаются и совпадают по размеру
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        mm = np.memmap(os.path.join(args.out, f"{split}.bin"), np.uint8, "r")
        for k in rng.choice(len(index[f"{split}_off"]), 20, replace=False):
            o, l = index[f"{split}_off"][k], index[f"{split}_len"][k]
            with Image.open(io.BytesIO(mm[o:o + l].tobytes())) as im:
                assert tuple(im.size) == tuple(index[f"{split}_size1"][k]), (split, k)
        print(f"Проверка чтения {split}: 20 случайных записей ✅", flush=True)
    with open(os.path.join(args.out, "MANIFEST.txt"), "w") as f:
        for name in ("train.bin", "val.bin", "index.npz", "classes.txt"):
            m = md5_file(os.path.join(args.out, name))
            f.write(f"{m}  {name}\n")
            print(f"md5 {name} {m}", flush=True)
    print(f"Готово за {(time.time() - t0) / 60:.1f} мин; папка {args.out}", flush=True)


if __name__ == "__main__":
    main()

"""
e_train_kaggle_c67.py — опыт E, полный: обучение ViT-Tiny с нуля на ImageNet-100 на GPU Kaggle. Добавлено 18.11 (реестр 277).

ОТКУДА: перенос e_train.py (пилот на Маке, md5 bb6afe7f…) на CUDA. Рецепт обучения тот же — сеть, аугментации, условия 1–5,
  коллажи, AdamW, график lr, сглаживание, обрезка градиента, 40 эпох, контрольные точки. Отличия (все — из-за платформы и данных):
  (1) данные: ImageNet-100 (100 классов ImageNet, список CMC: Tian, Krishnan, Isola 2019) вместо Food-101;
  (2) устройство CUDA; смешанная точность float16 (--amp fp16, по умолчанию) или float32 (--amp none) — решено замером: float16 (реестр 281, 284);
  (3) пачка 128 целиком (накопления нет, --micro 128; на GPU 16 ГБ хватает памяти);
  (4) проверка точности после каждой эпохи — на ВСЕХ проверочных фото (их 5000, это дёшево);
  (5) сессия Kaggle длится ≤ 12 ч: --hours (по умолчанию 11,3) — перед новой эпохой скрипт оценивает, успеет ли её закончить,
      и если нет — сохраняет last.pt и выходит; продолжение — та же команда с --resume <папка прошлого вывода>;
  (7) v3 (19.11, реестр 284): цветовые аугментации пилота (серый | соляризация | размытие, ColorJitter, нормировка) — пачкой
      на GPU (класс GPUAug, те же распределения параметров, независимо для каждой картинки и каждой половины коллажа); на CPU —
      только декодирование, RandomResizedCrop и отражения условий 2 / 3. Причина — упор в CPU (302 / 270 карт./с, реестр 283).
  (6) --bench: замер скорости (только загрузка; только шаг на GPU с AMP и без) без обучения — для решения о формате данных
      и точности до регистрации.
  (8) v4 (19.11, реестр 287): сверка GPUAug с PIL-путём пилота (check_aug, 16 фото) — в начале КАЖДОЙ сессии обучения, а не только
      при --bench; если наибольшая разница > 1 уровня из 255 — остановка до первого шага. Сверка не тратит случайность
      (параметры заданы числами), а при продолжении состояния генераторов восстанавливаются после неё. Число процессов
      загрузки (--workers) записывается в контрольные точки; продолжение с другим --workers остановлено (случайность обрезок
      зависит от числа процессов: облачная проверка 19.11 — при 1 и 2 процессах разные веса, при равном — совпадают).
  (9) v5 (19.11, реестр 305; дизайн МОИ_ЗАМЕТКИ/ПЛАН_КОЛЛАЖИ_v3.md v3.1, отдельная регистрация): условия 6 и 7 — коллажи с ДВУМЯ
      головами (один линейный выход на 200 = голова A [0:100] и голова B [100:200], одинаковый в обоих условиях):
      6 «упорядоченная пара»: A → класс левой, B → класс правой; потеря ½·CE(A, левый) + ½·CE(B, правый), сглаживание 0,1;
      7 «неупорядоченная пара»: обе головы → ½ / ½ на два класса; потеря ½·CE(A, ½л + ½п) + ½·CE(B, то же), сглаживание 0,1.
      Коллажи, аугментации, рецепт — как в условиях 4–5. Проверка после каждой эпохи на всех проверочных коллажах:
      узнавание половин r = доля (коллаж, половина), где класс половины в top-2 распределения p = ½(softmax A + softmax B)
      (гейт «выучила»: r ≥ 20% на эпохе 40); справка — A→левый, B→правый (top-1) и «перестановка половин» (поменять половины
      местами: доля, где top-1 головы A = класс НОВОЙ левой). Файл e_train_kaggle_c67.py — только для коллажей v3; зарегистрированный e_train_kaggle.py v4 (9b57fab6…) для условий 1–3 не меняется. Обучение условий 1–5 здесь то же (в метаданные добавлено только n_out).
  Все обучения полного E — на одной платформе, одном типе GPU и одной точности (Мак и Kaggle не смешивать); тип GPU печатается.

ДАННЫЕ (дополнено 19.11, реестр 282): если в папке --data есть index.npz — читаются упакованные данные e_prep_kaggle.py
  (фото, уменьшенные до короткой стороны 256, JPEG 90; причина — чтение 132 тыс. файлов набора Kaggle 14 файлов/с, реестр 281).
  Иначе папка --data ищется рекурсивно: каталоги с именем вида n01234567 внутри пути, где есть часть, начинающаяся на
  «train», — обучение; на «val» — проверка. Так читается и раскладка train.X1…X4 / val.X, и train/ val/, и уменьшенная копия
  с той же раскладкой. Набор классов сверяется со списком CMC (ниже); при расхождении — остановка с перечнем отличий.

ЗАПУСК (блокнот Kaggle, GPU, интернет не нужен; данные добавлены как Input):
  замер:     python -u e_train_kaggle.py --cond 1 --bench --data /kaggle/input/<набор> 2>&1 | tee bench_log.txt
  обучение:  python -u e_train_kaggle.py --cond 1 --seed 0 --data /kaggle/input/<набор> 2>&1 | tee -a e_full_c1_s0_log.txt
  продолжение в новой сессии: то же + --resume /kaggle/input/<вывод прошлой сессии>/e_full/c1_s0
  2 × T4: два процесса, --gpu 0 и --gpu 1, по --workers 2 (ядер CPU 4).
"""
import argparse
import glob
import hashlib
import io
import math
import os
import random
import re
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

CKPT_EPOCHS = (1, 2, 3, 5, 8, 12, 20, 30, 40)

# ImageNet-100, список CMC (github.com/HobbitLong/CMC, imagenet100.txt), переписан 18.11; 100 различных wnid.
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


def scan(root, split):
    """{wnid: [пути]} для split = 'train' / 'val' (по части пути ниже root)."""
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


class INet100(torch.utils.data.Dataset):
    """Два режима: (1) папка с файлами (рекурсивный поиск папок wnid); (2) упакованные данные e_prep_kaggle.py — в папке есть
    index.npz и <split>.bin (весь .bin читается в память один раз; процессы загрузки получают его через fork)."""

    def __init__(self, root, split, tf, limit=0, allow_classes=False):
        self.packed = os.path.exists(os.path.join(root, "index.npz"))
        if self.packed:
            z = np.load(os.path.join(root, "index.npz"))
            got = set(str(c) for c in z["classes"])
            labels = z[f"{split}_label"].astype(np.int64)
            self.off, self.len = z[f"{split}_off"], z[f"{split}_len"]
            t0 = time.time()
            self.blob = np.fromfile(os.path.join(root, f"{split}.bin"), dtype=np.uint8)
            assert len(self.blob) == int(self.off[-1] + self.len[-1]), "размер .bin не совпадает с индексом"
            print(f"  {split}.bin: {len(self.blob) / 2 ** 30:.2f} ГБ прочитано в память за {time.time() - t0:.0f} с", flush=True)
            classes = [str(c) for c in z["classes"]]
            found = {c: [int(k) for k in np.flatnonzero(labels == i)] for i, c in enumerate(classes)}
        else:
            found = scan(root, split)
            got = set(found)
        want = set(CMC100)
        if got != want:
            msg = (f"классы {split}: найдено {len(got)}; нет в данных {sorted(want - got)[:10]}…({len(want - got)}); "
                   f"лишние {sorted(got - want)[:10]}…({len(got - want)})")
            if not allow_classes:
                sys.exit("⚠️ " + msg + " — остановка (сверить список CMC и набор данных)")
            print("⚠️ " + msg, flush=True)
        self.classes = sorted(got)
        self.cls = {c: i for i, c in enumerate(self.classes)}
        items = [(p, self.cls[c]) for c in self.classes for p in found[c]]
        if limit:
            items = items[:: max(1, len(items) // limit)][:limit]
        self.items = items
        self.per_class = {c: len(found[c]) for c in self.classes}
        self.tf = tf

    def pil(self, i):
        from PIL import Image
        p = self.items[i][0]
        if self.packed:
            o, n = int(self.off[p]), int(self.len[p])
            with Image.open(io.BytesIO(self.blob[o:o + n].tobytes())) as im:
                return im.convert("RGB")
        with Image.open(p) as im:
            return im.convert("RGB")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.tf(self.pil(i)), self.items[i][1]


def transforms(cond, mean, std, train):
    import torchvision.transforms as T
    if not train:
        return T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean, std)])
    t = [T.RandomResizedCrop(224, scale=(0.25, 1.0))]
    if cond == 2:
        t.append(T.RandomHorizontalFlip(0.5))
    if cond == 3:
        t.append(T.RandomVerticalFlip(0.5))
    t += [T.RandomChoice([T.Grayscale(3), T.RandomSolarize(128, p=1.0), T.GaussianBlur(9, sigma=(0.1, 2.0))]),
          T.ColorJitter(0.3, 0.3, 0.3), T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(t)


def half_transform(mean, std, train):
    """Одна половина коллажа: 224 (высота) × 112 (ширина). Обучение — те же аугментации, что в условии 1 (без отражений)."""
    import torchvision.transforms as T
    if not train:
        return T.Compose([T.Resize(256), T.CenterCrop((224, 112)), T.ToTensor(), T.Normalize(mean, std)])
    return T.Compose([T.RandomResizedCrop((224, 112), scale=(0.25, 1.0), ratio=(0.375, 0.667)),
                      T.RandomChoice([T.Grayscale(3), T.RandomSolarize(128, p=1.0), T.GaussianBlur(9, sigma=(0.1, 2.0))]),
                      T.ColorJitter(0.3, 0.3, 0.3), T.ToTensor(), T.Normalize(mean, std)])


def cpu_transform(cond):
    """Обучение, v3 (19.11, реестр 284): на CPU — только геометрия пилота (RandomResizedCrop и отражение условия 2 / 3), выход uint8;
    цветовые аугментации пилота делает GPUAug пачкой на GPU."""
    import torchvision.transforms as T
    t = [T.RandomResizedCrop(224, scale=(0.25, 1.0))]
    if cond == 2:
        t.append(T.RandomHorizontalFlip(0.5))
    if cond == 3:
        t.append(T.RandomVerticalFlip(0.5))
    return T.Compose(t + [T.PILToTensor()])


def cpu_half_transform():
    """Половина коллажа для обучения, v3: геометрия как в half_transform пилота, выход uint8."""
    import torchvision.transforms as T
    return T.Compose([T.RandomResizedCrop((224, 112), scale=(0.25, 1.0), ratio=(0.375, 0.667)), T.PILToTensor()])


class GPUAug:
    """Цветовая часть аугментации пилота пачкой на GPU, независимо для каждой картинки (v3, 19.11, реестр 284):
    RandomChoice[Grayscale(3) | RandomSolarize(128, p=1) | GaussianBlur(9, sigma ~ U(0,1; 2,0))] → ColorJitter(0,3; 0,3; 0,3)
    (яркость, контраст, насыщенность ~ U(0,7; 1,3), порядок — случайная перестановка) → ToTensor + Normalize(mean, std).
    Работает в целых уровнях 0–255 с округлением после каждого шага, как PIL-путь пилота (формулы PIL: серый —
    (R·19595 + G·38470 + B·7471 + 32768) >> 16; контраст — смешение с серым цветом среднего уровня яркости картинки;
    насыщенность — смешение со своей серой копией; размытие — ядро torchvision, отражающая граница). Случайные параметры
    берутся из отдельного генератора self.gen (CPU) — его состояние сохраняется в last.pt."""

    def __init__(self, mean, std, seed, device):
        self.gen = torch.Generator().manual_seed(seed)
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        self.device = device
        x = torch.linspace(-4.0, 4.0, 9)
        self.kx = x.to(device)

    @staticmethod
    def gray(x):
        xi = x.to(torch.int64)
        g = (xi[:, 0] * 19595 + xi[:, 1] * 38470 + xi[:, 2] * 7471 + 32768) >> 16
        return g.unsqueeze(1).to(x.dtype)

    @staticmethod
    def blend(a, b, f):
        """PIL Image.blend(a, b, f) = a + f·(b − a), с обрезкой 0–255 и отбрасыванием дробной части."""
        return (a + f.view(-1, 1, 1, 1) * (b - a)).clamp(0, 255).floor()

    def blur(self, x, sigma):
        n, c, h, w = x.shape
        k1 = torch.exp(-0.5 * (self.kx.view(1, -1) / sigma.view(-1, 1)) ** 2)
        k1 = k1 / k1.sum(1, keepdim=True)
        k2 = (k1.view(n, 9, 1) * k1.view(n, 1, 9)).repeat_interleave(c, 0).view(n * c, 1, 9, 9)
        xp = F.pad(x.reshape(1, n * c, h, w), (4, 4, 4, 4), mode="reflect")
        return F.conv2d(xp, k2, groups=n * c).view(n, c, h, w).round().clamp(0, 255)

    def __call__(self, xu8):
        x = xu8.to(self.device, non_blocking=True).float()
        n = x.shape[0]
        g = self.gen
        choice = torch.randint(0, 3, (n,), generator=g)
        sigma = torch.empty(n).uniform_(0.1, 2.0, generator=g)
        fac = torch.empty(n, 3).uniform_(0.7, 1.3, generator=g)
        order = torch.argsort(torch.rand(n, 3, generator=g), dim=1)
        choice, sigma, fac, order = (t.to(self.device) for t in (choice, sigma, fac, order))
        m = choice == 0
        if m.any():
            x[m] = self.gray(x[m]).expand(-1, 3, -1, -1)
        m = choice == 1
        if m.any():
            x[m] = torch.where(x[m] >= 128, 255 - x[m], x[m])
        m = choice == 2
        if m.any():
            x[m] = self.blur(x[m], sigma[m])
        for k in range(3):
            op = order[:, k]
            gr = self.gray(x)
            mean_l = torch.floor(gr.mean(dim=(1, 2, 3), keepdim=True) + 0.5)
            cand = (self.blend(torch.zeros_like(x), x, fac[:, 0]),
                    self.blend(mean_l.expand_as(x), x, fac[:, 1]),
                    self.blend(gr.expand_as(x), x, fac[:, 2]))
            x = torch.where((op == 0).view(-1, 1, 1, 1), cand[0], torch.where((op == 1).view(-1, 1, 1, 1), cand[1], cand[2]))
        return (x / 255.0 - self.mean) / self.std

    def collage(self, xu8):
        """Половины коллажа — независимо (как в пилоте: каждая половина проходит свои аугментации), затем склейка."""
        w = xu8.shape[-1] // 2
        return torch.cat([self(xu8[..., :w]), self(xu8[..., w:])], dim=3)


class Collage(torch.utils.data.Dataset):
    """Коллаж из двух фото РАЗНЫХ классов: x = [левое | правое] (3 × 224 × 224), возвращает (x, класс левого, класс правого).
    Обучение: партнёр — случайный (torch-генератор процесса загрузки). Проверка: партнёр детерминирован."""

    def __init__(self, base, tf, train):
        self.base, self.tf, self.train = base, tf, train

    def __len__(self):
        return len(self.base.items)

    def _partner(self, i):
        n = len(self.base.items)
        ya = self.base.items[i][1]
        if self.train:
            while True:
                j = int(torch.randint(n, (1,)))
                if self.base.items[j][1] != ya:
                    return j
        j = (i * 7919 + 1) % n
        while self.base.items[j][1] == ya:
            j = (j + 1) % n
        return j

    def _load(self, k):
        return self.tf(self.base.pil(k)), self.base.items[k][1]

    def __getitem__(self, i):
        j = self._partner(i)
        a, ya = self._load(i)
        b, yb = self._load(j)
        return torch.cat([a, b], dim=2), ya, yb


def soft_target(cond, ya, yb, n_classes):
    """Условие 4: вся масса на классе левого; условие 5: 0,5 / 0,5. Сглаживание 0,1 добавляет cross_entropy."""
    t = torch.zeros(len(ya), n_classes, device=ya.device)
    if cond == 4:
        t[torch.arange(len(ya)), ya] = 1.0
    else:
        t[torch.arange(len(ya)), ya] = 0.5
        t[torch.arange(len(ya)), yb] = 0.5
    return t


N_OUT = {6: 200, 7: 200}          # условия с двумя головами: выход 200 = A [0:100] + B [100:200]


def two_head_loss(cond, logits, ya, yb, ncls):
    """Условия 6 и 7 (v5). logits: [n, 2·ncls] (float32)."""
    a, b = logits[:, :ncls], logits[:, ncls:]
    if cond == 6:
        return 0.5 * F.cross_entropy(a, ya, label_smoothing=0.1) + 0.5 * F.cross_entropy(b, yb, label_smoothing=0.1)
    t = soft_target(5, ya, yb, ncls)
    return 0.5 * F.cross_entropy(a, t, label_smoothing=0.1) + 0.5 * F.cross_entropy(b, t, label_smoothing=0.1)


def make_model(seed, n_classes):
    import timm
    torch.manual_seed(seed)
    return timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=n_classes, drop_path_rate=0.1)


def lr_at(step, total, warm, base, low=1e-5):
    if step < warm:
        return base * (step + 1) / warm
    q = (step - warm) / max(1, total - warm)
    return low + 0.5 * (base - low) * (1 + math.cos(math.pi * q))


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    hit = n = 0
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            out = model(x.to(device, non_blocking=True))
        hit += int((out.float().argmax(1).cpu() == y).sum())
        n += len(y)
    model.train()
    return hit / max(1, n)


@torch.no_grad()
def evaluate_collage(model, loader, device, amp):
    """Доля top-1 = левый; доля top-1 = правый; доля «оба класса в top-2»."""
    model.eval()
    left = right = both = n = 0
    for x, ya, yb in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            out = model(x.to(device, non_blocking=True))
        top2 = out.float().cpu().topk(2, dim=1).indices
        left += int((top2[:, 0] == ya).sum())
        right += int((top2[:, 0] == yb).sum())
        both += int((((top2[:, 0] == ya) & (top2[:, 1] == yb)) | ((top2[:, 0] == yb) & (top2[:, 1] == ya))).sum())
        n += len(ya)
    model.train()
    return left / max(1, n), right / max(1, n), both / max(1, n)


@torch.no_grad()
@torch.no_grad()
def evaluate_two_head(model, loader, device, amp, ncls):
    """Условия 6 и 7 (v5): r_left, r_right — класс половины в top-2 p = ½(softmax A + softmax B); A→левый, B→правый (top-1);
    перестановка половин: top-1 головы A на коллаже [правая | левая] = класс новой левой."""
    model.eval()
    rl = rr = al = br = sw = n = 0
    for x, ya, yb in loader:
        x = x.to(device, non_blocking=True)
        w = x.shape[-1] // 2
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            out = model(x).float().cpu()
            out_sw = model(torch.cat([x[..., w:], x[..., :w]], dim=3)).float().cpu()
        a, b = out[:, :ncls], out[:, ncls:]
        p = 0.5 * (a.softmax(1) + b.softmax(1))
        top2 = p.topk(2, dim=1).indices
        rl += int((top2 == ya[:, None]).any(1).sum())
        rr += int((top2 == yb[:, None]).any(1).sum())
        al += int((a.argmax(1) == ya).sum())
        br += int((b.argmax(1) == yb).sum())
        sw += int((out_sw[:, :ncls].argmax(1) == yb).sum())
        n += len(ya)
    model.train()
    d = max(1, n)
    return rl / d, rr / d, al / d, br / d, sw / d


@torch.no_grad()
def fp(model):
    return float(sum(p.detach().cpu().double().abs().sum() for p in model.parameters()))


def check_aug(base, device, n=16):
    """Сверка GPUAug с PIL-путём пилота (torchvision на PIL) на n фото набора, при одинаковых параметрах, на устройстве обучения."""
    import torchvision.transforms.functional as TF
    ims = [TF.center_crop(base.pil(k * 997 % len(base.items)), 224) for k in range(n)]
    X = torch.stack([TF.pil_to_tensor(c) for c in ims]).float().to(device)
    A = GPUAug((0.5,) * 3, (0.5,) * 3, 0, device)
    worst = 0.0

    def cmp(name, mine, ref_list):
        nonlocal worst
        ref = torch.stack([TF.pil_to_tensor(r) for r in ref_list]).float().to(device)
        d = (mine - ref).abs()
        worst = max(worst, d.max().item())
        print(f"  сверка {name:13s} макс. разница {d.max().item():.0f} уровней, доля несовпавших {(d > 0).float().mean().item():.5f}", flush=True)

    cmp("серый", A.gray(X).expand(-1, 3, -1, -1), [c.convert("L").convert("RGB") for c in ims])
    cmp("соляризация", torch.where(X >= 128, 255 - X, X), [TF.solarize(c, 128) for c in ims])
    for s_ in (0.1, 0.9, 2.0):
        cmp(f"размытие {s_}", A.blur(X.clone(), torch.full((n,), s_, device=device)), [TF.gaussian_blur(c, 9, s_) for c in ims])
    gr = A.gray(X)
    ml = torch.floor(gr.mean(dim=(1, 2, 3), keepdim=True) + 0.5)
    for f_ in (0.7, 1.3):
        fv = torch.full((n,), f_, device=device)
        cmp(f"яркость {f_}", A.blend(torch.zeros_like(X), X, fv), [TF.adjust_brightness(c, f_) for c in ims])
        cmp(f"контраст {f_}", A.blend(ml.expand_as(X), X, fv), [TF.adjust_contrast(c, f_) for c in ims])
        cmp(f"насыщенность {f_}", A.blend(gr.expand_as(X), X, fv), [TF.adjust_saturation(c, f_) for c in ims])
    print(f"СВЕРКА GPUAug с PIL-путём: наибольшая разница {worst:.0f} уровней из 255 {'✅' if worst <= 1 else '⚠️'}", flush=True)
    return worst


def bench(args, model, tr, device, ncls, augment):
    """Скорость: (а) шаг на GPU на случайных тензорах, с AMP и без; (б) только загрузка данных."""
    collage = args.cond in (4, 5, 6, 7)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    x = torch.randn(args.batch, 3, 224, 224, device=device)
    y = torch.randint(ncls, (args.batch,), device=device)
    for amp in ((False, True) if device.type == "cuda" else (False,)):
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        for k in range(5 + args.bench_steps):
            if k == 5:
                sync()
                t0 = time.time()
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sync()
        ips = args.bench_steps * args.batch / (time.time() - t0)
        print(f"ЗАМЕР шаг на GPU ({'float16 AMP' if amp else 'float32'}), пачка {args.batch}: {ips:.0f} карт./с; "
              f"память {(torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == 'cuda' else 0):.2f} ГБ", flush=True)
    for w in sorted({args.workers, max(1, (os.cpu_count() or 4))}):
        dl = torch.utils.data.DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=w, drop_last=True,
                                         pin_memory=True, persistent_workers=False)
        it, n, t0 = iter(dl), 0, None
        for k in range(3 + args.bench_steps):
            b = next(it)
            if k == 3:
                t0 = time.time()
            if k >= 3:
                n += len(b[1])
        print(f"ЗАМЕР загрузка ({'коллажи — два фото на картинку' if collage else 'одно фото'}), процессов {w}: "
              f"{n / (time.time() - t0):.0f} карт./с", flush=True)
        xb = b[0]
        del it, dl
    for k in range(3 + args.bench_steps):
        if k == 3:
            sync()
            t0 = time.time()
        augment(xb)
    sync()
    print(f"ЗАМЕР цветовые аугментации на {device}: {args.bench_steps * len(xb) / (time.time() - t0):.0f} карт./с", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", type=int, choices=(1, 2, 3, 4, 5, 6, 7), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--micro", type=int, default=128, help="часть пачки на одном проходе (накопление градиента)")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--amp", choices=("fp16", "none"), default="fp16")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--hours", type=float, default=11.3, help="предел времени этого запуска (сессия Kaggle ≤ 12 ч)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="/kaggle/working/e_full")
    ap.add_argument("--resume", default="", help="папка c<cond>_s<seed> из вывода прошлой сессии (Input)")
    ap.add_argument("--allow-classes", action="store_true", help="только для проверки: не останавливаться при другом наборе классов")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--bench-steps", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0, help="только для проверки: столько картинок")
    ap.add_argument("--max-steps", type=int, default=0, help="только для проверки: остановиться после стольких шагов")
    ap.add_argument("--cpu-test", action="store_true", help="только для проверки в облаке без GPU")
    args = ap.parse_args()
    t_launch = time.time()

    if args.cpu_test:
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            sys.exit("⚠️ GPU не найден: в блокноте Kaggle включить Accelerator → GPU")
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
    amp = args.amp == "fp16" and device.type == "cuda"
    import timm
    import torchvision
    tag = f"c{args.cond}_s{args.seed}"
    out = os.path.join(args.out, tag)
    os.makedirs(out, exist_ok=True)
    gname = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"\n=== Опыт E (ImageNet-100): условие {args.cond}, seed {args.seed}   {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC   "
          f"torch {torch.__version__}, timm {timm.__version__}, torchvision {torchvision.__version__}, "
          f"устройство {device} ({gname}), точность {'float16 AMP' if amp else 'float32'}, ядер CPU {os.cpu_count()}", flush=True)
    if args.limit or args.max_steps or args.cpu_test or args.allow_classes:
        print("⚠️ ПРОВЕРОЧНЫЙ прогон: числа не для статьи.", flush=True)
    print(f"md5 {os.path.basename(__file__)} {hashlib.md5(open(__file__, 'rb').read()).hexdigest()}", flush=True)

    n_out = N_OUT.get(args.cond, 100)
    model = make_model(args.seed, n_out)
    cfg = model.pretrained_cfg
    mean, std = tuple(cfg["mean"]), tuple(cfg["std"])
    print(f"Сеть: vit_tiny_patch16_224, выход {n_out}, {sum(p.numel() for p in model.parameters()) / 1e6:.2f} млн параметров; "
          f"сетка {tuple(model.patch_embed.grid_size)}; блоков {len(model.blocks)}; голов {model.blocks[0].attn.num_heads}; "
          f"нормировка {mean}/{std}; отпечаток начальных весов {fp(model):.4f}", flush=True)
    assert tuple(model.patch_embed.grid_size) == (14, 14) and len(model.blocks) == 12 and model.blocks[0].attn.num_heads == 3

    collage = args.cond in (4, 5, 6, 7)
    two = args.cond in (6, 7)
    t0 = time.time()
    base_tr = INet100(args.data, "train", None, args.limit, args.allow_classes)
    base_te = INet100(args.data, "val", None, args.limit // 3 if args.limit else 0, args.allow_classes)
    print(f"Данные ({time.time() - t0:.0f} с на поиск): обучение {len(base_tr.items)}, проверка {len(base_te.items)}, "
          f"классов {len(base_tr.classes)}; на класс (обучение) {min(base_tr.per_class.values())}–{max(base_tr.per_class.values())}, "
          f"(проверка) {min(base_te.per_class.values())}–{max(base_te.per_class.values())}; "
          f"md5 списка классов {hashlib.md5(' '.join(base_tr.classes).encode()).hexdigest()}", flush=True)
    assert base_tr.classes == base_te.classes
    if collage:
        tr = Collage(base_tr, cpu_half_transform(), True)
        te = Collage(base_te, half_transform(mean, std, False), False)
        kind = {4: "метка = класс левого", 5: "метка = оба класса, 0,5 / 0,5",
                6: "две головы: A = класс левого, B = класс правого (упорядоченная пара)",
                7: "две головы: обе = 0,5 / 0,5 на два класса (неупорядоченная пара)"}[args.cond]
        print(f"КОЛЛАЖ ({kind}); половина 224 × 112", flush=True)
    else:
        base_tr.tf = cpu_transform(args.cond)
        base_te.tf = transforms(args.cond, mean, std, False)
        tr, te = base_tr, base_te
    ncls = 100
    print(f"Аугментация на CPU: {tr.tf}; на GPU: GPUAug (серый | соляризация | размытие → ColorJitter 0,3/0,3/0,3 → нормировка)", flush=True)
    gaug = GPUAug(mean, std, 10_000 + args.seed, device)
    augment = gaug.collage if collage else gaug
    model.to(device).train()

    if args.bench:
        check_aug(base_tr, device)
        bench(args, model, tr, device, ncls, augment)
        print(f"Замер закончен за {time.time() - t_launch:.0f} с.", flush=True)
        return

    if check_aug(base_tr, device) > 1:
        sys.exit("⚠️ GPUAug расходится с PIL-путём пилота больше чем на 1 уровень — обучение не начато (сверить GPU / версии)")
    assert args.batch % args.micro == 0, "пачка должна делиться на часть"
    accum = args.batch // args.micro
    print(f"Пачка {args.batch} = {accum} × {args.micro}", flush=True)
    g = torch.Generator().manual_seed(args.seed)
    dl = torch.utils.data.DataLoader(tr, batch_size=args.micro, shuffle=True, num_workers=args.workers, drop_last=True,
                                     generator=g, pin_memory=device.type == "cuda", persistent_workers=False)
    # persistent_workers=False: зерна процессов загрузки берутся из g в начале КАЖДОЙ эпохи → продолжение с last.pt точное
    # (с постоянными процессами их случайность после продолжения расходится с непрерывным обучением — проверено 18.11).
    dte = torch.utils.data.DataLoader(te, batch_size=args.micro, shuffle=False, num_workers=args.workers,
                                      pin_memory=device.type == "cuda")

    decay = [p for n, p in model.named_parameters() if p.ndim > 1 and "pos_embed" not in n and "cls_token" not in n]
    no_decay = [p for n, p in model.named_parameters() if not (p.ndim > 1 and "pos_embed" not in n and "cls_token" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05}, {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    steps_per_epoch = len(dl) // accum
    total = steps_per_epoch * args.epochs
    warm = steps_per_epoch * 5
    start_epoch, step = 0, 0
    last = os.path.join(out, "last.pt")
    if not os.path.exists(last) and args.resume:
        src = os.path.join(args.resume, "last.pt")
        if not os.path.exists(src):
            sys.exit(f"⚠️ --resume: нет {src}")
        for f in sorted(glob.glob(os.path.join(args.resume, "*.pt"))):
            shutil.copy2(f, out)
        print(f"Скопировано из {args.resume}: {sorted(os.path.basename(f) for f in glob.glob(os.path.join(out, '*.pt')))}", flush=True)
    if os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        assert ck.get("cond", args.cond) == args.cond and ck.get("seed", args.seed) == args.seed, "чужой last.pt"
        assert ck.get("amp", amp) == amp, "точность прошлой сессии другая — не смешивать"
        assert ck.get("workers", args.workers) == args.workers, \
            f"--workers прошлой сессии {ck.get('workers')} ≠ {args.workers}: случайность загрузки зависит от числа процессов — задать как раньше"
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        start_epoch, step = ck["epoch"], ck["step"]
        torch.set_rng_state(ck["rng_cpu"])
        random.setstate(ck["rng_py"])
        np.random.set_state(ck["rng_np"])
        g.set_state(ck["rng_dl"])
        assert "rng_aug" in ck, "last.pt без состояния GPUAug (старая версия скрипта) — не смешивать версии"
        gaug.gen.set_state(ck["rng_aug"])
        if device.type == "cuda" and "rng_cuda" in ck:
            torch.cuda.set_rng_state(ck["rng_cuda"], device)
        print(f"Продолжение с эпохи {start_epoch + 1} (шаг {step}); отпечаток загруженных весов {fp(model):.4f}", flush=True)
    if start_epoch >= args.epochs:
        print("Обучение уже закончено (last.pt — последняя эпоха).", flush=True)
        return

    ep_times = []
    for epoch in range(start_epoch, args.epochs):
        if ep_times:
            need = max(ep_times[-2:]) * 1.15
            if time.time() - t_launch + need > args.hours * 3600:
                print(f"⏸ Остановка по времени перед эпохой {epoch + 1}: прошло {(time.time() - t_launch) / 3600:.2f} ч, "
                      f"эпоха ≈ {need / 60:.0f} мин, предел {args.hours} ч. last.pt = эпоха {epoch}. "
                      f"Продолжить: та же команда + --resume <вывод этой сессии>/e_full/{tag}", flush=True)
                return
        te0, seen, run_loss = time.time(), 0, 0.0
        j = -1
        for i, batch in enumerate(dl):
            if i >= steps_per_epoch * accum:
                break
            k = i % accum
            if k == 0:
                for gr in opt.param_groups:
                    gr["lr"] = lr_at(step, total, warm, args.lr)
                opt.zero_grad(set_to_none=True)
            x = augment(batch[0])  # цветовые аугментации — в float32, вне autocast
            if collage:
                ya, yb = batch[1].to(device, non_blocking=True), batch[2].to(device, non_blocking=True)
                y = ya
            else:
                y = batch[1].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(x)
                if two:
                    loss = two_head_loss(args.cond, logits.float(), ya, yb, ncls)
                elif collage:
                    loss = F.cross_entropy(logits.float(), soft_target(args.cond, ya, yb, ncls), label_smoothing=0.1)
                else:
                    loss = F.cross_entropy(logits.float(), y, label_smoothing=0.1)
            scaler.scale(loss / accum).backward()
            lv = loss.item()
            if not math.isfinite(lv):
                print("⚠️ потери не число — остановка", flush=True)
                sys.exit(1)
            run_loss += lv / accum
            seen += len(y)
            if k < accum - 1:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            j = i // accum
            if (j + 1) % 100 == 0 or j + 1 == steps_per_epoch:
                el = time.time() - te0
                ips = seen / el
                left_h = ((args.epochs - epoch - 1) * steps_per_epoch * args.batch + (steps_per_epoch - j - 1) * args.batch) / ips / 3600
                print(f"  эпоха {epoch + 1}/{args.epochs}, шаг {j + 1}/{steps_per_epoch}: потери {run_loss / (j + 1):.3f}, "
                      f"lr {opt.param_groups[0]['lr']:.2e}, {ips:.0f} карт./с, осталось ≈ {left_h:.1f} ч", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        e1 = epoch + 1
        msg = f"Эпоха {e1} закончена: потери {run_loss / max(1, j + 1):.3f}, {time.time() - te0:.0f} с"
        if amp:
            msg += f", масштаб AMP {scaler.get_scale():.0f}"
        if two:
            rl, rr, al, br, sw = evaluate_two_head(model, dte, device, amp, ncls)
            acc = (rl + rr) / 2
            msg += (f"; ВСЕ проверочные коллажи: узнавание половин r {acc * 100:.2f}% (левые {rl * 100:.2f}%, правые {rr * 100:.2f}%; "
                    f"гейт ≥ 20%); A→левый {al * 100:.2f}%, B→правый {br * 100:.2f}%; перестановка половин: A → новый левый {sw * 100:.2f}%")
        elif collage:
            la, ra, ba = evaluate_collage(model, dte, device, amp)
            acc = la if args.cond == 4 else ba
            msg += f"; ВСЕ проверочные коллажи: top-1 = левый {la * 100:.2f}%, top-1 = правый {ra * 100:.2f}%, оба в top-2 {ba * 100:.2f}%"
        else:
            acc = evaluate(model, dte, device, amp)
            msg += f"; top-1 на ВСЕХ проверочных фото {acc * 100:.2f}%"
        print(msg + f"   ({time.strftime('%H:%M:%S', time.gmtime())} UTC)", flush=True)
        if e1 == 3 and acc < 0.03 and not (args.limit or args.max_steps):
            print("⚠️ РАННИЙ СИГНАЛ: после эпохи 3 точность ниже 3% — сеть не учится. Остановить и прислать лог.", flush=True)
        meta = {"epoch": e1, "cond": args.cond, "seed": args.seed, "amp": amp, "gpu": gname, "classes": base_tr.classes,
                "workers": args.workers, "n_out": n_out}
        if e1 in CKPT_EPOCHS or (args.max_steps and step >= args.max_steps):
            torch.save({"model": model.state_dict(), **meta}, os.path.join(out, f"ep{e1:02d}.pt"))
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": step,
              "rng_cpu": torch.get_rng_state(), "rng_py": random.getstate(), "rng_np": np.random.get_state(),
              "rng_dl": g.get_state(), "rng_aug": gaug.gen.get_state(), **meta}
        if device.type == "cuda":
            ck["rng_cuda"] = torch.cuda.get_rng_state(device)
        torch.save(ck, last + ".tmp")
        os.replace(last + ".tmp", last)
        ep_times.append(time.time() - te0)
        if args.max_steps and step >= args.max_steps:
            print(f"Проверочный прогон остановлен после {step} шагов.", flush=True)
            break
    print(f"Готово: {time.time() - t_launch:.0f} с в этом запуске; отпечаток весов {fp(model):.4f}; контрольные точки в {out}", flush=True)


if __name__ == "__main__":
    main()

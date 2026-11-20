"""
e_train.py — опыт E, ПИЛОТ: обучение маленькой ViT с нуля на Food-101. Добавлено 16.11 (реестр 241).

ЗАЧЕМ ПИЛОТ (это не зарегистрированный опыт; он будет раскрыт в регистрации опыта E):
  В опыте E одинаковые малые ViT обучаются с нуля в условиях, различающихся только отношением к отражению
  слева направо. Прежде чем тратить неделю счёта или деньги на аренду GPU, нужно проверить, что такая сеть
  при таком масштабе вообще (i) учится, (ii) хранит привязку «верхний» на дальнем фоне, (iii) становится
  нечувствительной к отражению слева направо. Если нет — опыт E в этом виде не имеет смысла.

СЕТЬ: timm vit_tiny_patch16_224 (192 измерения, 12 блоков × 3 головы; вход 224, патч 16, сетка 14 × 14 — как во
  всех прежних опытах, поэтому сцены, маски и определение дальнего фона не меняются), 101 класс, drop_path 0,1,
  нормировка входа — из pretrained_cfg timm (mean = std = 0,5). Веса случайные (seed), предобучения нет.
ДАННЫЕ: Food-101 (Bossard и др., 2014): 75 750 фото для обучения, 25 250 для проверки, 101 класс;
  архив food-101.tar.gz (sha256 d97d15e4…), распакован в ~/binding-vit-data/food-101.
УСЛОВИЯ (--cond): 1 — без отражений; 2 — отражение слева направо с вероятностью 0,5 (аугментация);
  3 — отражение СВЕРХУ ВНИЗ с вероятностью 0,5 (добавлено 18.11 после ставки Ерназара, реестр 262–264: пилот гипотезы
  «симметрия задачи» — сделать данные симметричными по вертикали; ожидание по гипотезе — «верхний» падает);
  4 — КОЛЛАЖ «левое»: два фото разных классов рядом (каждое — обрезка 224 × 112 по высоте × ширине), метка = класс ЛЕВОГО;
  5 — КОЛЛАЖ «оба»: те же коллажи, метка = оба класса без порядка (мягкая метка 0,5 / 0,5, та же кросс-энтропия со
  сглаживанием 0,1). Добавлено 18.11 (реестр 270), план E, условия 4–5: те же картинки, отличается только то, нужна ли
  задаче сторона. В условиях 4–5 отражений нет; каждая половина проходит те же аугментации, что в условии 1, независимо.
  Проверка точности в 4–5 — на проверочных коллажах (партнёр выбирается детерминированно): 4 — top-1 = класс левого и
  доля ответов «класс правого»; 5 — оба класса в top-2.
  Остальные условия E (штраф за неинвариантность, задача «отражена ли», восстановление пикселей, контроль) —
  в окончательном коде опыта, после ставки Ерназара. ПИЛОТ = условие 2, seed 0.
АУГМЕНТАЦИЯ (одинаковая во всех условиях, кроме отражения): RandomResizedCrop(224, scale 0,25–1); одно из трёх —
  оттенки серого / соляризация / размытие (как «3-Augment» у DeiT III); ColorJitter 0,3. Геометрических преобразований,
  кроме обрезки, НЕТ. [Исправлено 16.11 до запуска, реестр 243: в первой версии стоял TrivialAugmentWide с поворотами до
  135° и наклонами. Поворот на 180° = отражение ←→ + отражение ↑↓, поэтому приучение к поворотам тянет зеркальное
  отношение к 1 и мешает как раз той мере, ради которой ставится опыт.]
ОБУЧЕНИЕ: AdamW (lr 5e-4, weight decay 0,05), пачка 128, 5 эпох разогрева, затем косинус до 1e-5, сглаживание меток 0,1,
  обрезка градиента 1,0, float32; 40 эпох (по замеру e_bench.py ≈ 9–11 ч на Маке). Настройки не подбирались
  (это выбор по опыту похожих обучений, а не проверенный рецепт) — поэтому есть ранний сигнал, см. ниже.
КОНТРОЛЬНЫЕ ТОЧКИ: веса после эпох 1, 2, 3, 5, 8, 12, 20, 30, 40 (для мер по ходу обучения) и last.pt после каждой эпохи.
  Продолжение после остановки: запустить ту же команду ещё раз — обучение начнётся с начала незаконченной эпохи.
ПРОВЕРКА ТОЧНОСТИ: после каждой эпохи — top-1 на каждой пятой проверочной фотографии (5 050 фото; Resize 256,
  CenterCrop 224); после последней эпохи — на всех 25 250.
РАННИЙ СИГНАЛ: если после эпохи 3 top-1 ниже 3% (угадывание — 1%), печатается предупреждение «сеть не учится»:
  тогда остановить (Ctrl+C) и разбираться, а не ждать до утра.

КРИТЕРИИ ПИЛОТА (записаны ДО запуска, реестр 241; меры — e_measure.py, как в опыте B, many_nets.py):
  (i)   сеть учится: top-1 на всех проверочных фото после последней эпохи ≥ 20% (случайное угадывание — 1%).
        [Порог изменён 16.11 до запуска, реестр 243: 40% было названо без опоры; смысл критерия — убедиться, что обучение
        работает, а не оценить качество классификатора; ViT с нуля на 76 тыс. фото за 40 эпох может дать и 30–40%.]
  (ii)  «верхний» (обычная проба, дальний фон, последний слой, все 2000 сцен data_v2) выше, чем у каждой из трёх случайных
        ViT-Tiny (seed 100, 101, 102 — не начальные веса обучаемой сети), не меньше чем на 5 п.п.;
  (iii) зеркальное отношение (631 симметричная сцена, среднее по патчам, последний слой): верхняя граница 95% интервала
        ниже наименьшего отношения трёх случайных ViT-Tiny.
  Пилот «пройден», если выполнены (i)–(iii). Справка (не критерии): разрыв «верхний − левый»; пары голов по мере D (G3)
  с порогом по трём случайным ViT-Tiny; меры по контрольным точкам.
  Если (ii) не выполнено — сеть этого размера на этих данных не годится для E.
  Если (i) и (ii) выполнены, а (iii) нет — это ещё не значит, что слепоты нет (замечание Ерназара 14:42): еду часто снимают
  сверху, отражение ↑↓ для таких фото естественно, сеть может быть мало чувствительна к обоим отражениям, и отношение
  останется около 1. Тогда (и в любом случае, если (i) и (ii) выполнены) — второй шаг пилота: та же сеть без отражений
  (--cond 1, тот же seed) и сравнение частей отношения. Чтение, записанное заранее: «отражения при обучении снижают
  чувствительность к ←→», если d(←→) в условии 2 меньше, чем в условии 1, а d(↑↓) отличается меньше (по одному зерну —
  только указание, не вывод).
  Ограничения пилота: одно зерно; Food-101 — не «вертикальные» фото, как ImageNet; настройки обучения не подбирались.

ПАЧКА ПО ЧАСТЯМ (добавлено 16.11 в 20:50 после первого запуска, реестр 252): на Маке M3 с 8 ГБ при пачке 128 память MPS
  5,18 ГБ, Мак уходит в подкачку, шаг идёт 11 карт./с (e_diag2.py); при пачке 64 — 125 карт./с. Поэтому пачка 128 считается
  как 2 части по 64 с накоплением градиента (--micro 64): потери каждой части делятся на 2, обрезка градиента и шаг AdamW —
  один раз на 128 картинок. В сети нет BatchNorm (только LayerNorm), поэтому расчёт тот же, что при цельной пачке 128:
  те же 128 картинок по порядку перемешивания, тот же шаг, тот же график lr (шаги считаются по пачкам 128). Пилот до этого
  не дал ни одного шага (остановлен), поэтому это не изменение настроек по результатам.

ЗАПУСК (из папки binding-vit; ~9–11 ч; можно прервать Ctrl+C и запустить ту же команду снова — продолжит с места):
  caffeinate -is python -u e_train.py --cond 2 --seed 0 2>&1 | tee -a e_train_c2_s0_log.txt
  (Мак на зарядке, крышка открыта: при закрытой крышке он уснёт, несмотря на caffeinate)
"""
import argparse
import hashlib
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

CKPT_EPOCHS = (1, 2, 3, 5, 8, 12, 20, 30, 40)


class Food(torch.utils.data.Dataset):
    def __init__(self, root, split, tf, limit=0, every=1):
        self.root = root
        lines = [l.strip() for l in open(os.path.join(root, "meta", f"{split}.txt")) if l.strip()]
        classes = sorted({l.split("/")[0] for l in lines})
        self.cls = {c: i for i, c in enumerate(classes)}
        if limit:
            lines = lines[:: max(1, len(lines) // limit)][:limit]
        lines = lines[::every]
        self.items = [(os.path.join(root, "images", l + ".jpg"), self.cls[l.split("/")[0]]) for l in lines]
        self.tf = tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image
        p, y = self.items[i]
        with Image.open(p) as im:
            x = self.tf(im.convert("RGB"))
        return x, y


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
        from PIL import Image
        p, y = self.base.items[k]
        with Image.open(p) as im:
            return self.tf(im.convert("RGB")), y

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


@torch.no_grad()
def evaluate_collage(model, loader, device):
    """Возвращает: доля top-1 = левый; доля top-1 = правый; доля «оба класса в top-2»."""
    model.eval()
    left = right = both = n = 0
    for x, ya, yb in loader:
        out = model(x.to(device)).cpu()
        top2 = out.topk(2, dim=1).indices
        left += int((top2[:, 0] == ya).sum())
        right += int((top2[:, 0] == yb).sum())
        both += int((((top2[:, 0] == ya) & (top2[:, 1] == yb)) | ((top2[:, 0] == yb) & (top2[:, 1] == ya))).sum())
        n += len(ya)
    model.train()
    return left / max(1, n), right / max(1, n), both / max(1, n)


def make_model(seed, n_classes):
    import timm
    torch.manual_seed(seed)
    m = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=n_classes, drop_path_rate=0.1)
    return m


def lr_at(step, total, warm, base, low=1e-5):
    if step < warm:
        return base * (step + 1) / warm
    q = (step - warm) / max(1, total - warm)
    return low + 0.5 * (base - low) * (1 + math.cos(math.pi * q))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    hit = n = 0
    for x, y in loader:
        out = model(x.to(device))
        hit += int((out.argmax(1).cpu() == y).sum())
        n += len(y)
    model.train()
    return hit / max(1, n)


@torch.no_grad()
def fp(model):
    return float(sum(p.detach().cpu().double().abs().sum() for p in model.parameters()))  # MPS не умеет float64 — считаем на CPU


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", type=int, choices=(1, 2, 3, 4, 5), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--micro", type=int, default=64, help="часть пачки на одном проходе (накопление градиента)")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--data", default=os.path.expanduser("~/binding-vit-data/food-101"))
    ap.add_argument("--out", default=os.path.expanduser("~/binding-vit-data/e_pilot"))
    ap.add_argument("--limit", type=int, default=0, help="только для проверки в облаке: столько картинок")
    ap.add_argument("--max-steps", type=int, default=0, help="только для проверки в облаке: остановиться после стольких шагов")
    args = ap.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    import timm
    import torchvision
    tag = f"c{args.cond}_s{args.seed}"
    out = os.path.join(args.out, tag)
    os.makedirs(out, exist_ok=True)
    print(f"\n=== Опыт E, пилот: условие {args.cond}, seed {args.seed}   {time.strftime('%Y-%m-%d %H:%M:%S')}   "
          f"torch {torch.__version__}, timm {timm.__version__}, torchvision {torchvision.__version__}, устройство {device}", flush=True)
    if args.limit or args.max_steps:
        print("⚠️ ПРОВЕРОЧНЫЙ прогон: числа не для статьи.", flush=True)
    me = hashlib.md5(open(__file__, "rb").read()).hexdigest()
    print(f"md5 e_train.py {me}", flush=True)

    model = make_model(args.seed, 101)
    cfg = model.pretrained_cfg
    mean, std = tuple(cfg["mean"]), tuple(cfg["std"])
    print(f"Сеть: vit_tiny_patch16_224, {sum(p.numel() for p in model.parameters()) / 1e6:.2f} млн параметров; "
          f"сетка {tuple(model.patch_embed.grid_size)}; блоков {len(model.blocks)}; голов {model.blocks[0].attn.num_heads}; "
          f"нормировка {mean}/{std}; отпечаток начальных весов {fp(model):.4f}", flush=True)
    assert tuple(model.patch_embed.grid_size) == (14, 14) and len(model.blocks) == 12 and model.blocks[0].attn.num_heads == 3

    collage = args.cond in (4, 5)
    if collage:
        tr = Collage(Food(args.data, "train", None, args.limit), half_transform(mean, std, True), True)
        te = Collage(Food(args.data, "test", None, args.limit // 3 if args.limit else 0), half_transform(mean, std, False), False)
        te5 = Collage(Food(args.data, "test", None, args.limit // 3 if args.limit else 0, every=5), half_transform(mean, std, False), False)
        ncls = len(tr.base.cls)
        print(f"КОЛЛАЖ ({'метка = класс левого' if args.cond == 4 else 'метка = оба класса, 0,5 / 0,5'}); половина 224 × 112", flush=True)
    else:
        tr = Food(args.data, "train", transforms(args.cond, mean, std, True), args.limit)
        te = Food(args.data, "test", transforms(args.cond, mean, std, False), args.limit // 3 if args.limit else 0)
        te5 = Food(args.data, "test", transforms(args.cond, mean, std, False), args.limit // 3 if args.limit else 0, every=5)
        ncls = len(tr.cls)
    assert ncls == 101 or args.limit, f"классов {ncls}"
    print(f"Данные: обучение {len(tr)}, проверка {len(te)} (каждая пятая: {len(te5)}), классов {ncls}; аугментация: {tr.tf}", flush=True)
    assert args.batch % args.micro == 0, "пачка должна делиться на часть"
    accum = args.batch // args.micro
    print(f"Пачка {args.batch} = {accum} × {args.micro} (накопление градиента)", flush=True)
    g = torch.Generator().manual_seed(args.seed)
    dl = torch.utils.data.DataLoader(tr, batch_size=args.micro, shuffle=True, num_workers=args.workers, drop_last=True,
                                     generator=g, persistent_workers=args.workers > 0)
    dte = torch.utils.data.DataLoader(te, batch_size=args.micro, shuffle=False, num_workers=args.workers)
    dte5 = torch.utils.data.DataLoader(te5, batch_size=args.micro, shuffle=False, num_workers=args.workers)

    model.to(device).train()
    decay = [p for n, p in model.named_parameters() if p.ndim > 1 and "pos_embed" not in n and "cls_token" not in n]
    no_decay = [p for n, p in model.named_parameters() if not (p.ndim > 1 and "pos_embed" not in n and "cls_token" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05}, {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    steps_per_epoch = len(dl) // accum
    total = steps_per_epoch * args.epochs
    warm = steps_per_epoch * 5
    start_epoch, step = 0, 0
    last = os.path.join(out, "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch, step = ck["epoch"], ck["step"]
        torch.set_rng_state(ck["rng_cpu"])
        random.setstate(ck["rng_py"])
        np.random.set_state(ck["rng_np"])
        g.set_state(ck["rng_dl"])
        print(f"Продолжение с эпохи {start_epoch + 1} (шаг {step}) из {last}", flush=True)

    t_start = time.time()
    for epoch in range(start_epoch, args.epochs):
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
            if collage:
                x, ya, yb = (t.to(device, non_blocking=True) for t in batch)
                y = ya
                loss = F.cross_entropy(model(x), soft_target(args.cond, ya, yb, 101), label_smoothing=0.1)
            else:
                x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
                loss = F.cross_entropy(model(x), y, label_smoothing=0.1)
            (loss / accum).backward()
            lv = loss.item()
            if not math.isfinite(lv):
                print("⚠️ потери не число — остановка", flush=True)
                sys.exit(1)
            run_loss += lv / accum
            seen += len(y)
            if k < accum - 1:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            j = i // accum
            if (j + 1) % 50 == 0 or j + 1 == steps_per_epoch:
                el = time.time() - te0
                ips = seen / el
                left_h = ((args.epochs - epoch - 1) * steps_per_epoch * args.batch + (steps_per_epoch - j - 1) * args.batch) / ips / 3600
                print(f"  эпоха {epoch + 1}/{args.epochs}, шаг {j + 1}/{steps_per_epoch}: потери {run_loss / (j + 1):.3f}, "
                      f"lr {opt.param_groups[0]['lr']:.2e}, {ips:.0f} карт./с, осталось ≈ {left_h:.1f} ч", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        e1 = epoch + 1
        msg = f"Эпоха {e1} закончена: потери {run_loss / max(1, j + 1):.3f}, {time.time() - te0:.0f} с"
        if collage:
            l5, r5, b5 = evaluate_collage(model, dte5, device)
            acc5 = l5 if args.cond == 4 else b5
            msg += (f"; каждый пятый проверочный коллаж: top-1 = левый {l5 * 100:.2f}%, top-1 = правый {r5 * 100:.2f}%, "
                    f"оба в top-2 {b5 * 100:.2f}%")
            if e1 == args.epochs:
                la, ra, ba = evaluate_collage(model, dte, device)
                msg += f"; ВСЕ проверочные коллажи: левый {la * 100:.2f}%, правый {ra * 100:.2f}%, оба в top-2 {ba * 100:.2f}%"
        else:
            acc5 = evaluate(model, dte5, device)
            msg += f"; top-1 на каждой пятой проверочной фотографии {acc5 * 100:.2f}%"
            if e1 == args.epochs:
                acc = evaluate(model, dte, device)
                msg += f"; top-1 на ВСЕХ проверочных фото {acc * 100:.2f}%"
        print(msg + f"   ({time.strftime('%H:%M:%S')})", flush=True)
        if e1 == 3 and acc5 < 0.03 and not (args.limit or args.max_steps):
            print("⚠️ РАННИЙ СИГНАЛ: после эпохи 3 точность ниже 3% — сеть не учится. Останови (Ctrl+C) и пришли лог.", flush=True)
        if e1 in CKPT_EPOCHS or (args.max_steps and step >= args.max_steps):
            torch.save({"model": model.state_dict(), "epoch": e1, "cond": args.cond, "seed": args.seed}, os.path.join(out, f"ep{e1:02d}.pt"))
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": e1, "step": step,
                    "rng_cpu": torch.get_rng_state(), "rng_py": random.getstate(), "rng_np": np.random.get_state(),
                    "rng_dl": g.get_state()}, last + ".tmp")
        os.replace(last + ".tmp", last)
        if args.max_steps and step >= args.max_steps:
            print(f"Проверочный прогон остановлен после {step} шагов.", flush=True)
            break
    print(f"Готово: {time.time() - t_start:.0f} с в этом запуске; отпечаток весов {fp(model):.4f}; контрольные точки в {out}", flush=True)


if __name__ == "__main__":
    main()

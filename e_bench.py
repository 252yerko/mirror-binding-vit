"""
e_bench.py — замер скорости обучения малых ViT на этом компьютере (16.11, реестр 235). ЭТО НЕ ОПЫТ И НЕ РЕЗУЛЬТАТ.

ЗАЧЕМ: опыт E — обучение одинаковых малых ViT с нуля в нескольких условиях. Прежде чем выбирать размер сети,
число условий и зёрен, нужно знать, сколько картинок в секунду обучается на этом Маке.
ЧТО ДЕЛАЕТ: создаёт ViT-Tiny (192 измерения, 3 головы в блоке) и ViT-Small (384, 6 голов), 12 блоков, вход 224 px,
патч 16 (сетка 14 x 14, как во всех прежних опытах), и делает шаги обучения на СЛУЧАЙНЫХ картинках (шум):
обычный шаг (один проход) и шаг с двумя проходами (картинка и её отражение — как в условиях со штрафом).
Настоящие данные и обученные веса не используются; ничего не сохраняется, кроме лога.
ЗАПУСК: python -u e_bench.py 2>&1 | tee e_bench_log.txt   (~5–10 минут; строка печатается после каждого шага)
16.11 14:2x: исправлено — печать сразу (без буфера), пачка 64 вместо 128, ход виден по шагам; первая версия молчала до конца.
"""
import time
import torch
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer

import sys
QUICK = "--quick" in sys.argv                      # короткая проверка, что код работает (для теста на CPU)
dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 8 if QUICK else 64
WARM, STEPS = (1, 2) if QUICK else (3, 12)
N_IMAGES, EPOCHS = 126_000, 100                    # ориентир: набор из ~126 тыс. картинок, 100 эпох


def sync():
    if dev == "mps":
        torch.mps.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()


def bench(name, dim, heads, two_views):
    torch.manual_seed(0)
    net = VisionTransformer(img_size=224, patch_size=16, embed_dim=dim, depth=12, num_heads=heads,
                            num_classes=100).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
    x = torch.randn(BATCH, 3, 224, 224, device=dev)
    y = torch.randint(0, 100, (BATCH,), device=dev)
    net.train()
    print(f"  {name}, {'два прохода' if two_views else 'один проход'}: начало", flush=True)
    losses = []
    for i in range(WARM + STEPS):
        if i == WARM:
            sync()
            t0 = time.time()
        out = net(x)
        loss = F.cross_entropy(out, y)
        if two_views:
            out2 = net(torch.flip(x, dims=[3]))
            loss = loss + F.cross_entropy(out2, y) + (out - out2).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        print(f"    шаг {i + 1}/{WARM + STEPS}", flush=True)
    sync()
    dt = time.time() - t0
    ips = BATCH * STEPS / dt
    hours = N_IMAGES * EPOCHS / ips / 3600
    assert all(l == l for l in losses), "NaN в функции потерь"
    npar = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"  {name:9s} ({npar:4.1f} млн параметров), {'два прохода' if two_views else 'один проход':11s}: "
          f"{ips:7.1f} картинок/с; {N_IMAGES // 1000} тыс. картинок x {EPOCHS} эпох = {hours:6.1f} ч"
          f"   (потери: {losses[0]:.2f} -> {losses[-1]:.2f})", flush=True)
    del net, opt, x, y
    return ips


print(f"Замер скорости (не опыт): устройство {dev}, torch {torch.__version__}, пачка {BATCH}, "
      f"шагов {STEPS} после {WARM} прогревочных", flush=True)
res = {}
for name, dim, heads in (("ViT-Tiny", 192, 3), ("ViT-Small", 384, 6)):
    for two in (False, True):
        res[(name, two)] = bench(name, dim, heads, two)
print("Примечание: чтение и аугментация настоящих картинок добавят время сверх этого замера.")
print("Готово.")

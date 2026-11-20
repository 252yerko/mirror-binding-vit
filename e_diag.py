"""
e_diag.py — диагностика: почему e_train.py не печатает шаги (16.11, реестр 250).
Измеряет по отдельности: (A) чтение и аугментацию одного фото, (B) загрузчик с 0 и с 6 процессами,
(C) шаг обучения на MPS на случайных тензорах, (D) несколько настоящих шагов. Ничего не сохраняет.
Запуск (из папки binding-vit, venv активирован; ~2–5 мин):
  python -u e_diag.py 2>&1 | tee e_diag_log.txt
"""
import os
import sys
import time

import torch
import torch.nn.functional as F

from e_train import Food, transforms, make_model


def t():
    return time.time()


def main():
    data = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/binding-vit-data/food-101")
    dev = torch.device("mps") if torch.backends.mps.is_available() else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print(f"устройство {dev}; ядер CPU {os.cpu_count()}; потоков torch {torch.get_num_threads()}; torch {torch.__version__}", flush=True)
    tf = transforms(2, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), True)
    t0 = t()
    tr = Food(data, "train", tf)
    print(f"[0] список фото: {len(tr)} за {t() - t0:.1f} с", flush=True)

    # A: одно фото, без процессов
    t0 = t()
    for i in range(0, 40 * 997, 997):
        tr[i % len(tr)]
    a = (t() - t0) / 40
    print(f"[A] одно фото (чтение + аугментация), 1 процесс: {a * 1000:.0f} мс → ~{1 / a:.0f} фото/с на ядро", flush=True)

    # B0: загрузчик без процессов, 2 пачки по 128
    smp = lambda: torch.utils.data.RandomSampler(tr, replacement=True, num_samples=10 ** 6)
    dl0 = torch.utils.data.DataLoader(tr, batch_size=128, sampler=smp(), num_workers=0)
    t0 = t()
    it = iter(dl0)
    for _ in range(2):
        next(it)
    print(f"[B0] загрузчик, 0 процессов: {(t() - t0) / 2:.1f} с на пачку 128 → {256 / (t() - t0):.0f} фото/с", flush=True)

    # B6: загрузчик с 6 процессами (как в e_train.py), с тайм-аутом
    dl6 = torch.utils.data.DataLoader(tr, batch_size=128, sampler=smp(), num_workers=6, persistent_workers=True, timeout=300)
    t0 = t()
    it = iter(dl6)
    next(it)
    first = t() - t0
    print(f"[B6] загрузчик, 6 процессов: первая пачка через {first:.1f} с (запуск процессов)", flush=True)
    t0 = t()
    for _ in range(6):
        next(it)
    b6 = 6 * 128 / (t() - t0)
    print(f"[B6] следующие 6 пачек: {b6:.0f} фото/с", flush=True)

    # C: шаг обучения на случайных тензорах
    model = make_model(0, 101).to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    x = torch.randn(128, 3, 224, 224, device=dev)
    y = torch.randint(0, 101, (128,), device=dev)
    for k in range(6):
        if k == 1:
            if dev.type == "mps":
                torch.mps.synchronize()
            t0 = t()
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        loss.item()
    c = 5 * 128 / (t() - t0)
    print(f"[C] шаг обучения на случайных тензорах: {c:.0f} карт./с", flush=True)

    # D: настоящие шаги (загрузчик 6 процессов + обучение)
    t0 = t()
    for k in range(6):
        xb, yb = next(it)
        xb, yb = xb.to(dev), yb.to(dev)
        loss = F.cross_entropy(model(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        loss.item()
    d = 6 * 128 / (t() - t0)
    print(f"[D] настоящие шаги: {d:.0f} карт./с; потери {loss.item():.3f}", flush=True)
    print(f"ИТОГ: узкое место — {'загрузка фото' if b6 < c * 0.9 else 'видеокарта'}; "
          f"40 эпох при {d:.0f} карт./с ≈ {40 * 75750 / max(d, 1e-9) / 3600:.1f} ч", flush=True)


if __name__ == "__main__":
    main()

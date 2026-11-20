"""
e_diag2.py — диагностика 2 (16.11, реестр 251): почему шаг обучения на MPS даёт 6 карт./с, а e_bench.py — 98,8.
Отличия e_train от e_bench: пачка 128 (а не 64) и drop_path 0,1. Подозрение: при пачке 128 не хватает памяти
и Мак уходит в подкачку. Меряем шаг обучения на случайных тензорах при пачке 32, 64, 128, с drop_path и без,
и печатаем память. Загрузчика нет. Ничего не сохраняет. Запуск (venv активирован; ~3–6 мин):
  python -u e_diag2.py 2>&1 | tee e_diag2_log.txt
"""
import subprocess
import time

import timm
import torch
import torch.nn.functional as F

dev = torch.device("mps") if torch.backends.mps.is_available() else (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))


def sync():
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:
        return f"нет ({e})"


def swap_outs():
    s = sh("vm_stat | grep -i swapouts")
    try:
        return int(s.split(":")[1].strip().rstrip("."))
    except Exception:
        return -1


def run(batch, dpr, steps=5, warm=2, limit_s=240):
    torch.manual_seed(0)
    m = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=101, drop_path_rate=dpr).to(dev).train()
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=0.05)
    x = torch.randn(batch, 3, 224, 224, device=dev)
    y = torch.randint(0, 101, (batch,), device=dev)
    so0 = swap_outs()
    t_begin = time.time()
    done = 0
    for i in range(warm + steps):
        if i == warm:
            sync()
            t0 = time.time()
        loss = F.cross_entropy(m(x), y, label_smoothing=0.1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        loss.item()
        if i >= warm:
            done += 1
        if time.time() - t_begin > limit_s:
            print(f"    (остановлено по времени после {i + 1} шагов)", flush=True)
            break
    sync()
    ips = batch * done / max(time.time() - t0, 1e-9) if done else 0.0
    mem = torch.mps.driver_allocated_memory() / 2 ** 30 if dev.type == "mps" else float("nan")
    so1 = swap_outs()
    print(f"  пачка {batch:3d}, drop_path {dpr}: {ips:6.1f} карт./с; память MPS {mem:.2f} ГБ; "
          f"выгрузок в подкачку за замер {so1 - so0 if so0 >= 0 and so1 >= 0 else 'нет данных'}", flush=True)
    del m, opt, x, y
    if dev.type == "mps":
        torch.mps.empty_cache()
    return ips


if __name__ == "__main__":
    print(f"устройство {dev}; torch {torch.__version__}; timm {timm.__version__}", flush=True)
    print(f"Мак: {sh('sysctl -n machdep.cpu.brand_string')}; ОЗУ {int(sh('sysctl -n hw.memsize') or 0) / 2 ** 30:.0f} ГБ", flush=True)
    print(f"режим энергосбережения (lowpowermode): {sh('pmset -g | grep -i lowpowermode') or 'нет данных'}", flush=True)
    print(f"питание: {sh('pmset -g batt | head -2 | tail -1')}", flush=True)
    print(f"python-процессы сейчас: {sh('pgrep -fl python | grep -v e_diag2 | wc -l')}", flush=True)
    print(f"подкачка: {sh('sysctl -n vm.swapusage')}", flush=True)
    res = {}
    for b, d in ((32, 0.1), (64, 0.0), (64, 0.1), (128, 0.1)):
        res[(b, d)] = run(b, d)
    print(f"подкачка после: {sh('sysctl -n vm.swapusage')}", flush=True)
    best = max(res, key=res.get)
    print(f"ИТОГ: быстрее всего пачка {best[0]} (drop_path {best[1]}): {res[best]:.1f} карт./с → "
          f"40 эпох ≈ {40 * 75750 / max(res[best], 1e-9) / 3600:.1f} ч", flush=True)

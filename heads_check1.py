"""
heads_check1.py — разбор проверки 1 опыта F после запуска (реестр 182). НЕ правило и не часть регистрации gdnvb.

У DeiT III B/16 проверка 1 (наш прогон с явным вниманием = run_layers, макс. разница < 1e-3) дала 4,32e-03.
Вопрос: ошибка в коде или округление float32 на больших числах?
  (а) float64 на CPU: если наш прогон верен, разница с run_layers должна быть ~1e-10 и меньше;
  (б) float32 на устройстве прогона: абсолютная разница по слоям, наибольшее |h| слоя и относительная разница.
Запуск (из папки binding-vit): python heads_check1.py 2>&1 | tee heads_check1_log.txt   (~2 мин)
"""
import torch

import heads as H
import many_nets as mn
from extract import run_layers, pick_device

N_IMG = 4


@torch.no_grad()
def compare(model, x):
    a, b = H.forward(model, x), run_layers(model, x)
    rows = []
    for L, (u, v) in enumerate(zip(a, b)):
        d = float((u - v).abs().max())
        m = float(v.abs().max())
        rows.append((L, d, m, d / m))
    return rows


def main():
    import json
    meta = [json.loads(line) for line in open(f"{H.DATA}/meta.jsonl")][:N_IMG // 2]
    arrs = [H.load_u8(f"{H.DATA}/images/{r['id']}_{v}.png") for r in meta for v in ("S", "Sp")]
    device = pick_device()
    for key in ("dino", "deit3", "clip"):
        name, timm_name = H.NETS[key]
        model, w = mn.load(timm_name, None)
        _, _, n_pre, nb, mean, std = mn.setup(model, H.P)
        x = H.to_x(arrs, mean, std)
        # (а) float64, CPU
        m64 = model.double().cpu()
        r64 = compare(m64, x.double())
        worst64 = max(r[1] for r in r64)
        # (б) float32, устройство прогона
        m32 = model.float().to(device)
        r32 = compare(m32, x.to(device))
        worst = max(r32, key=lambda r: r[1])
        print(f"\n### {name}: {w}")
        print(f"  (а) float64 на CPU: макс. разница по всем слоям {worst64:.2e}  {'✅ код совпадает' if worst64 < 1e-8 else '⚠️ проверь'}")
        print(f"  (б) float32 на {device.type}: макс. разница {worst[1]:.2e} (слой L{worst[0]:02d}, где наибольшее |h| = {worst[2]:.1f}; "
              f"относительно {worst[3]:.1e}); по слоям: " + ", ".join(f"L{L:02d} {d:.1e}/{m:.0f}" for L, d, m, _ in r32))
        del model, m64, m32


if __name__ == "__main__":
    main()

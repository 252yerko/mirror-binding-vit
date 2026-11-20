"""
equiv_check2_fix.py — перепроверка проверки 2 опыта «Эквивариантность» (osf.io/syevu) для AugReg B/16 и L/16.
19.11 (реестр 298). В equiv_nets.py проверка 2 считала forward_features БЕЗ torch.no_grad (в many_nets.py опыта B — с ним),
а сам замер (passes) — с no_grad. Здесь та же проверка в обоих режимах: если с no_grad разница < 1e-3, а без — ≈ 1,7e-3 / 2,9e-3,
то провал — ошибка кода проверки, а не замера. Запуск: python -u equiv_check2_fix.py 2>&1 | tee equiv_check2_fix_log.txt
"""
import json
import torch
import many_nets as mn
from extract import run_layers, fingerprint, pick_device

DATA = mn.DATA
meta = [json.loads(l) for l in open(f"{DATA}/meta.jsonl")]
device = pick_device()
for name, tn, fam, size, P, part in mn.NETS:
    if name not in ("AugReg B/16", "AugReg L/16"):
        continue
    model, w = mn.load(tn, None)
    fp = fingerprint(model)
    ok, G, n_pre, nb, mean, std = mn.setup(model, P)
    model.to(device)
    x2 = torch.stack([mn.img(f"{DATA}/images/{meta[0]['id']}_{v}.png", mean, std) for v in ("S", "Sp")]).to(device)
    ref_g = model.forward_features(x2)
    mine_g = model.norm(run_layers(model, x2)[nb])
    with torch.no_grad():
        ref = model.forward_features(x2)
        mine = model.norm(run_layers(model, x2)[nb])
        ref_vs_refg = float((ref - ref_g).abs().max())
    d_ng = float((mine - ref).abs().max())
    d_g = float((mine_g.detach() - ref_g.detach()).abs().max())
    scale = float(ref.abs().max())
    print(f"{name}: отпечаток {fp:.4f}; с no_grad: разница {d_ng:.2e} {'✅' if d_ng < 1e-3 else '⚠️'}; "
          f"без no_grad (как в equiv_nets.py): {d_g:.2e}; forward_features с/без no_grad: {ref_vs_refg:.2e}; "
          f"макс. |признак| {scale:.1f} (относительная разница без no_grad {d_g / scale:.1e})", flush=True)
    del model

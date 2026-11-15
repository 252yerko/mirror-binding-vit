"""
heads_explore.py — разведочный разбор опыта F ПОСЛЕ запуска (реестр 183). НЕ правило и не часть регистрации gdnvb.

Зачем: в heads_m.csv записано только m = m_h − m_v. Сильно отрицательное m может значить и «голова смотрит в зеркало ↑↓»
(большое m_v), и «голова смотрит широко вдоль строки» (отрицательное m_h). Здесь m_h и m_v считаются по отдельности.
Второе: сходство позиционных кодов с зеркалом — «сырое» и за вычетом среднего по строке / столбцу, с 5 знаками.
Запуск (из папки binding-vit): python heads_explore.py 2>&1 | tee heads_explore_log.txt   (~10 мин)
"""
import csv

import numpy as np
import torch

import flip_test as ft
import heads as H
import many_nets as mn
from extract import pick_device


@torch.no_grad()
def hv_pass(model, arrs, device, mean, std, n_pre):
    MH = np.zeros((len(arrs), H.NB, H.NH))
    MV = np.zeros((len(arrs), H.NB, H.NH))
    for s in range(0, len(arrs), H.BATCH):
        x = H.to_x(arrs[s:s + H.BATCH], mean, std).to(device)
        cur = {}

        def cb(b, att, v, a):
            _, mh, mv = H.m_from_attn(att[:, :, n_pre:, n_pre:].float().cpu())
            cur[b] = (mh.numpy(), mv.numpy())

        H.forward(model, x, attn_cb=cb)
        for b in range(1, H.NB + 1):
            MH[s:s + len(x), b - 1], MV[s:s + len(x), b - 1] = cur[b]
    return MH.mean(0), MV.mean(0)


def pos_report(model):
    pe = model.pos_embed[0].detach().float().cpu()[-H.G * H.G:]
    pe = pe / pe.norm(dim=-1, keepdim=True)
    S = (pe @ pe.T).double().numpy()
    raw_h = S[H.VALID_H, H.MH[H.VALID_H]].mean()
    raw_v = S[H.VALID_V, H.MV[H.VALID_V]].mean()
    _, mh, mv = H.m_from_attn(torch.from_numpy(S)[None])
    return raw_h, raw_v, float(mh), float(mv)


def main(seed=None, limit=None):
    import json
    meta = [json.loads(line) for line in open(f"{H.DATA}/meta.jsonl")]
    sym = [r for r in meta if all(o["shape"] in ft.SYM_SHAPES for o in r["objects"])][:limit]
    arrs = [H.load_u8(f"{H.DATA}/images/{r['id']}_S.png") for r in sym]
    device = pick_device()
    rows = []
    u = 1.0 / 197
    for key in ("dino", "deit3", "clip"):
        name, timm_name = H.NETS[key]
        model, w = mn.load(timm_name, seed)
        _, _, n_pre, nb, mean, std = mn.setup(model, H.P)
        rh, rv, ph, pv = pos_report(model)
        model.to(device)
        MHm, MVm = hv_pass(model, arrs, device, mean, std, n_pre)
        m = MHm - MVm
        print(f"\n### {name}: {w}; картинок {len(arrs)}")
        print(f"  позиционные коды: косинус с зеркалом ←→ {rh:+.5f}, ↑↓ {rv:+.5f} (сырой); "
              f"за вычетом среднего по строке / столбцу: ←→ {ph:+.5f}, ↑↓ {pv:+.5f}")
        kinds = {"m_h ≥ u (зеркало ←→)": MHm >= u, "m_v ≥ u (зеркало ↑↓)": MVm >= u,
                 "m_h ≤ −u (широко вдоль строки)": MHm <= -u, "m_v ≤ −u (широко вдоль столбца)": MVm <= -u}
        for k, mask in kinds.items():
            hs = [(b + 1, h) for b in range(H.NB) for h in range(H.NH) if mask[b, h]]
            print(f"  {k}: {len(hs)} голов ({H.hs_str(hs) or '—'})")
        order = np.argsort(-np.abs(m).ravel())[:12]
        print("  12 голов с наибольшим |m| (блок.голова: m = m_h − m_v): " +
              "; ".join(f"{i // H.NH + 1}.{i % H.NH}: {m.ravel()[i]:+.4f} = {MHm.ravel()[i]:+.4f} − {MVm.ravel()[i]:+.4f}" for i in order))
        for b in range(H.NB):
            for h in range(H.NH):
                rows.append(dict(net=name, block=b + 1, head=h, m_h=MHm[b, h], m_v=MVm[b, h], m=m[b, h]))
        del model
    out = "heads_explore.csv" if seed is None and not limit else "heads_explore_test.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nЧисла записаны в {out}")


if __name__ == "__main__":
    main()

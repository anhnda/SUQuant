"""
§8 measurement pass for the MP trust-region: MEASURE BEFORE YOU BUILD.

Reads the cached saliency-weighted Hessians (and the n_eff files written next to
the saliency cache) and reports, per layer/group, the diagnostics that decide
whether MP is applicable at all:

  1. Whitened eigenvalue spectrum: is there a clean bulk (near sigma^2) + spikes
     (above lambda_+) separation? If the spectrum is a continuous smear with no
     gap, the spiked model is wrong and you should fall back to split-calibration
     rather than trusting the MP threshold.
  2. Condition number before/after: confirms H is PD but ill-conditioned
     (cond ~ 1e6-1e7), i.e. the bottom eigenspace really is the problem.
  3. n_eff, gamma_eff: how much saliency reweighting shrinks the sample size.
     gamma_eff >> gamma means very few directions survive.
  4. Number of BBP spikes above lambda_+: the proposed rank k. Sanity-check it is
     neither 0 nor near d.

Usage:
  python measure_mp.py --hessians cache/hessians/<...>_g1_mc4 \
                       --saliency cache/saliency/<...>_g1_mc4 \
                       [--layers 0 1 2] [--max_groups 1]

Nothing is modified; this only prints. Use the reported n_eff as AP_MP_NEFF when
you turn the trust region on.
"""
import argparse
import os
import torch
import functools
torch.load = functools.partial(torch.load, weights_only=False)

from any_precision.quantization.mp_trust_region import (
    _whiten, _bulk_sigma2, mp_edges, effective_sample_size,
)


def spectrum_report(H_g, gamma_eff):
    Hf = H_g.double()
    e_raw = torch.linalg.eigvalsh(Hf).clamp_min(0)
    cond_raw = (e_raw.max() / e_raw.clamp_min(1e-30).min()).item()

    Ht, _ = _whiten(Hf)
    ev = torch.linalg.eigvalsh(Ht).clamp_min(0)
    sigma2 = _bulk_sigma2(ev, gamma_eff)
    lam_minus, lam_plus = mp_edges(gamma_eff, sigma2)
    n_spike = int((ev > lam_plus).sum().item())

    # crude bulk/spike separation score: gap between lambda_+ and the largest
    # sub-edge eigenvalue vs the bulk width.
    below = ev[ev <= lam_plus]
    top_bulk = float(below.max().item()) if below.numel() else 0.0
    above = ev[ev > lam_plus]
    first_spike = float(above.min().item()) if above.numel() else float("nan")
    bulk_width = max(lam_plus - lam_minus, 1e-8)
    gap_ratio = (first_spike - lam_plus) / bulk_width if above.numel() else float("nan")

    return {
        "cond_raw": cond_raw,
        "sigma2": sigma2,
        "lam_plus": lam_plus,
        "n_spike": n_spike,
        "gap_ratio": gap_ratio,
        "d": Hf.shape[0],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hessians", required=True, help="hessians cache dir (has l{L}.pt)")
    ap.add_argument("--saliency", default=None, help="saliency cache dir (has l{L}_neff.pt)")
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--max_groups", type=int, default=4, help="report at most this many groups/layer")
    ap.add_argument("--n_fallback", type=float, default=None,
                    help="n_eff to use if no neff file is found")
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.hessians)
                   if f.startswith("l") and f.endswith(".pt") and "_neff" not in f)
    layer_ids = [int(f[1:-3]) for f in files]
    if args.layers:
        layer_ids = [l for l in layer_ids if l in args.layers]

    print(f"{'layer':>5} {'module':>16} {'grp':>3} {'d':>6} {'n_eff':>9} "
          f"{'gamma_eff':>9} {'sigma2':>8} {'lam+':>9} {'#spk':>5} {'gap':>7} {'cond_raw':>10}")
    print("-" * 100)

    for l in layer_ids:
        H_layer = torch.load(os.path.join(args.hessians, f"l{l}.pt"), map_location="cpu")
        neff_layer = None
        if args.saliency:
            neff_path = os.path.join(args.saliency, f"l{l}_neff.pt")
            if os.path.exists(neff_path):
                neff_layer = torch.load(neff_path, map_location="cpu")

        # H_layer: { module_name -> tensor (num_groups, d, d) } (or similar)
        for module_name, H in H_layer.items():
            if H is None:
                continue
            if H.dim() == 2:
                H = H.unsqueeze(0)
            num_groups = H.shape[0]
            for g in range(min(num_groups, args.max_groups)):
                # resolve n_eff for this module/group
                if neff_layer is not None and module_name in neff_layer \
                        and neff_layer[module_name] is not None:
                    ne_vec = neff_layer[module_name]
                    ne = float(ne_vec[min(g, ne_vec.numel() - 1)].item())
                elif args.n_fallback is not None:
                    ne = args.n_fallback
                else:
                    ne = 4.0 * H.shape[1]
                gamma_eff = H.shape[1] / max(ne, 1.0)
                r = spectrum_report(H[g], gamma_eff)
                print(f"{l:>5} {module_name[-16:]:>16} {g:>3} {r['d']:>6} {ne:>9.0f} "
                      f"{gamma_eff:>9.4f} {r['sigma2']:>8.3f} {r['lam_plus']:>9.3f} "
                      f"{r['n_spike']:>5} {r['gap_ratio']:>7.2f} {r['cond_raw']:>10.2e}")

    print("\nĐọc kết quả:")
    print("  - gap > 0 và #spk trong khoảng hợp lý (không 0, không ~d): MP áp được.")
    print("  - #spk ~ d hoặc gap < 0 / smear: phổ không tách -> dùng split-calibration.")
    print("  - Lấy n_eff cột trên (median qua layer) làm AP_MP_NEFF khi bật MP.")


if __name__ == "__main__":
    main()

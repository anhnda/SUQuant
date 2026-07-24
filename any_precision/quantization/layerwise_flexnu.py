"""
FlexNu solver for the GuidedQuant layer-wise objective.

Drop-in replacement for `train_least_squares` in layerwise_quantize.py.

WHAT CHANGES vs. the standalone FlexNu
--------------------------------------
The standalone FlexNu minimises  Tr(R G R^T)  with G = E[x x^T], a single Gram
shared by every output row of the layer. GuidedQuant (Eq. 7) minimises

    sum_k  sum_{j in J_k}  (w_j - w_hat_j)^T H_k (w_j - w_hat_j)

i.e. output channels are partitioned into g groups, and each group k has its own
saliency-weighted Hessian  H_k = X^T Diag(s_k) X,  where s_k is the squared
end-loss gradient averaged over the channels in J_k. So the objective is the
same functional with a *per-group* Gram instead of a single one.

The substitution is literal: G <- H_k for the rows in group k. Everything else
about FlexNu -- the sorted-codebook STE, the FlexRound divisor, the anchor +
softplus-gaps parameterisation, the snapshot-only best-iterate guard -- is
untouched.

Concretely this file differs from quantizers/flexnu.py in four places:

  1. `energy()` selects H[k] per row rather than using one shared G. Row blocks
     are aligned to group boundaries so a block never straddles two Hessians.
  2. Layout is [out, in] with FULL-WIDTH blocks (nb == 1). See "BLOCK SIZE" below.
  3. Init comes from GuidedQuant's cached SqueezeLLM labels/centroids, not from
     an internal k-means. This matters: SqueezeLLM's seed is weighted by the
     squared END-LOSS gradient per weight, which is strictly better information
     than diag(H_k).
  4. Returns (labels, C, log_dict) in exactly LNQ's format so the caller,
     packer, and Any-Precision kernel are unchanged.

BLOCK SIZE -- THE BINDING CONSTRAINT
------------------------------------
pack.py line 121 reads `layer_lut[name][r_idx][0]` with the comment
"the 0 here assumes group_count == 1". The Any-Precision kernel stores exactly
ONE codebook of 2**bit entries per output row, covering the whole input dim.
Standalone FlexNu defaults to block_size=64, which would produce in_features/64
codebooks per row and is NOT representable in this format.

So we run at block_size = in_features (nb = 1). This costs some codebook
expressiveness relative to standalone FlexNu but it is what LNQ and SqueezeLLM
already do here (see layerwise_quantize.py: `C` is [output_dim, n_cluster]), so
the comparison against LNQ is apples-to-apples and the bit-rate is identical.

MEMORY -- READ THIS BEFORE TUNING row_block
-------------------------------------------
The STE forward materialises a [row_block, in, K-1] tensor and autograd retains
a few of them. At in=11008, K=8 (3-bit), fp32:

    row_block=16  ->  16 * 11008 * 7 * 4B  = 4.9 MB per tensor
    row_block=128 ->  128 * 11008 * 7 * 4B = 39 MB per tensor

with ~4-6 live copies through the backward. That is fine. The 4-bit case
(K=16) roughly doubles it. Start at row_block=64 and raise it until the GPU
complains; it does not change the result, only peak memory (rows are
independent given their group's Hessian).

The Hessian itself is [g, in, in] fp32 -- 484 MB at in=11008, g=1. That is
already what LNQ loads, so no regression.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Straight-through step: forward 1[x > 0], backward sigmoid'(x/tau)/tau.
# --------------------------------------------------------------------------- #
class _StepSTE(torch.autograd.Function):
    """Hard Heaviside forward, logistic-bump backward.

    tau is BACKWARD-ONLY. It never enters the forward pass, so there is no
    forward/backward objective gap to anneal away and no temperature schedule.
    This is the concrete advantage over a softmax/Gumbel relaxation, where low
    temperature kills the gradient and high temperature optimises the wrong
    objective.
    """

    @staticmethod
    def forward(ctx, x, tau):
        ctx.save_for_backward(x)
        ctx.tau = tau
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        s = torch.sigmoid(x / ctx.tau)
        return grad_out * s * (1.0 - s) / ctx.tau, None


def _step_ste(x: torch.Tensor, tau: float) -> torch.Tensor:
    return _StepSTE.apply(x, tau)


# --------------------------------------------------------------------------- #
# Codebook parameterisation: c_0 = a, c_{j+1} = c_j + softplus(g_j)
# --------------------------------------------------------------------------- #
def _inv_softplus(y: torch.Tensor) -> torch.Tensor:
    """Stable inverse softplus: log(exp(y) - 1).

    Two regimes, or the init silently yields NaN:
      small y: exp(y)-1 underflows -> use log(expm1(y))
      large y: exp(y) overflows    -> use y + log(-expm1(-y))
    """
    y = y.clamp(min=1e-12)
    small = torch.log(torch.expm1(y.clamp(max=20.0)))
    large = y + torch.log((-torch.expm1(-y)).clamp(min=1e-30))
    return torch.where(y < 20.0, small, large)


def _codebook_from_params(anchor: torch.Tensor, gaps_raw: torch.Tensor) -> torch.Tensor:
    """[R,1] anchor + [R,K-1] raw gaps -> strictly increasing codebook [R,K].

    Sortedness is STRUCTURAL: softplus > 0 forces c_{j+1} > c_j at every point
    of training, with no sort, no projection, no constraint violation. This is
    what keeps the step-function reconstruction monotone (hence the STE valid),
    and what guarantees the committed LUT is strictly increasing -- which
    `searchsorted` at commit time requires.
    """
    deltas = torch.nn.functional.softplus(gaps_raw)
    return torch.cat([anchor, anchor + torch.cumsum(deltas, dim=-1)], dim=-1)


# --------------------------------------------------------------------------- #
# Objective: GuidedQuant Eq. (7), grouped.
# --------------------------------------------------------------------------- #
def _group_energy(Res: torch.Tensor, H_g: torch.Tensor) -> torch.Tensor:
    """sum_j r_j^T H_g r_j for a slab of rows sharing one Hessian.

    Res : [R, in]   residual W_hat - W
    H_g : [in, in]  that group's Hessian

    Equals Tr(Res H_g Res^T). Computed as (Res @ H_g * Res).sum() -- one
    [R,in] x [in,in] matmul, no [in,in] intermediate per row.

    NOTE ON SCALE: LNQ's objective_function() divides by num_groups (it takes
    .mean() over the group axis after einsum). We deliberately do NOT do that
    here, because we sum over row-blocks and the per-block totals must be
    additive. The absolute number is therefore g times LNQ's for g > 1;
    compare the RELATIVE drop, or use `objective_lnq_scale` below when you want
    a number directly comparable to LNQ's log output.
    """
    return (Res @ H_g * Res).sum()


@torch.no_grad()
def objective_lnq_scale(
    W: torch.Tensor,      # [out, in]
    H: torch.Tensor,      # [g, in, in]
    labels: torch.Tensor, # [out, in]
    C: torch.Tensor,      # [out, K]
) -> float:
    """Reproduce LNQ's objective_function() exactly, for log comparability.

    Use this -- not _group_energy -- whenever you want a number to put next to
    LNQ's "Objective:" lines. It applies the same reshape + mean-over-groups
    that layerwise_quantize.objective_function does.
    """
    W_hat = torch.gather(C.unsqueeze(1).expand(-1, labels.shape[1], -1),
                         dim=2, index=labels.unsqueeze(-1).long()).squeeze(-1)
    dW = W_hat - W
    g = H.shape[0]
    gs = W.shape[0] // g
    dW = dW.reshape(g, gs, dW.shape[-1])
    return torch.einsum('nij,njk,nik->i', dW, H, dW).mean().item()


# --------------------------------------------------------------------------- #
# Core: FlexNu on one slab of rows, all sharing a single group Hessian.
# --------------------------------------------------------------------------- #
def _optimize_rows(
    Wrows: torch.Tensor,        # [R, in]  fp32, on device
    H_g: torch.Tensor,          # [in, in] fp32, on device -- this group's Hessian
    init_labels: torch.Tensor,  # [R, in]  int64, on device
    init_C: torch.Tensor,       # [R, K]   fp32, on device
    *,
    iters: int,
    lr_scale: float,
    lr_cb: float,
    tau_frac: float,
    use_delta3: bool,
    freeze_codebook: bool,
    freeze_scale: bool,
    stage_frac: float,
    eval_every: int,
    lambda_s2: float,
    delta_init_noise: float = 0.0,   # <-- add
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """Returns (codebook [R,K], labels [R,in], e_init, e_best)."""
    R, din = Wrows.shape
    K = init_C.shape[-1]
    dev, wdt = Wrows.device, Wrows.dtype

    # ---- init from the cached SqueezeLLM codebook -------------------------
    # Enforce a strict minimum gap so inverse-softplus is well-conditioned.
    # SqueezeLLM centroids are already sorted per row in this repo, but a
    # degenerate (empty-cluster) level can produce a zero gap.
    cb0 = init_C.clone()
    cb0, _ = torch.sort(cb0, dim=-1)
    eps = 1e-6
    for k in range(1, K):
        cb0[:, k] = torch.maximum(cb0[:, k], cb0[:, k - 1] + eps)

    anchor0 = cb0[:, :1].clone()                       # [R,1]
    gaps0 = _inv_softplus(cb0[:, 1:] - cb0[:, :-1])    # [R,K-1]

    def energy(W_hat: torch.Tensor) -> torch.Tensor:
        return _group_energy(W_hat - Wrows, H_g)

    # ---- baseline: hard nearest-codeword on the init codebook -------------
    # We use searchsorted rather than the passed-in labels: the labels came
    # from SqueezeLLM's own (diagonal-Fisher) objective and need not be the
    # nearest-codeword assignment for cb0. Starting from nearest-codeword makes
    # `e_init` an honest floor, and matches what LNQ's initial objective means.
    with torch.no_grad():
        th0 = 0.5 * (cb0[:, 1:] + cb0[:, :-1])
        idx0 = torch.searchsorted(th0.contiguous(), Wrows.contiguous()).clamp_(0, K - 1)
        e_init = float(energy(torch.gather(cb0, 1, idx0)).item())

    n_steps = 0 if (freeze_codebook and freeze_scale) else int(iters)

    with torch.enable_grad():
        anchor = anchor0.detach().clone().requires_grad_(not freeze_codebook)
        gaps = gaps0.detach().clone().requires_grad_(not freeze_codebook)
        # delta2: element-wise log-divisor (FlexRound S2), init 0 -> divisor 1
        # delta2: element-wise log-divisor (FlexRound S2), init 0 -> divisor 1
        if delta_init_noise > 0:
            delta2 = (torch.randn(R, din, device=dev, dtype=wdt)
                      * delta_init_noise).requires_grad_(not freeze_scale)
        else:
            delta2 = torch.zeros(R, din, device=dev, dtype=wdt
                                 ).requires_grad_(not freeze_scale)        # delta3: per-output-channel log-divisor (FlexRound s3), init 0
        delta3 = (torch.zeros(R, 1, device=dev, dtype=wdt).requires_grad_(not freeze_scale)
                  if use_delta3 else None)

        learn_cb, learn_sc = not freeze_codebook, not freeze_scale
        _sp = [delta2] + ([delta3] if delta3 is not None else [])

        # DEFAULT IS JOINT (stage_frac = 0). Staging the divisor after the
        # codebook starts delta2 = 0 on a grid already fitted to W -- exactly
        # the nearest-neighbour point the divisor exists to escape -- and the
        # effect collapses (79.2% -> 13.1% in the paper's Table 3). Staging is
        # kept only for the ablation.
        opt_joint = opt_cb = opt_sc = None
        n_phase1 = 0
        if learn_cb and learn_sc and stage_frac <= 0.0:
            opt_joint = torch.optim.Adam([{"params": [anchor, gaps], "lr": lr_cb},
                                          {"params": _sp, "lr": lr_scale}])
        else:
            if learn_cb:
                opt_cb = torch.optim.Adam([anchor, gaps], lr=lr_cb)
            if learn_sc:
                opt_sc = torch.optim.Adam(_sp, lr=lr_scale)
            n_phase1 = (int(round(stage_frac * n_steps)) if (learn_cb and learn_sc)
                        else (n_steps if learn_cb else 0))

        # ---- best-iterate tracking: SNAPSHOT, never a constraint -----------
        # The STE gradient is exact for the SMOOTHED objective, not the
        # piecewise-constant true one, so iterates keep moving after the hard
        # energy bottoms out and momentum walks them uphill. The last iterate
        # is essentially never the best.
        #
        # SUBTLETY: escaping nearest-neighbour requires CROSSING a threshold,
        # and a crossing is a transient in which the smoothed objective
        # improves while the hard energy briefly worsens. A guard that rejects,
        # clips, or rewinds to such iterates filters out exactly the moves the
        # method depends on. So the optimiser runs completely unconstrained and
        # we only RECORD what it passes through. The SqueezeLLM init seeds the
        # incumbent, so the committed result is never worse than nearest-
        # codeword -- a floor on the OUTPUT, not a leash on the SEARCH.
        best_e = e_init
        best_state = None
        ev = max(1, int(eval_every))

        @torch.no_grad()
        def _hard_energy():
            cb_ = _codebook_from_params(anchor, gaps)
            th_ = 0.5 * (cb_[:, 1:] + cb_[:, :-1])
            ld_ = delta2 if delta3 is None else delta2 + delta3
            q_ = Wrows * torch.exp(-ld_)
            i_ = torch.searchsorted(th_.contiguous(), q_.contiguous()).clamp_(0, K - 1)
            return float(energy(torch.gather(cb_, 1, i_)).item())

        for step in range(n_steps):
            opt = opt_joint if opt_joint is not None else (
                opt_cb if step < n_phase1 else opt_sc)
            if opt is None:
                break
            opt.zero_grad(set_to_none=True)

            cb = _codebook_from_params(anchor, gaps)        # [R,K]
            deltas = cb[:, 1:] - cb[:, :-1]                 # [R,K-1] > 0
            thresh = 0.5 * (cb[:, 1:] + cb[:, :-1])         # [R,K-1]

            # tau tracks the codebook scale, recomputed each step so the
            # backward bump neither saturates nor smears as the gaps move.
            tau = float(tau_frac * deltas.detach().mean().clamp(min=1e-12))

            # FlexRound divisor: quantize against W / (S2 . s3) ...
            log_div = delta2 if delta3 is None else delta2 + delta3
            q = Wrows * torch.exp(-log_div)                 # [R,in]

            # ... but dequantize by the codebook alone, so W_hat need NOT be
            # the nearest codeword to W. That asymmetry is the whole method.
            ind = _step_ste(q.unsqueeze(-1) - thresh.unsqueeze(1), tau)   # [R,in,K-1]
            W_hat = cb[:, :1] + (ind * deltas.unsqueeze(1)).sum(dim=-1)   # [R,in]

            loss = energy(W_hat)
            if lambda_s2 > 0.0 and not freeze_scale:
                reg = (delta2 * delta2).sum()
                if delta3 is not None:
                    reg = reg + (delta3 * delta3).sum()
                loss = loss + lambda_s2 * reg

            loss.backward()
            opt.step()

            if (step % ev == 0) or (step == n_steps - 1):
                e_now = _hard_energy()
                if e_now < best_e:
                    best_e = e_now
                    best_state = (anchor.detach().clone(), gaps.detach().clone(),
                                  delta2.detach().clone(),
                                  None if delta3 is None else delta3.detach().clone())

        if best_state is not None:
            with torch.no_grad():
                anchor.copy_(best_state[0]); gaps.copy_(best_state[1])
                delta2.copy_(best_state[2])
                if delta3 is not None:
                    delta3.copy_(best_state[3])
        elif n_steps > 0:
            # Nothing ever beat the init: fall back to it exactly.
            with torch.no_grad():
                anchor.copy_(anchor0); gaps.copy_(gaps0); delta2.zero_()
                if delta3 is not None:
                    delta3.zero_()

    # ---- commit: hard assignment, divisors DISCARDED ----------------------
    with torch.no_grad():
        cb = _codebook_from_params(anchor, gaps).detach()
        thresh = 0.5 * (cb[:, 1:] + cb[:, :-1])
        log_div = delta2 if delta3 is None else delta2 + delta3
        q = Wrows * torch.exp(-log_div.detach())
        # searchsorted on the (structurally sorted) thresholds == argmin
        # |q - c_j|, but O(log K) and allocation-free.
        idx = torch.searchsorted(thresh.contiguous(), q.contiguous()).clamp_(0, K - 1)
        e_final = float(energy(torch.gather(cb, 1, idx)).item())

    return cb, idx, e_init, e_final


# --------------------------------------------------------------------------- #
# Public entry point: same signature/return as LNQ's train_least_squares.
# --------------------------------------------------------------------------- #
def train_flexnu(
    W: np.ndarray,               # [out, in]
    init_labels: np.ndarray,     # [out, in]
    init_centroids: np.ndarray,  # [out, K]
    H: np.ndarray,               # [g, in, in]
    *,
    iters: int = 300,
    lr_scale: float = 3e-3,
    lr_cb: float = 1e-5,
    tau_frac: float = 0.5,
    use_delta3: bool = True,
    freeze_codebook: bool = False,
    freeze_scale: bool = False,
    stage_frac: float = 0.0,
    eval_every: int = 1,
    row_block: int = 64,
    lambda_s2: float = 0.0,
    damp_hessian: bool = True,
    delta_init_noise: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """FlexNu solver for GuidedQuant Eq. (7).

    Signature and return value match layerwise_quantize.train_least_squares so
    this is a drop-in swap inside seed_layer().

    Returns (labels [out,in] int, C [out,K] fp32, log_dict).
    """
    device = torch.device("cuda")
    t0 = time.time()

    Wt = torch.tensor(W, dtype=torch.float32, device=device)
    Ht = torch.tensor(H, dtype=torch.float32, device=device)
    C0 = torch.tensor(init_centroids, dtype=torch.float32, device=device)
    L0 = torch.tensor(init_labels, dtype=torch.long, device=device)

    out_dim, din = Wt.shape
    g = Ht.shape[0]
    K = C0.shape[-1]
    assert out_dim % g == 0, f"out_dim {out_dim} not divisible by num_groups {g}"
    group_size = out_dim // g

    # ---- Hessian conditioning ---------------------------------------------
    # LNQ needs H positive DEFINITE because it Cholesky-factorises for the
    # closed-form codebook solve. FlexNu only ever uses H in the matmul
    # (Res @ H * Res).sum(), so PSD suffices and no factorisation is needed --
    # one fewer numerical failure mode, and the O(d^3) Cholesky disappears.
    #
    # We still add a tiny ridge by default: a semidefinite H has null
    # directions in which the residual is free, and the divisor will happily
    # wander there. Cheap insurance, off by damp_hessian=False.
    if damp_hessian:
        diag = torch.arange(din, device=device)
        for i in range(g):
            avg = torch.diagonal(Ht[i]).mean()
            Ht[i, diag, diag] += 1e-5 * avg

    labels_out = torch.empty(out_dim, din, dtype=torch.long, device=device)
    C_out = torch.empty(out_dim, K, dtype=torch.float32, device=device)

    e_init_tot = e_final_tot = 0.0
    log_dict = {"objective": [], "iteration": [], "group": [], "row_start": []}

    # ---- iterate over groups, then row-blocks WITHIN each group -----------
    # Row blocks must never straddle a group boundary, or a slab would need two
    # different Hessians. Nesting group-outer guarantees that.
    rb = max(1, min(int(row_block), group_size))
    for k in range(g):
        H_g = Ht[k]
        g_lo = k * group_size
        for r0 in range(g_lo, g_lo + group_size, rb):
            r1 = min(r0 + rb, g_lo + group_size)
            cb, idx, e_i, e_f = _optimize_rows(
                Wt[r0:r1], H_g, L0[r0:r1], C0[r0:r1],
                iters=iters, lr_scale=lr_scale, lr_cb=lr_cb, tau_frac=tau_frac,
                use_delta3=use_delta3, freeze_codebook=freeze_codebook,
                freeze_scale=freeze_scale, stage_frac=stage_frac,
                eval_every=eval_every, lambda_s2=lambda_s2,
                delta_init_noise=delta_init_noise
            )
            C_out[r0:r1] = cb
            labels_out[r0:r1] = idx
            e_init_tot += e_i
            e_final_tot += e_f
            log_dict["objective"].append(e_f)
            log_dict["iteration"].append(len(log_dict["objective"]))
            log_dict["group"].append(k)
            log_dict["row_start"].append(r0)
            del cb, idx
        torch.cuda.empty_cache()

    rel = (e_init_tot - e_final_tot) / max(e_init_tot, 1e-12)

    # ---- diagnostic: the fraction of weights that left nearest-codeword ----
    # THIS IS THE MEASUREMENT THE WHOLE CLAIM RESTS ON. If it is ~0 on real
    # layers, the escape is not firing and any perplexity change came from
    # somewhere else (probably the codebook path). Log it per module.
    with torch.no_grad():
        th = 0.5 * (C_out[:, 1:] + C_out[:, :-1])
        nn_idx = torch.searchsorted(th.contiguous(), Wt.contiguous()).clamp_(0, K - 1)
        moved = (nn_idx != labels_out).float().mean().item()

    obj_lnq = objective_lnq_scale(Wt, Ht, labels_out, C_out)

    logging.info(
        f"[flexnu] E_init={e_init_tot:.6e} E_final={e_final_tot:.6e} "
        f"drop={100.0 * rel:.2f}% | moved={100.0 * moved:.2f}% | "
        f"obj(LNQ-scale)={obj_lnq:.4f} | {time.time() - t0:.1f}s"
    )

    log_dict["energy_init"] = e_init_tot
    log_dict["energy_final"] = e_final_tot
    log_dict["energy_rel_drop"] = rel
    log_dict["moved_frac"] = moved
    log_dict["objective_lnq_scale"] = obj_lnq

    labels_np = labels_out.detach().cpu().numpy().astype(np.uint8)
    C_np = C_out.detach().cpu().numpy().astype(np.float32)
    del Wt, Ht, C0, L0, labels_out, C_out
    torch.cuda.empty_cache()

    return labels_np, C_np, log_dict

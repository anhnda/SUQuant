"""
LNQ-F : LNQ with the first-order (linear) term of the end-loss retained.

WHY THIS EXISTS
---------------
GuidedQuant / LNQ minimise the *second-order* term of the Taylor expansion of
the change in end loss:

        Delta_l  ~=  1/2 (w_hat - w)^T H (w_hat - w)                       (B2)

with H = X^T Diag(s) X the saliency-weighted (Fisher) Hessian. The first-order
term  g_bar^T (w_hat - w)  is dropped under the assumption that the pretrained
model has converged, so the *mean* gradient g_bar = (1/n) sum_i grad_l_i is ~ 0.

That assumption is only approximately true on a small, distribution-shifted
calibration set, and the residual first-order term is the part that end-to-end
fine-tuning is later observed to recover (GuidedQuant Table 15, where the gap
narrows at higher bit-width). LNQ-F keeps it:

        phi(w_hat) = g_bar^T (w_hat - w) + 1/2 (w_hat - w)^T H (w_hat - w)  (B1+B2)

COMPLETE-THE-SQUARE  ->  TARGET SHIFT
-------------------------------------
phi is a quadratic in w_hat with the SAME curvature H. Completing the square,

        phi(w_hat)  ==  1/2 (w_hat - w_tilde)^T H (w_hat - w_tilde) + const,
        w_tilde = w - H^{-1} g_bar          (one Newton step against H).

So retaining the first-order term is EXACTLY the plain B2 objective with the
reconstruction target moved from w to w_tilde. Every piece of the LNQ machine
(Cholesky, closed-form codebook Eq. 9, cyclic-CD assignment, Prop. 4.1
monotone-descent guarantee) is reused verbatim on w_tilde. Nothing in the solver
changes; only the target it fits.

DAMPING (why H^{-1} g_bar is not used raw)
------------------------------------------
H is typically ill-conditioned (Diag(s) kills directions where the loss is flat),
and g_bar is a noisy average over a small calibration set. A raw H^{-1} g_bar
amplifies that noise along the small-eigenvalue directions and can push w_tilde
far from w, which is exactly where the second-order Taylor model stops being
accurate. We therefore use a damped (Levenberg-style) Newton step

        w_tilde = w - (H + mu * avg_diag(H) * I)^{-1} g_bar,

controlled by `mu`:

        mu -> inf  =>  w_tilde -> w        =>  LNQ-F degenerates to plain LNQ (B2)
        mu -> 0    =>  full Newton step    =>  strongest first-order correction

`mu` is thus a *continuous knob* between "second-order only" (the published LNQ)
and "first + second order". A trust-region cap `max_shift` additionally clamps
the per-row L2 norm of the shift so a bad row cannot be dragged out of the region
where B2 is valid.

WHAT g_bar HAS TO BE
--------------------
Per output channel j, the first-order term of Remark 3.1 is
    g_bar_j = (1/n) sum_i (d l_i / d z_ij) * X_{i,:}    in R^{d_in},
i.e. the SIGNED end-loss gradient w.r.t. the layer output, contracted with the
inputs. NOTE the sign: the cached GuidedQuant saliency s = (d l/d z)^2 has thrown
the sign away, so it CANNOT be reused for this term. A signed first-order cache
must be supplied separately (see get_firstorder in gradients_firstorder.py). When
no first-order cache is found, LNQ-F falls back to plain LNQ and says so in the
log -- it never silently fabricates a shift.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch

from .layerwise_quantize import train_least_squares


@torch.no_grad()
def _damped_newton_shift(
    W: torch.Tensor,        # [out, in]     original weights (target)
    H: torch.Tensor,        # [g, in, in]   saliency-weighted Hessian blocks
    g_bar: torch.Tensor,    # [out, in]     signed first-order term, per row
    mu: float,
    max_shift: float,
) -> torch.Tensor:
    """
    Return the shifted target  w_tilde = w - (H + mu*avg_diag*I)^{-1} g_bar,
    solved block-wise (one Hessian block per output-channel group), with an
    optional per-row L2 trust-region clamp `max_shift` (<=0 disables it).
    """
    device = W.device
    out_dim, in_dim = W.shape
    num_groups = H.shape[0]
    assert out_dim % num_groups == 0, "out_dim must be divisible by num_groups"
    group_size = out_dim // num_groups

    diag_idx = torch.arange(in_dim, device=device)
    shift = torch.empty_like(W)

    for gi in range(num_groups):
        Hi = H[gi].clone()
        avg_diag = Hi[diag_idx, diag_idx].mean()
        Hi[diag_idx, diag_idx] += mu * avg_diag           # Levenberg damping

        # rows (output channels) that belong to this group
        r0, r1 = gi * group_size, (gi + 1) * group_size
        rhs = g_bar[r0:r1].to(device).T                    # [in, group_size]

        # (H+muI) delta = g_bar   ->   delta = solve(...)
        L = torch.linalg.cholesky(Hi)
        delta = torch.cholesky_solve(rhs, L).T             # [group_size, in]
        shift[r0:r1] = delta

    if max_shift and max_shift > 0:
        row_norm = shift.norm(dim=1, keepdim=True).clamp_min(1e-12)
        scale = (max_shift / row_norm).clamp_max(1.0)
        shift = shift * scale

    return W - shift                                       # w_tilde


def train_least_squares_firstorder(
    W: np.ndarray,               # [out, in]
    init_labels: np.ndarray,     # [out, in]   SqueezeLLM init
    init_centroids: np.ndarray,  # [out, K]    SqueezeLLM init
    H: np.ndarray,               # [g, in, in] saliency-weighted Hessian
    *,
    num_iterations: int = 3,
    cd_cycles: int = 4,
    g_bar: Optional[np.ndarray] = None,   # [out, in] signed first-order term
    mu: float = 1.0,
    max_shift: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    LNQ-F. If g_bar is None -> identical to plain LNQ (train_least_squares) but
    tagged in the log. Otherwise, shift the reconstruction target by a damped
    Newton step and hand the SAME LNQ solver the shifted target.
    """
    if g_bar is None:
        logging.warning(
            "[lnqf] no first-order term supplied (g_bar=None) -> "
            "falling back to plain LNQ (second-order only). "
            "Provide a signed first-order cache to activate the B1 correction."
        )
        labels, C, log_dict = train_least_squares(
            W, init_labels, init_centroids, H,
            num_iterations=num_iterations, cd_cycles=cd_cycles,
        )
        log_dict = {"lnqf_active": False, **({} if log_dict is None else log_dict)}
        return labels, C, log_dict

    device = torch.device("cuda")
    W_t = torch.tensor(W, dtype=torch.float32, device=device)
    H_t = torch.tensor(H, dtype=torch.float32, device=device)
    g_t = torch.tensor(g_bar, dtype=torch.float32, device=device)

    W_tilde = _damped_newton_shift(W_t, H_t, g_t, mu=mu, max_shift=max_shift)

    shift_norm = (W_t - W_tilde).norm().item()
    w_norm = W_t.norm().item()
    logging.info(
        f"[lnqf] first-order target shift active | mu={mu:.3g} "
        f"max_shift={max_shift:.3g} | ||w_tilde - w|| = {shift_norm:.4e} "
        f"({100.0 * shift_norm / max(w_norm, 1e-12):.3f}% of ||w||)"
    )

    # Fit the SHIFTED target with the unchanged LNQ solver. Prop. 4.1 holds
    # because this is still a plain quadratic in (w_hat - w_tilde) with PD H.
    W_tilde_np = W_tilde.detach().cpu().numpy().astype(np.float32)
    labels, C, log_dict = train_least_squares(
        W_tilde_np, init_labels, init_centroids, H,
        num_iterations=num_iterations, cd_cycles=cd_cycles,
    )

    log_dict = {
        "lnqf_active": True,
        "lnqf_mu": mu,
        "lnqf_max_shift": max_shift,
        "lnqf_shift_norm": shift_norm,
        "lnqf_shift_frac": shift_norm / max(w_norm, 1e-12),
        **({} if log_dict is None else log_dict),
    }
    return labels, C, log_dict

"""
LNQ-F : LNQ with the first-order (linear) end-loss term retained.

DESIGN (option A -- NO global H^{-1})
-------------------------------------
GuidedQuant / LNQ minimise only the SECOND-order term of the change in end loss:

        Delta_l ~= 1/2 (w_hat - w)^T H (w_hat - w).                        (B2)

The first-order term  g_bar^T (w_hat - w)  is dropped under the "converged model
=> mean gradient ~ 0" assumption. On a small, distribution-shifted calibration
set that residual is real (most LLM tokens are not saturated, so d l/d z != 0),
and it is what end-to-end fine-tuning is later seen to recover. LNQ-F keeps it:

        phi(w_hat) = g_bar^T (w_hat - w) + 1/2 (w_hat - w)^T H (w_hat - w). (B1+B2)

We do NOT implement this via a Newton target shift  w_tilde = w - H^{-1} g_bar.
That would require inverting the (ill-conditioned) full H, blow up along its
small-eigenvalue directions, and need damping/trust-region band-aids. Instead we
fold g_bar DIRECTLY into LNQ's two closed-form update steps, exactly as derived:

  * Codebook (given P):   c = (P^T H P)^{-1} (P^T H w - P^T g_bar).
        The inverse is the SAME small (m x m) P^T H P as vanilla LNQ; the only
        change is subtracting P^T g_bar from the normal-equation RHS. In this
        implementation the RHS enters through b = L^T w with H = L L^T, so we
        subtract z = L^{-1} g_bar (ONE triangular solve with the already-computed
        Cholesky factor -- not a full H^{-1}).

  * Assignment (given c):  per-coordinate CD update becomes
        w_hat_i <- Round( w_i - (1/H_ii) ( g_bar_i + sum_{k!=i} H_ik (w_hat_k - w_k) ) ).
        This adds only  -g_bar_i / H_ii  inside the round -- a division by the
        SCALAR diagonal H_ii. No matrix inverse at all. Concretely this is a
        per-coordinate target shift by g_bar / diag(H), applied to the rounding
        target only; the descent term B still measures the true (w_hat - w).

Both steps remain exact minimizers of phi along their block/coordinate, so LNQ's
Prop. 4.1 monotone-descent + convergence guarantee carries over unchanged. When
g_bar is None, every _fo function reduces bit-for-bit to the vanilla LNQ step.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import numpy as np
import torch

from .utils import get_progress_bar
# objective_function is imported lazily inside train_least_squares_firstorder to
# avoid a circular import (layerwise_quantize imports this module at top level).


@torch.no_grad()
def update_P_fo(
    W: torch.Tensor,        # (out, in)
    H: torch.Tensor,        # (g, in, in)
    labels: torch.Tensor,   # (out, in)
    C: torch.Tensor,        # (out, n_cluster)
    cd_cycles: int,
    g_over_diag: Optional[torch.Tensor] = None,  # (out, in) = g_bar / diag(H), per row
    verbose: bool = True,
):
    """
    Cyclic-CD assignment update with the first-order term folded in.

    Identical to layerwise_quantize.update_P, except the ROUNDING target is
    w_i - g_bar_i/H_ii - B_i  instead of  w_i - B_i. B (the descent correction)
    still uses the true (W_hat - W). Setting g_over_diag=None reproduces vanilla.
    """
    device = torch.device("cuda")
    C = C.to(device)
    assignments_prev = labels.to(device).long()
    b, d = assignments_prev.shape
    num_groups = H.shape[0]
    group_size = W.shape[0] // num_groups
    assert W.shape[0] % num_groups == 0

    assignments = assignments_prev.clone()
    update_size = cd_cycles * d
    W_hat = torch.gather(C.unsqueeze(1).expand(-1, d, -1), dim=2,
                         index=assignments.unsqueeze(-1)).squeeze(-1)

    pb = get_progress_bar(update_size, "Updating P inside [fo]")

    W_grp = W.reshape(num_groups, group_size, d)
    C_grp = C.reshape(num_groups, group_size, C.shape[-1])
    W_hat_grp = W_hat.reshape(num_groups, group_size, d)
    H_grp = H.clone().to(device)
    B_grp = torch.zeros_like(W_grp).to(device)

    # rounding target = W - g_bar/diag(H); descent term B still uses true W.
    if g_over_diag is not None:
        Wtgt_grp = (W - g_over_diag.to(device)).reshape(num_groups, group_size, d)
    else:
        Wtgt_grp = W_grp

    for i in range(num_groups):
        H_grp_diag = H_grp[i, torch.arange(d), torch.arange(d)].reshape(1, 1, -1)
        H_grp[i, :, :] = H_grp[i, :, :] / H_grp_diag

    cd_block_size = 128

    for k in range(cd_cycles):
        B_grp = torch.bmm(W_hat_grp - W_grp, torch.tril(H_grp, diagonal=-1))
        for start_idx in range(0, d, cd_block_size):
            end_idx = min(start_idx + cd_block_size, d)
            for update_idx in range(start_idx, end_idx):
                index = torch.arange(update_idx, update_idx + 1, device=device)
                # ---- only change vs vanilla: Wtgt_grp instead of W_grp ----
                sol = Wtgt_grp[:, :, index] - B_grp[:, :, index]
                sol_dist = torch.abs(sol - C_grp)
                min_dist, argmin_dist = sol_dist.min(dim=-1)
                assignments[:, index] = argmin_dist.reshape(-1, 1)
                W_hat_grp[:, :, index] = torch.gather(C_grp, dim=-1, index=argmin_dist.unsqueeze(-1))
                if update_idx < end_idx - 1:
                    B_grp[:, :, update_idx + 1:end_idx] += torch.bmm(
                        W_hat_grp[:, :, index] - W_grp[:, :, index],
                        H_grp[:, index, update_idx + 1:end_idx])
                pb.update(1)
            B_grp[:, :, end_idx:] += torch.bmm(
                W_hat_grp[:, :, start_idx:end_idx] - W_grp[:, :, start_idx:end_idx],
                H_grp[:, start_idx:end_idx, end_idx:])
    pb.close()

    num_changed = (assignments_prev != assignments).sum().item()
    pct = num_changed / assignments_prev.numel() * 100
    if verbose:
        logging.info(f"Percentage of assignments changed: {pct:.2f}%")
    return assignments


@torch.no_grad()
def update_C_fo(
    W: torch.Tensor,        # (out, in)
    H: torch.Tensor,        # (g, in, in)
    labels: torch.Tensor,   # (out, in)
    C: torch.Tensor,        # (out, n_cluster)
    g_bar: Optional[torch.Tensor] = None,  # (out, in)
):
    """
    Closed-form codebook update with the first-order term folded in.

    Solves, per output channel, the SAME regularized least squares as
    layerwise_quantize.update_C but with RHS  b = L^T w - L^{-1} g_bar, so the
    normal equations read  (P^T H P) c = P^T H w - P^T g_bar  (Eq. 9 + B1).
    The (m x m) system solved by lstsq is unchanged; only b shifts. The L^{-1}
    is a single triangular solve with the already-computed Cholesky factor L,
    NOT a full H^{-1}. g_bar=None reproduces vanilla update_C exactly.
    """
    device = torch.device("cuda")
    channel_size = W.shape[0]
    input_size = H.shape[1]
    sub_channel_size = 64
    sub_input_size = 2 ** 16
    num_groups = H.shape[0]
    group_size = W.shape[0] // num_groups

    L = torch.empty_like(H)
    for i in range(num_groups):
        L[i] = torch.linalg.cholesky(H[i])
    reduced_X = L.transpose(-2, -1)   # = L^T

    # z_group = L^{-1} g_bar, per group, precomputed via triangular solve.
    # g_bar is (out, in); rows share the group's L. Solve L z^T = g_bar^T.
    z_all = None
    if g_bar is not None:
        z_all = torch.empty_like(W)   # (out, in), holds (L^{-1} g_bar) per row
        for gi in range(num_groups):
            r0, r1 = gi * group_size, (gi + 1) * group_size
            rhs = g_bar[r0:r1].to(device).transpose(0, 1)          # (in, group_size)
            zi = torch.linalg.solve_triangular(L[gi], rhs, upper=False)  # (in, group_size)
            z_all[r0:r1] = zi.transpose(0, 1)

    assert channel_size // sub_channel_size >= num_groups
    assert channel_size % (sub_channel_size * num_groups) == 0

    C_hat_list = []
    pb = get_progress_bar(channel_size // sub_channel_size, "Updating centroids [fo]")
    for st_idx in range(0, channel_size, sub_channel_size):
        group_idx = st_idx // group_size
        reduced_X_blk = reduced_X[group_idx]
        end_idx = min(st_idx + sub_channel_size, channel_size)

        A_batch_list, b_batch_list = [], []
        labels_batch = labels[st_idx:end_idx].to(device)
        for st_idx_inp in range(0, input_size, sub_input_size):
            end_idx_inp = min(st_idx_inp + sub_input_size, input_size)
            X_batch = reduced_X_blk[st_idx_inp:end_idx_inp].to(device)   # (rows_in, in) slice of L^T
            P_batch = torch.nn.functional.one_hot(labels_batch.long(), num_classes=C.shape[-1]).float()
            A_batch_tmp = torch.einsum('bj,ijc->ibc', X_batch, P_batch)
            b_batch_tmp = torch.einsum('bj,ij->ib', X_batch, W[st_idx:end_idx]).unsqueeze(-1)
            A_batch_list.append(A_batch_tmp)
            b_batch_list.append(b_batch_tmp)

        A_batch = torch.cat(A_batch_list, dim=1)
        b_batch = torch.cat(b_batch_list, dim=1)   # = L^T w

        # ---- first-order shift of the RHS:  b <- b - L^{-1} g_bar ----
        if z_all is not None:
            # z_all rows are (L^{-1} g_bar); it lives in the same reduced space as b.
            b_shift = z_all[st_idx:end_idx].to(device).unsqueeze(-1)   # (out_blk, in, 1)
            b_batch = b_batch - b_shift

        ######### REGULARIZATION (unchanged) #########
        lambda_reg = 1e-7
        batch_size, num_samples, n_cluster = A_batch.shape
        dtype, device2 = A_batch.dtype, A_batch.device
        sqrt_lambda = torch.sqrt(torch.tensor(lambda_reg, dtype=dtype, device=device2))
        I = sqrt_lambda * torch.eye(n_cluster, dtype=dtype, device=device2).unsqueeze(0).expand(batch_size, -1, -1)
        A_batch = torch.cat([A_batch.transpose(1, 2), I], dim=2).transpose(1, 2)
        zeros = torch.zeros((batch_size, n_cluster, 1), dtype=dtype, device=device2)
        b_batch = torch.cat([b_batch, zeros], dim=1)
        ##############################################

        C_hat_batch = torch.linalg.lstsq(A_batch, b_batch).solution
        if torch.isnan(C_hat_batch).any():
            logging.error(f"NaN in C_hat_batch for indices {st_idx}:{end_idx}")
            exit()
        C_hat_list.append(C_hat_batch.squeeze(-1))
        pb.update(1)
    pb.close()

    return torch.cat(C_hat_list, dim=0).cpu()


def train_least_squares_firstorder(
    W: np.ndarray,               # (out, in)
    init_labels: np.ndarray,     # (out, in)
    init_centroids: np.ndarray,  # (out, K)
    H: np.ndarray,               # (g, in, in)
    *,
    num_iterations: int = 3,
    cd_cycles: int = 4,
    g_bar: Optional[np.ndarray] = None,   # (out, in) signed first-order term
    mu: float = 0.0,          # kept for CLI compatibility; unused in option A
    max_shift: float = 0.0,   # kept for CLI compatibility; unused in option A
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    LNQ-F alternating minimization: update_P_fo / update_C_fo with g_bar folded
    directly into both steps. NO global H^{-1}. g_bar=None -> vanilla LNQ.
    """
    from .layerwise_quantize import objective_function  # lazy: breaks import cycle

    device = torch.device("cuda")

    labels = torch.tensor(init_labels, dtype=torch.int8, device="cpu")
    C = torch.tensor(init_centroids, dtype=torch.float32, device="cpu")
    W = torch.tensor(W, dtype=torch.float32).to(device)
    H = torch.tensor(H, dtype=torch.float32).to(device)

    g_bar_t = None
    g_over_diag = None
    if g_bar is not None:
        g_bar_t = torch.tensor(g_bar, dtype=torch.float32, device=device)
        # per-coordinate diagonal shift  g_bar / diag(H); each row uses its group's H.
        num_groups = H.shape[0]
        group_size = W.shape[0] // num_groups
        d = H.shape[1]
        g_over_diag = torch.empty_like(g_bar_t)
        for gi in range(num_groups):
            r0, r1 = gi * group_size, (gi + 1) * group_size
            diagH = H[gi, torch.arange(d), torch.arange(d)].clamp_min(1e-12)  # (in,)
            g_over_diag[r0:r1] = g_bar_t[r0:r1] / diagH.unsqueeze(0)
        logging.info("[lnqf] first-order term folded into update_P/update_C (option A, no global H^{-1}).")
    else:
        logging.warning("[lnqf] g_bar=None -> vanilla LNQ (second-order only).")

    # dampen H to PD (identical to vanilla train_least_squares)
    diag = torch.arange(H.shape[1], device=device)
    for i in range(H.shape[0]):
        avg_diag = torch.mean(torch.diag(H[i]))
        damp, prev_damp = 1e-5, 0.
        while True:
            try:
                torch.linalg.cholesky(H[i])
                logging.info(f"{i+1}-th H is PD, dampening factor={prev_damp:.2e}")
                break
            except Exception as e:
                logging.info(f"{i+1}-th H not PD, dampening factor={damp:.2e}")
                H[i, diag, diag] += (damp - prev_damp) * avg_diag
                prev_damp = damp
                damp *= 10
                if damp > 1e0:
                    exit()

    # NOTE: objective_function measures pure B2 (delta_w^T H delta_w); with the
    # first-order term the true objective is phi = B2 + g_bar^T delta_w. We log
    # the B2 part for comparability with vanilla LNQ, plus the linear part.
    def full_obj(labels_, C_):
        b2 = objective_function(W, H, labels_, C_).item()
        if g_bar_t is None:
            return b2, b2, 0.0
        Wq = torch.gather(C_.to(device).unsqueeze(1).expand(-1, W.shape[1], -1),
                          2, labels_.to(device).long().unsqueeze(-1)).squeeze(-1)
        lin = (g_bar_t * (Wq - W)).sum().item() / W.shape[0]  # mean over rows, matches B2 scale
        return b2 + lin, b2, lin

    best_phi, best_b2, best_lin = full_obj(labels, C)
    best_labels, best_C = labels.clone(), C.clone()
    logging.info(f"Initial: phi={best_phi:.4f} (B2={best_b2:.4f}, lin={best_lin:.4f})")

    log_dict = {"objective": [best_phi], "iteration": [0], "lnqf_active": g_bar is not None}

    for iteration in range(num_iterations):
        t0 = time.time()

        if iteration > 0:
            labels = update_P_fo(W, H, labels, C, cd_cycles=cd_cycles, g_over_diag=g_over_diag)
        phi_p, b2_p, lin_p = full_obj(labels, C)
        logging.info(f"Iter {iteration+1} (P): phi={phi_p:.4f} (B2={b2_p:.4f}, lin={lin_p:.4f})")

        C = update_C_fo(W, H, labels, C, g_bar=g_bar_t)
        phi_c, b2_c, lin_c = full_obj(labels, C)
        log_dict["objective"].append(phi_c); log_dict["iteration"].append(iteration + 1)

        if phi_c < best_phi:
            best_phi, best_b2, best_lin = phi_c, b2_c, lin_c
            best_labels, best_C = labels.clone(), C.clone()
            logging.info(f"Iter {iteration+1} (C): phi={phi_c:.4f} (B2={b2_c:.4f}, lin={lin_c:.4f}) | improved.")
        else:
            logging.info(f"Iter {iteration+1} (C): phi={phi_c:.4f} | not improved, keeping best.")
            labels, C = best_labels, best_C
            break

        logging.info(f"Iter {iteration+1}/{num_iterations} done in {time.time()-t0:.2f}s")

    labels = best_labels.detach().cpu().numpy().astype(np.float32)
    C = best_C.detach().cpu().numpy().astype(np.float32)
    return labels, C, log_dict
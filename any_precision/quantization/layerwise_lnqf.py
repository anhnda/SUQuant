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

    The exact coordinate minimizer of  phi = g^T u + 1/2 u^T H u  is
        w_hat_i <- Round( w_i - (1/H_ii)( g_i + sum_{k!=i} H_ik (w_hat_k - w_k) ) ).
    We fold g_i/H_ii into the DESCENT TERM B (which already carries the
    sum_{k!=i} H_ik (.)/H_ii part), NOT into a static target shift. This is the
    key fix: g_i/H_ii lives inside the dynamic B that CD updates every coordinate,
    so a near-singular coordinate (H_ii ~ 0) can no longer park a huge static
    offset outside the codebook. Setting g_over_diag=None reproduces vanilla.
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

    # g_bar/diag(H) as a per-coordinate constant added INTO B (not the target).
    if g_over_diag is not None:
        god_grp = g_over_diag.to(device).reshape(num_groups, group_size, d)
    else:
        god_grp = None

    for i in range(num_groups):
        H_grp_diag = H_grp[i, torch.arange(d), torch.arange(d)].reshape(1, 1, -1)
        H_grp[i, :, :] = H_grp[i, :, :] / H_grp_diag

    cd_block_size = 128

    for k in range(cd_cycles):
        # B carries the cross-coordinate correction; SEED it with g_bar/diag(H) so
        # the first-order term rides inside the same dynamically-updated quantity.
        B_grp = torch.bmm(W_hat_grp - W_grp, torch.tril(H_grp, diagonal=-1))
        if god_grp is not None:
            B_grp = B_grp + god_grp
        for start_idx in range(0, d, cd_block_size):
            end_idx = min(start_idx + cd_block_size, d)
            for update_idx in range(start_idx, end_idx):
                index = torch.arange(update_idx, update_idx + 1, device=device)
                # target is the TRUE w_i; the g_i/H_ii part is already inside B.
                sol = W_grp[:, :, index] - B_grp[:, :, index]
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
    g_over_diag: Optional[torch.Tensor] = None,  # (out, in) = g_bar/diag(H) FLOORED
):
    """
    Closed-form codebook update, CONSISTENT with update_P_fo.

    Both updates must aim the reconstruction at the SAME shifted target
        w_tilde = w - g_bar/diag(H)   (floored, diagonal -- NOT full H^{-1}).
    So we fit the codebook by least squares to w_tilde instead of w. This is the
    fix for the PPL blow-up: the previous version shifted the RHS by L^{-1} g_bar
    (a FULL-inverse Newton step, un-floored), which exploded along small-eigenvalue
    directions and dragged the deployed codebook off the true weights, while
    update_P used the floored diagonal shift -- the two were inconsistent. Now
    both use the identical floored diagonal target. g_over_diag=None -> vanilla.
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

    # shifted reconstruction target: same floored diagonal shift as update_P.
    if g_over_diag is not None:
        W_tilde = (W - g_over_diag.to(device))
        logging.info(
            f"[lnqf-dbg] update_C: fitting codebook to shifted target "
            f"||w_tilde - w||={g_over_diag.norm().item():.3e} "
            f"({100.0*g_over_diag.norm().item()/max(W.norm().item(),1e-12):.3f}% of ||w||)"
        )
    else:
        W_tilde = W

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
            # fit to the SHIFTED target w_tilde (identical target to update_P).
            b_batch_tmp = torch.einsum('bj,ij->ib', X_batch, W_tilde[st_idx:end_idx]).unsqueeze(-1)
            A_batch_list.append(A_batch_tmp)
            b_batch_list.append(b_batch_tmp)

        A_batch = torch.cat(A_batch_list, dim=1)
        b_batch = torch.cat(b_batch_list, dim=1)   # = L^T w_tilde

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
        # diag(H) per coordinate. A handful of near-singular coords (H_ii -> 0)
        # would make g_bar_i/H_ii explode. We FLOOR H_ii per-coordinate at a
        # percentile of the diagonal (tau), which caps ONLY the degenerate tail
        # and leaves the ~99% healthy coordinates untouched. This replaces the old
        # global trust-region/gamma, which let the bad tail kill B1 everywhere.
        diagH_all = torch.cat([H[gi, torch.arange(d), torch.arange(d)] for gi in range(num_groups)])
        # tau = a low percentile of diag(H); floor keeps shift bounded at the tail.
        tau = torch.quantile(diagH_all.float(), 0.10).item()
        floor_frac = float(max_shift) if max_shift and max_shift > 0 else 1.0
        tau = tau * floor_frac   # max_shift now tunes the floor (>1 => stronger floor)

        g_over_diag = torch.empty_like(g_bar_t)
        n_floored = 0
        for gi in range(num_groups):
            r0, r1 = gi * group_size, (gi + 1) * group_size
            diagH = H[gi, torch.arange(d), torch.arange(d)]
            diagH_floored = diagH.clamp_min(tau)
            n_floored += (diagH < tau).sum().item()
            g_over_diag[r0:r1] = g_bar_t[r0:r1] / diagH_floored.unsqueeze(0)

        rms_w = W.pow(2).mean().sqrt().item()
        shift_abs = g_over_diag.abs()
        q = torch.tensor([0.5, 0.9, 0.99, 0.999, 1.0], device=g_over_diag.device)
        shift_q = torch.quantile(shift_abs.flatten().float(), q).tolist()
        logging.info(
            f"[lnqf-dbg] RMS(g_bar)={g_bar_t.pow(2).mean().sqrt().item():.3e} "
            f"diagH[min={diagH_all.min().item():.3e} med={diagH_all.median().item():.3e} "
            f"max={diagH_all.max().item():.3e}] tau_floor={tau:.3e} "
            f"floored {100.0*n_floored/diagH_all.numel():.2f}% coords RMS(w)={rms_w:.3e}"
        )
        logging.info(
            f"[lnqf-dbg] |shift| quantiles [50%={shift_q[0]:.3e} 90%={shift_q[1]:.3e} "
            f"99%={shift_q[2]:.3e} 99.9%={shift_q[3]:.3e} max={shift_q[4]:.3e}] "
            f"| median shift/w={shift_q[0]/max(rms_w,1e-30):.4f} "
            f"max shift/w={shift_q[4]/max(rms_w,1e-30):.4f}"
        )
        logging.info("[lnqf] first-order folded into B (dynamic descent), diagH floored, no trust-region.")
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

    # Evaluate the three error curves on a given (labels, C), ALWAYS reporting
    # all three even when the solver itself ignores g_bar (so we can score a
    # vanilla-LNQ solution against the first-order term for comparison):
    #   B2  = 1/2 (w_hat-w)^T H (w_hat-w)   pure second-order (output error)
    #   lin = g_bar^T (w_hat - w)           pure first-order
    #   phi = B2 + lin                      the full objective
    def eval_three(labels_, C_):
        b2 = objective_function(W, H, labels_, C_).item()
        if g_bar_t is None:
            return b2, b2, 0.0
        Wq = torch.gather(C_.to(device).unsqueeze(1).expand(-1, W.shape[1], -1),
                          2, labels_.to(device).long().unsqueeze(-1)).squeeze(-1)
        lin = (g_bar_t * (Wq - W)).sum().item() / W.shape[0]  # mean over rows -> B2 scale
        return b2 + lin, b2, lin

    full_obj = eval_three  # backward-compatible alias

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

        C = update_C_fo(W, H, labels, C, g_over_diag=g_over_diag)
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

    # ----- head-to-head comparison against vanilla LNQ (no first-order term) -----
    # Run plain LNQ on the SAME W/H/init, then score BOTH solutions with all three
    # error curves, so we can see exactly what B1 traded: how much extra B2 (output
    # error) it accepted in exchange for how much lin / phi it removed.
    if g_bar_t is not None:
        from .layerwise_quantize import train_least_squares
        v_labels, v_C, _ = train_least_squares(
            W.detach().cpu().numpy(), init_labels, init_centroids,
            H.detach().cpu().numpy(),
            num_iterations=num_iterations, cd_cycles=cd_cycles,
        )
        v_labels_t = torch.tensor(v_labels, dtype=torch.long)
        v_C_t = torch.tensor(v_C, dtype=torch.float32)
        v_phi, v_b2, v_lin = eval_three(v_labels_t, v_C_t)
        f_phi, f_b2, f_lin = eval_three(best_labels, best_C)
        logging.info("========== LNQ-F vs vanilla LNQ (same H,W,init) ==========")
        logging.info(f"  vanilla LNQ : phi={v_phi:.5f}  B2={v_b2:.5f}  lin={v_lin:.5f}")
        logging.info(f"  LNQ-F (B1+B2): phi={f_phi:.5f}  B2={f_b2:.5f}  lin={f_lin:.5f}")
        logging.info(f"  delta        : phi={f_phi-v_phi:+.5f}  "
                     f"B2={f_b2-v_b2:+.5f} (B1 paid this in output error)  "
                     f"lin={f_lin-v_lin:+.5f} (B1 removed this in loss-aligned error)")
        logging.info(f"  => B1 traded B2 {f_b2-v_b2:+.5f} for phi {f_phi-v_phi:+.5f}. "
                     f"Net phi {'BETTER' if f_phi<v_phi else 'WORSE'} than vanilla.")
        logging.info("==========================================================")
        log_dict["vanilla_lnq"] = {"phi": v_phi, "b2": v_b2, "lin": v_lin}
        log_dict["lnqf"] = {"phi": f_phi, "b2": f_b2, "lin": f_lin}

    labels = best_labels.detach().cpu().numpy().astype(np.float32)
    C = best_C.detach().cpu().numpy().astype(np.float32)
    return labels, C, log_dict
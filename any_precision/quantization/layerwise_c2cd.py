"""
C2CD solver: correlation-matching M2-CD assignment optimizer.

Same interface as `train_least_squares` (LNQ): consumes (W, init_labels,
init_centroids, H) and returns (labels, C, log_dict) with identical shapes and
dtypes, so it drops straight into the existing dispatch in
`layerwise_quantize.py` behind `--solver c2cd`.

The only thing swapped vs LNQ is the assignment step. LNQ's `update_P` does
1-opt scalar coordinate descent (each coord flipped independently over the m
levels). Here we replace it with the "** corr-matching M2-CD **" runner from
m2cd_verify.py: coordinates are paired by a greedy max-weight matching over the
top-nu correlation graph (|H_ik| normalised), and every pair is updated by an
EXACT m^2 joint enumeration. This crosses the pair-barriers that no single
1-opt flip can reach (M2-CD dominates natural- and paired-order scalar CD from
the same init). The codebook solve (Eq. 9 / update_C) is untouched.

GPU per preference. Nothing runs on import.
"""

import time
import logging

import numpy as np
import torch

from .layerwise_quantize import objective_function, update_C


# ----------------------------------------------------------------------------
# Correlation matching (ported from m2cd_verify.py, adapted to a raw d x d H)
# ----------------------------------------------------------------------------
def _correlation_graph(H, nu):
    """Normalised |H_ik| top-nu neighbour edges for a single (d,d) Hessian."""
    d = H.shape[0]
    diag = torch.diagonal(H).clamp_min(1e-12).sqrt()
    rho = H.abs() / (diag.view(-1, 1) * diag.view(1, -1))
    rho.fill_diagonal_(0.0)
    nu = min(nu, d - 1)
    vals, idx = rho.topk(nu, dim=1)
    return vals, idx


def _greedy_correlation_matching(H, nu, avoid=None):
    """Greedy max-weight matching over the sparse top-nu correlation graph.

    `avoid` is a set of frozenset edges already used, mildly penalised so that
    successive sweeps pick diverse matchings (§ same trick as the verify script).
    """
    vals, idx = _correlation_graph(H, nu)
    d = H.shape[0]
    edges = []
    for i in range(d):
        for j in range(vals.shape[1]):
            k = int(idx[i, j])
            if k > i:
                w = float(vals[i, j])
                pen = -0.1 if (avoid and frozenset((i, k)) in avoid) else 0.0
                edges.append((w + pen, i, k))
    edges.sort(reverse=True)
    used = torch.zeros(d, dtype=torch.bool)
    matching, new_edges = [], set()
    for w, i, k in edges:
        if not used[i] and not used[k]:
            used[i] = used[k] = True
            matching.append((i, k))
            new_edges.add(frozenset((i, k)))
    for i in range(d):
        if not used[i]:
            matching.append((i, None))
    return matching, new_edges


# ----------------------------------------------------------------------------
# M2-CD assignment sweep (per-row codebook, exact m^2 pair block)
# ----------------------------------------------------------------------------
def _m2cd_update_P(W, H, labels, C, matching):
    """One exact M2-CD sweep. Drop-in replacement for LNQ's update_P.

    W:      (out, d)
    H:      (d, d)   single-group Hessian
    labels: (out, d) int
    C:      (out, m) per-row codebook
    matching: list of (i, k) disjoint pairs plus (i, None) singletons.

    Maintains e = C[p]-W and G = e @ H, updates in place, returns new labels.
    Gain in DELTA form (identical algebra to m2cd_verify.selftest):
        dl = 2(di*G_i + dk*G_k) + di^2 H_ii + dk^2 H_kk + 2 di dk H_ik
    """
    Hii = torch.diagonal(H)                         # (d,)
    p = labels.clone()
    # per-row residual for the currently chosen label of each coord
    e = torch.gather(C, 1, p.long())                # (out, d)  == C[p]
    e = e - W
    G = e @ H                                        # (out, d)
    m = C.shape[1]

    for (i, k) in matching:
        if k is None:
            # candidate residual e_i for each level q: C[:,q]-W[:,i]
            cand_res = C - W[:, i:i + 1]                       # (out, m)
            d_move = cand_res - e[:, i:i + 1]                  # (out, m)
            gain = 2 * d_move * G[:, i:i + 1] + d_move.pow(2) * Hii[i]
            bq = gain.argmin(dim=1)                            # (out,)
            new_ei = cand_res.gather(1, bq.unsqueeze(1)).squeeze(1)
            de = new_ei - e[:, i]
            e[:, i] = new_ei
            G += de.unsqueeze(1) * H[i].view(1, -1)
            p[:, i] = bq
            continue

        ri = C - W[:, i:i + 1]        # (out, m) candidate e_i
        rk = C - W[:, k:k + 1]        # (out, m) candidate e_k
        di = ri - e[:, i:i + 1]       # (out, m)
        dk = rk - e[:, k:k + 1]       # (out, m)

        Hik = H[i, k]
        DI = di.unsqueeze(2)          # (out, m, 1)
        DK = dk.unsqueeze(1)          # (out, 1, m)
        dl = (2 * (DI * G[:, i].view(-1, 1, 1) + DK * G[:, k].view(-1, 1, 1))
              + DI.pow(2) * Hii[i] + DK.pow(2) * Hii[k]
              + 2 * DI * DK * Hik)                            # (out, m, m)

        flat = dl.view(dl.shape[0], -1)                       # (out, m*m)
        best = flat.argmin(dim=1)
        qi = torch.div(best, m, rounding_mode="floor")
        qk = best % m

        new_ei = ri.gather(1, qi.unsqueeze(1)).squeeze(1)
        new_ek = rk.gather(1, qk.unsqueeze(1)).squeeze(1)
        dei = new_ei - e[:, i]
        dek = new_ek - e[:, k]
        e[:, i] = new_ei
        e[:, k] = new_ek
        G += dei.unsqueeze(1) * H[i].view(1, -1) + dek.unsqueeze(1) * H[k].view(1, -1)
        p[:, i] = qi
        p[:, k] = qk

    return p


# ----------------------------------------------------------------------------
# Trainer (same contract as train_least_squares)
# ----------------------------------------------------------------------------
def train_c2cd(
    W: np.ndarray,             # (output_dim, input_dim)
    init_labels: np.ndarray,   # (output_dim, input_dim)
    init_centroids: np.ndarray,  # (output_dim, n_cluster)
    H: np.ndarray,             # (num_groups, input_dim, input_dim)
    num_iterations: int = 3,
    cd_cycles: int = 4,
    c2cd_nu: int = 16,
    c2cd_cycles: int = None,
    **_ignored,
):
    """LNQ-style alternating solve with M2-CD corr-matching as the P-step.

    c2cd_nu     : top-nu correlation neighbours per coord for the matching graph.
    c2cd_cycles : number of M2-CD sweeps per P-step (fresh diverse matching each
                  sweep). Defaults to cd_cycles so it matches LNQ's CD budget.
    """
    device = torch.device("cuda")
    if c2cd_cycles is None:
        c2cd_cycles = cd_cycles
    c2cd_nu = int(c2cd_nu)
    c2cd_cycles = int(c2cd_cycles)

    labels = torch.tensor(init_labels, dtype=torch.int8, device="cpu")
    C = torch.tensor(init_centroids, dtype=torch.float32, device="cpu")
    W = torch.tensor(W, dtype=torch.float32).to(device)
    H = torch.tensor(H, dtype=torch.float32).to(device)

    assert H.shape[0] == 1, "c2cd assumes group_count == 1 (matches LNQ pipeline)"
    Hd = H[0]  # (d, d)

    # PD guard identical in spirit to LNQ's damping loop.
    diag = torch.arange(Hd.shape[0], device=device)
    avg_diag = torch.mean(torch.diag(Hd))
    damp, prev_damp = 1e-5, 0.0
    while True:
        try:
            torch.linalg.cholesky(Hd)
            break
        except Exception:
            Hd[diag, diag] += (damp - prev_damp) * avg_diag
            prev_damp, damp = damp, damp * 10
            if damp > 1e0:
                raise RuntimeError("c2cd: H not PD even after heavy damping")
    H = Hd.unsqueeze(0)  # keep update_C's (num_groups, d, d) contract

    # Precompute diverse correlation matchings once (H is fixed across iters).
    matchings, used = [], set()
    for _ in range(c2cd_cycles):
        mt, new = _greedy_correlation_matching(Hd, nu=c2cd_nu, avoid=used)
        matchings.append(mt)
        used |= new

    best_obj = objective_function(W, H, labels, C).item()
    best_labels, best_C = labels.detach().cpu().clone(), C.detach().cpu().clone()
    logging.info(f"[c2cd] Initial objective: {best_obj:.6f} "
                 f"(nu={c2cd_nu}, sweeps={c2cd_cycles})")

    log_dict = {"objective": [best_obj], "iteration": [0]}

    for iteration in range(num_iterations):
        start = time.time()

        ######### Update P (M2-CD corr-matching, replaces 1-opt CD) #########
        if iteration > 0:
            lab = labels.to(device).long()
            for mt in matchings:
                lab = _m2cd_update_P(W, Hd, lab, C.to(device), mt)
            labels = lab.to(torch.int8).cpu()

        obj_p = objective_function(W, H, labels, C).item()
        logging.info(f"[c2cd] Iter {iteration + 1} (P update): Objective: {obj_p:.4f}")
        log_dict["objective"].append(obj_p)
        log_dict["iteration"].append(iteration + 1)

        ######### Update C (unchanged Eq. 9 solve) #########
        C = update_C(W, H, labels, C, iteration)

        cur = objective_function(W, H, labels, C).item()
        log_dict["objective"].append(cur)
        log_dict["iteration"].append(iteration + 1)
        if cur < best_obj:
            best_obj = cur
            best_labels, best_C = labels.detach().cpu().clone(), C.detach().cpu().clone()
            logging.info(f"[c2cd] Iter {iteration + 1} (C update): Objective: {cur:.4f} | improved.")
        else:
            logging.info(f"[c2cd] Iter {iteration + 1} (C update): Objective: {cur:.4f} | not improved, reverting.")
            labels, C = best_labels, best_C
            break

        logging.info(f"[c2cd] Iter {iteration + 1}/{num_iterations} done "
                     f"({time.time() - start:.2f}s)")

    labels = best_labels.detach().cpu().numpy()
    C = best_C.detach().cpu().numpy().astype(np.float32)
    return labels, C, log_dict

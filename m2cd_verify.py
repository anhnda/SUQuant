"""
M2-CD (Matched Multi-Level 2-Block Coordinate Descent) verification.

Verifies the proposal's core claims on synthetic GuidedQuant-style problems:
    min_p  L_r(p) = e_r^T H e_r,   e_ri = c_{p_i} - W_ri,   p_i in {1..m}
with H = X^T Diag(s) X  (PSD), shared across R output rows.

Methods compared per the doc's Section 15 comparison block:
    - B0/B2 : 1-opt scalar CD (natural order) and paired-order scalar CD
    - M2-CD : exact m^2 pair-block update, random vs correlation matching
              vs repeated matching (B4)

Torch is used for the math (per preference). Nothing runs on import;
call main() yourself.
"""

import torch


# ----------------------------------------------------------------------------
# Synthetic problem
# ----------------------------------------------------------------------------
def make_problem(d=256, R=64, n=512, m=4, block_corr=8, seed=0,
                 device="cpu", dtype=torch.float64):
    """
    Build a GuidedQuant-style instance with deliberately correlated coordinates
    so that pair barriers actually exist (block-structured X columns).

    Returns H (d,d PSD), W (R,d) target weights, codebook c (m,) shared per row.
    """
    g = torch.Generator(device=device).manual_seed(seed)

    # Block-correlated design matrix: coords in the same block share a latent
    # factor -> off-diagonal H_ik with meaningful sign structure.
    X = torch.randn(n, d, generator=g, device=device, dtype=dtype)
    nblocks = d // block_corr
    for b in range(nblocks):
        sl = slice(b * block_corr, (b + 1) * block_corr)
        factor = torch.randn(n, 1, generator=g, device=device, dtype=dtype)
        sign = (torch.randint(0, 2, (block_corr,), generator=g, device=device) * 2 - 1)
        X[:, sl] = 0.6 * X[:, sl] + 1.4 * factor * sign.to(dtype)

    s = torch.rand(n, generator=g, device=device, dtype=dtype) + 0.2  # GuidedQuant weights
    H = X.t() @ (s.unsqueeze(1) * X)          # (d,d) PSD
    H = 0.5 * (H + H.t())                      # symmetrize numerically

    # True continuous weights, then a shared codebook (uniform grid over range).
    W = 0.5 * torch.randn(R, d, generator=g, device=device, dtype=dtype)
    lo, hi = W.min(), W.max()
    c = torch.linspace(lo.item(), hi.item(), m, device=device, dtype=dtype)  # (m,)

    return {"H": H, "W": W, "c": c, "m": m, "d": d, "R": R}


def init_labels(prob, device="cpu"):
    """Nearest-codeword init (independent per coord/row), shape (R,d)."""
    W, c = prob["W"], prob["c"]
    dist = (W.unsqueeze(-1) - c.view(1, 1, -1)).abs()   # (R,d,m)
    return dist.argmin(dim=-1)                           # (R,d)


# ----------------------------------------------------------------------------
# Objective + gradient bookkeeping
# ----------------------------------------------------------------------------
def residual(prob, p):
    """e = C[p] - W, shape (R,d)."""
    return prob["c"][p] - prob["W"]


def objective(prob, p):
    """L_r = e_r^T H e_r summed over rows. Returns per-row (R,) and total."""
    e = residual(prob, p)          # (R,d)
    He = e @ prob["H"]             # (R,d)
    Lr = (e * He).sum(dim=1)       # (R,)
    return Lr, Lr.sum()


# ----------------------------------------------------------------------------
# 1-opt scalar CD  (exact per-coordinate over m levels)
# ----------------------------------------------------------------------------
def scalar_cd_sweep(prob, p, order):
    """
    One in-place scalar-CD sweep over coords in `order`.
    Maintains G = e @ H so each coord update is O(m) + O(d) propagation.
    Delta_i(q) = 2*delta*G_i + delta^2 * H_ii.
    """
    H, c = prob["H"], prob["c"]
    Hii = torch.diagonal(H)                 # (d,)
    e = residual(prob, p)                    # (R,d)
    G = e @ H                                # (R,d)
    moves = 0

    for i in order:
        delta = c.view(1, -1) - e[:, i:i + 1] - prob["W"][:, i:i + 1] + prob["W"][:, i:i + 1]
        # delta_i(q) = c_q - c_{p_i} = c_q - (e_i + W_i)  -> current residual = e_i
        cand_res = c.view(1, -1) - prob["W"][:, i:i + 1]          # candidate e_i for each q, (R,m)
        d_move = cand_res - e[:, i:i + 1]                          # delta per candidate (R,m)
        gain = 2 * d_move * G[:, i:i + 1] + d_move.pow(2) * Hii[i] # (R,m)
        best_q = gain.argmin(dim=1)                                # (R,)
        best_gain = gain.gather(1, best_q.unsqueeze(1)).squeeze(1) # (R,)

        changed = best_q != p[:, i]
        moves += int(changed.sum())

        # apply: update residual col i and propagate to G via H row i
        new_ei = cand_res.gather(1, best_q.unsqueeze(1)).squeeze(1)  # (R,)
        de = new_ei - e[:, i]                                        # (R,)
        e[:, i] = new_ei
        G += de.unsqueeze(1) * H[i].view(1, -1)                      # rank-1 update
        p[:, i] = best_q

    return p, moves


# ----------------------------------------------------------------------------
# M2-CD  (exact m^2 pair-block update)
# ----------------------------------------------------------------------------
def m2cd_sweep(prob, p, matching):
    """
    One M2-CD sweep. `matching` is a list of (i,k) disjoint pairs plus optional
    singletons as (i, None). Each pair enumerated over all m^2 assignments
    against the exact local objective  l = 2 r^T b + r^T H_TT r.
    Returns updated p and a stats dict for barrier instrumentation.
    """
    H, c, m = prob["H"], prob["c"], prob["m"]
    Hii = torch.diagonal(H)
    e = residual(prob, p)     # (R,d)
    G = e @ H                 # (R,d)  (full field; b_T derived from it)

    stats = {"pair_moves": 0, "single_moves": 0, "blocks": 0,
             "pair_gain": 0.0, "jump_dist": []}

    cvec = c.view(-1)  # (m,)

    for (i, k) in matching:
        stats["blocks"] += 1
        if k is None:
            # singleton: scalar exact update
            cand_res = cvec.view(1, -1) - prob["W"][:, i:i + 1]        # (R,m)
            d_move = cand_res - e[:, i:i + 1]
            gain = 2 * d_move * G[:, i:i + 1] + d_move.pow(2) * Hii[i]
            bq = gain.argmin(dim=1)
            changed = bq != p[:, i]
            stats["single_moves"] += int(changed.sum())
            new_ei = cand_res.gather(1, bq.unsqueeze(1)).squeeze(1)
            de = new_ei - e[:, i]
            e[:, i] = new_ei
            G += de.unsqueeze(1) * H[i].view(1, -1)
            p[:, i] = bq
            continue

        # candidate residuals for each coord: (R,m)
        ri = cvec.view(1, -1) - prob["W"][:, i:i + 1]   # (R,m)  candidate e_i
        rk = cvec.view(1, -1) - prob["W"][:, k:k + 1]   # (R,m)  candidate e_k

        # deltas relative to current residual
        di = ri - e[:, i:i + 1]     # (R,m)
        dk = rk - e[:, k:k + 1]     # (R,m)

        # outside field b_T = H[T,-T] e_{-T} = G_T - H_TT e_T
        # G already includes contribution of coords i,k; subtract them.
        Hik = H[i, k]
        bi = G[:, i] - (Hii[i] * e[:, i] + Hik * e[:, k])   # (R,)
        bk = G[:, k] - (Hik * e[:, i] + Hii[k] * e[:, k])   # (R,)

        # exact local change for joint move (qi,qk):
        #   dl = 2(di*bi + dk*bk) + di^2 Hii + dk^2 Hkk + 2 di dk Hik
        # broadcast over (R, m_i, m_k)
        DI = di.unsqueeze(2)   # (R,m,1)
        DK = dk.unsqueeze(1)   # (R,1,m)
        dl = (2 * (DI * bi.view(-1, 1, 1) + DK * bk.view(-1, 1, 1))
              + DI.pow(2) * Hii[i] + DK.pow(2) * Hii[k]
              + 2 * DI * DK * Hik)                          # (R,m,m)

        flat = dl.view(dl.shape[0], -1)                     # (R, m*m)
        best = flat.argmin(dim=1)                           # (R,)
        best_gain = flat.gather(1, best.unsqueeze(1)).squeeze(1)  # (R,)
        qi = best // m
        qk = best % m

        old_i, old_k = p[:, i].clone(), p[:, k].clone()
        ci = qi != old_i
        ck = qk != old_k
        both = ci & ck
        stats["pair_moves"] += int(both.sum())
        stats["single_moves"] += int((ci ^ ck).sum())
        stats["pair_gain"] += float((best_gain[both]).sum()) if both.any() else 0.0
        if both.any():
            jumps = (qi[both].to(torch.int64) - old_i[both].to(torch.int64)).abs()
            stats["jump_dist"].extend(jumps.tolist())

        # commit + propagate both columns
        new_ei = ri.gather(1, qi.unsqueeze(1)).squeeze(1)
        new_ek = rk.gather(1, qk.unsqueeze(1)).squeeze(1)
        dei = new_ei - e[:, i]
        dek = new_ek - e[:, k]
        e[:, i] = new_ei
        e[:, k] = new_ek
        G += dei.unsqueeze(1) * H[i].view(1, -1) + dek.unsqueeze(1) * H[k].view(1, -1)
        p[:, i] = qi
        p[:, k] = qk

    return p, stats


# ----------------------------------------------------------------------------
# Matchings
# ----------------------------------------------------------------------------
def correlation_graph(prob, nu=16):
    """Normalized |H_ik| top-nu neighbor edges."""
    H = prob["H"]
    d = prob["d"]
    diag = torch.diagonal(H).clamp_min(1e-12).sqrt()
    rho = H.abs() / (diag.view(-1, 1) * diag.view(1, -1))
    rho.fill_diagonal_(0.0)
    nu = min(nu, d - 1)
    vals, idx = rho.topk(nu, dim=1)          # (d,nu)
    return vals, idx


def greedy_correlation_matching(prob, nu=16, avoid=None, seed=0):
    """Greedy max-weight matching over sparse top-nu graph. `avoid` is a set of
    frozenset edges already used, to encourage diverse matchings across sweeps."""
    vals, idx = correlation_graph(prob, nu)
    d = prob["d"]
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


def random_matching(d, seed=0):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(d, generator=g).tolist()
    matching = []
    for j in range(0, d - 1, 2):
        matching.append((perm[j], perm[j + 1]))
    if d % 2 == 1:
        matching.append((perm[-1], None))
    return matching


# ----------------------------------------------------------------------------
# Runners
# ----------------------------------------------------------------------------
def run_scalar(prob, p0, K, paired_order=False, matchings=None):
    p = p0.clone()
    d = prob["d"]
    traj = [objective(prob, p)[1].item()]
    for s in range(K):
        if paired_order:
            mt = matchings[s % len(matchings)]
            order = []
            for (i, k) in mt:
                order.append(i)
                if k is not None:
                    order.append(k)
        else:
            order = list(range(d))
        p, _ = scalar_cd_sweep(prob, p, order)
        traj.append(objective(prob, p)[1].item())
    return p, traj


def run_m2cd(prob, p0, matchings):
    p = p0.clone()
    traj = [objective(prob, p)[1].item()]
    all_stats = []
    for mt in matchings:
        p, st = m2cd_sweep(prob, p, mt)
        all_stats.append(st)
        traj.append(objective(prob, p)[1].item())
    return p, traj, all_stats


# ----------------------------------------------------------------------------
# Oracle barrier recall diagnostic (Section 13)
# ----------------------------------------------------------------------------
def find_1opt_fixed(prob, p, max_sweeps=50):
    """Drive scalar CD to a 1-opt fixed point."""
    p = p.clone()
    d = prob["d"]
    prev = objective(prob, p)[1].item()
    for _ in range(max_sweeps):
        p, moves = scalar_cd_sweep(prob, p, list(range(d)))
        cur = objective(prob, p)[1].item()
        if moves == 0 or abs(prev - cur) < 1e-12:
            break
        prev = cur
    return p


def oracle_improving_pairs(prob, p, neighbor_idx, row=0):
    """For a single row at a 1-opt point, count exact improving pairs over the
    sparse neighbor graph. Returns set of frozenset edges that improve."""
    H, c, m = prob["H"], prob["c"], prob["m"]
    Hii = torch.diagonal(H)
    e = (c[p] - prob["W"])[row]        # (d,)
    G = (e @ H)                        # (d,)
    improving = set()
    cvec = c.view(-1)
    for i in range(prob["d"]):
        for j in range(neighbor_idx.shape[1]):
            k = int(neighbor_idx[i, j])
            if k <= i:
                continue
            ri = cvec - prob["W"][row, i]     # (m,)
            rk = cvec - prob["W"][row, k]
            di = ri - e[i]
            dk = rk - e[k]
            Hik = H[i, k]
            bi = G[i] - (Hii[i] * e[i] + Hik * e[k])
            bk = G[k] - (Hik * e[i] + Hii[k] * e[k])
            DI = di.view(-1, 1)
            DK = dk.view(1, -1)
            dl = (2 * (DI * bi + DK * bk) + DI.pow(2) * Hii[i]
                  + DK.pow(2) * Hii[k] + 2 * DI * DK * Hik)
            # improving pair barrier: joint move helps but neither single move does
            best = dl.min()
            single_best = min(dl[:, p[k]].min(), dl[p[i], :].min())
            if best < -1e-9 and best < single_best - 1e-9:
                improving.add(frozenset((i, k)))
    return improving


# ----------------------------------------------------------------------------
# Main experiment
# ----------------------------------------------------------------------------
def main():
    torch.manual_seed(0)
    print("=" * 74)
    for m, bits in [(4, 2), (8, 3)]:
        prob = make_problem(d=256, R=64, n=512, m=m, block_corr=8, seed=1)
        p0 = init_labels(prob)
        L0 = objective(prob, p0)[1].item()
        K = 4

        # precompute diverse correlation matchings + a fixed one + random ones
        corr_ms, used = [], set()
        for s in range(K):
            mt, new = greedy_correlation_matching(prob, nu=16, avoid=used)
            corr_ms.append(mt)
            used |= new
        fixed_m = corr_ms[0]
        rand_ms = [random_matching(prob["d"], seed=s) for s in range(K)]

        # --- runs ---
        _, t_scalar = run_scalar(prob, p0, K)                              # B0
        _, t_paired = run_scalar(prob, p0, K, paired_order=True,
                                 matchings=corr_ms)                        # B2
        _, t_rand, _ = run_m2cd(prob, p0, rand_ms)                         # B3
        _, t_fixed, _ = run_m2cd(prob, p0, [fixed_m] * K)                  # B4
        p_corr, t_corr, stats = run_m2cd(prob, p0, corr_ms)               # main

        def red(traj):  # relative reduction vs L0
            return 100.0 * (L0 - traj[-1]) / L0

        print(f"[{bits}-bit  m={m}]  L0={L0:.4f}   (lower L = better)")
        print(f"  B0 scalar CD (natural)      L={t_scalar[-1]:.4f}  red={red(t_scalar):6.3f}%")
        print(f"  B2 paired-order scalar CD   L={t_paired[-1]:.4f}  red={red(t_paired):6.3f}%")
        print(f"  B3 random-matching M2-CD    L={t_rand[-1]:.4f}  red={red(t_rand):6.3f}%")
        print(f"  B4 repeated-matching M2-CD  L={t_fixed[-1]:.4f}  red={red(t_fixed):6.3f}%")
        print(f"  ** corr-matching M2-CD **   L={t_corr[-1]:.4f}  red={red(t_corr):6.3f}%")

        tot_pair = sum(s["pair_moves"] for s in stats)
        tot_gain = sum(s["pair_gain"] for s in stats)
        jumps = [j for s in stats for j in s["jump_dist"]]
        print(f"  barrier crossings: pair_moves={tot_pair}  "
              f"pair_gain={tot_gain:.4f}  "
              f"mean_jump={ (sum(jumps)/len(jumps)) if jumps else 0:.2f}")

        # monotonicity check
        mono = all(t_corr[i + 1] <= t_corr[i] + 1e-9 for i in range(len(t_corr) - 1))
        print(f"  monotone descent (corr M2-CD): {mono}")

        # M2 dominates paired-order scalar from same init (Section 5 claim)
        dom = t_corr[-1] <= t_paired[-1] + 1e-9
        print(f"  M2-CD <= paired-order scalar CD: {dom}")

        # --- oracle barrier recall (Section 13) on row 0 ---
        p_fix = find_1opt_fixed(prob, p0)
        _, nb_idx = correlation_graph(prob, nu=16)
        oracle = oracle_improving_pairs(prob, p_fix, nb_idx, row=0)
        covered = set()
        for mt in corr_ms:
            for (i, k) in mt:
                if k is not None:
                    covered.add(frozenset((i, k)))
        recall = (len(oracle & covered) / len(oracle)) if oracle else float("nan")
        print(f"  oracle improving pair-barriers (row0): {len(oracle)}  "
              f"covered={len(oracle & covered)}  recall={recall:.3f}")
        print("-" * 74)


if __name__ == "__main__":
    main()
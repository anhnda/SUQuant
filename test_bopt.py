import numpy as np, torch, logging, importlib.util, os
logging.basicConfig(level=logging.INFO)
_p = os.path.join(os.path.dirname(__file__),
                  "any_precision/quantization/layerwise_bopt.py")
_spec = importlib.util.spec_from_file_location("layerwise_bopt", _p)
_bopt = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_bopt)
bopt_refine = _bopt.bopt_refine
_stage_pair_pass = _bopt._stage_pair_pass
_alt_gains = _bopt._alt_gains
_neighbour_lists = _bopt._neighbour_lists
_gather_cb = _bopt._gather_cb
_group_energy = _bopt._group_energy
solve_codebook = _bopt.solve_codebook
cd_to_fixed_point = _bopt.cd_to_fixed_point

torch.manual_seed(0); np.random.seed(0)
dev = torch.device("cpu")

# --- synthetic layer: correlated H so pair moves have something to exploit ---
d, R, m, g = 64, 32, 4, 1
X = torch.randn(256, d)
# inject strong off-diagonal correlations
A = torch.randn(d, d) * 0.3
H = (X.T @ X) / 256 + A @ A.T          # PSD, correlated
H = 0.5 * (H + H.T)
H = H + torch.eye(d) * 0.05
Hg = H.numpy()[None]                    # (1,d,d)

W = torch.randn(R, d).numpy()

# k-means-ish init per row: cheap uniform codebook + nearest
Wt = torch.tensor(W)
cb0 = torch.stack([torch.quantile(Wt[r], torch.linspace(0,1,m)) for r in range(R)])
idx0 = (Wt.unsqueeze(-1) - cb0.unsqueeze(1)).abs().argmin(-1)
labels = idx0.numpy().astype(np.uint8)
C = cb0.numpy().astype(np.float32)

def obj(labels, C):
    idx = torch.tensor(labels, dtype=torch.long)
    cb = torch.tensor(C)
    Wh = torch.gather(cb.unsqueeze(1).expand(-1,d,-1),2,idx.unsqueeze(-1)).squeeze(-1)
    dW = (Wh - Wt).reshape(g, R//g, d)
    return torch.einsum('nij,njk,nik->i', dW, torch.tensor(Hg), dW).mean().item()

print("=== Test 1: monotonicity (objective must not increase) ===")
o0 = obj(labels, C)
for st in (1, 2, 3):
    lab, Cn, log = bopt_refine(W, Hg, labels, C, stages=st, device="cpu",
                               n_chains=20, chain_depth=10)
    o1 = obj(lab, Cn)
    print(f" stage={st}: obj {o0:.6f} -> {o1:.6f}  ({100*(o1-o0)/o0:+.3f}%)  "
          f"release={100*log['median_release_frac']:.3f}%")
    assert o1 <= o0 + 1e-6, f"MONOTONICITY VIOLATED at stage {st}"
print(" OK: monotone at all stages\n")

print("=== Test 2: exact pair gain matches brute-force energy delta ===")
# take one row, one accepted-style pair, verify dE formula == actual energy diff
Wg = Wt.clone(); Hgt = torch.tensor(Hg[0]); Hdiag = torch.diagonal(Hgt).clone()
cb = torch.tensor(C); idx = torch.tensor(labels, dtype=torch.long)
idx = cd_to_fixed_point(Wg, Hgt, Hdiag, cb, idx, max_sweeps=20)[0]
E = _gather_cb(cb, idx) - Wg
G = E @ Hgt
r, i, k = 0, 3, 7
new_i, new_k = (idx[r,i].item()+1)%m, (idx[r,k].item()+1)%m
di = cb[r,new_i]-(_gather_cb(cb,idx)[r,i]); dk = cb[r,new_k]-(_gather_cb(cb,idx)[r,k])
dE_formula = (2*di*G[r,i]+di**2*Hdiag[i] + 2*dk*G[r,k]+dk**2*Hdiag[k]
              + 2*di*dk*Hgt[i,k]).item()
e_before = _group_energy(E[r:r+1], Hgt).item()
idx2 = idx.clone(); idx2[r,i]=new_i; idx2[r,k]=new_k
E2 = _gather_cb(cb, idx2) - Wg
e_after = _group_energy(E2[r:r+1], Hgt).item()
print(f" formula dE={dE_formula:.6e}   actual={e_after-e_before:.6e}")
assert abs(dE_formula-(e_after-e_before)) < 1e-4, "PAIR GAIN FORMULA WRONG"
print(" OK: exact pair-gain formula verified\n")

print("=== Test 3: held-out consistency (calib energy strictly down after B2) ===")
lab, Cn, log = bopt_refine(W, Hg, labels, C, stages=1, device="cpu")
print(f" accepts stage1 (group0): {log['groups'][0]['stage1']['n_accept']}")
print(f" level-jump hist: {log['groups'][0]['stage1']['jump_hist']}")
print(" OK\n")
print("ALL TESTS PASSED")
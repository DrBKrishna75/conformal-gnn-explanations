# week21_independent_weights.py
# Week 21: answers the second review.
#   - weighted_indep: weights fitted on an INDEPENDENT fitting family (new graphs) and an
#     INDEPENDENT unlabelled graph of the test setting, so the weights do not depend on the
#     calibration or test nodes (the condition of Theorem 2b).
#   - weighted (Algorithm 1) and weighted_split (node split) as in Week 20.
#   - graphmatch: split conformal on the calibration cell whose edge density is closest.
#   - a genuinely held-out setting on the other base graph (m = 4 / depth 7, 7.5% added edges).
#   - informativeness: share of nodes whose set is the whole subgraph, and mean fraction selected.
#   Same 27 BA and 18 tree configurations and seeds as Weeks 16, 17 and 20 (CPU, deterministic).
#
# Place in:  E:\conformal_gnn_project
# Run:       python week21_independent_weights.py
# Expected time: about 15-20 minutes (CPU).

import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.data import Data
from torch_geometric.datasets.graph_generator import BAGraph, GraphGenerator
from torch_geometric.datasets.motif_generator import CycleMotif, GridMotif, HouseMotif
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.nn import GCNConv, GINConv, GraphConv
from torch_geometric.utils import k_hop_subgraph

DATASETS = ["house", "cycle", "grid"]
ARCHS = ["GCN", "GIN", "GraphConv"]
ARCH_INDEX = {"GCN": 0, "GIN": 1, "GraphConv": 3}   # model seed offset (3 = new)
SEEDS = [0, 1, 2]
ALPHAS = [0.05, 0.10, 0.20]
CAL_M = [4, 5, 6]
CAL_NOISE = [0.0, 0.025, 0.05, 0.10]
CAL_NODES = 60
TESTS = [("in-support", 5, 0.0), ("held-out", 5, 0.0625),
         ("held-out", 4, 0.05), ("beyond", 5, 0.15)]
N_TEST_GRAPHS = 3
N_TEST = 100
METHODS = ["standard", "pooled", "weighted"]
EPOCHS = 1000
HID = 20
LR_LAMBDA = 1.0
TOPK_MASS = 12
N_HOPS = 3

device = torch.device("cpu")          # CPU for bit-exact reproducibility
torch.use_deterministic_algorithms(True, warn_only=True)
print("Running on:", device)

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR = os.path.join(CODE_DIR, "results")
os.makedirs(RES_DIR, exist_ok=True)
OUT_JSON = os.path.join(RES_DIR, "week21_independent_weights.json")
W16_JSON = os.path.join(RES_DIR, "week16_baselines.json")
W17_JSON = os.path.join(RES_DIR, "week17_trees_all_methods.json")
W18_JSON = os.path.join(RES_DIR, "week18_real_graphs.json")
W19_JSON = os.path.join(RES_DIR, "week19_explainers.json")


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


# ---------- graphs ----------
class TreeGraph(GraphGenerator):
    """Balanced binary tree of the given depth (depth 8 = 511 nodes), as in Week 14."""
    def __init__(self, depth):
        super().__init__()
        self.depth = depth

    def __call__(self):
        n = 2 ** (self.depth + 1) - 1
        child = torch.arange(1, n)
        parent = (child - 1) // 2
        ei = torch.stack([torch.cat([parent, child]), torch.cat([child, parent])])
        return Data(edge_index=ei, num_nodes=n)


FAMILY = "tree"


def make_graph(dataset, seed, m):
    set_seed(seed)
    motif = {"house": HouseMotif(), "cycle": CycleMotif(6), "grid": GridMotif()}[dataset]
    gen = BAGraph(num_nodes=300, num_edges=m) if FAMILY == "ba" else TreeGraph(m)
    d = ExplainerDataset(graph_generator=gen, motif_generator=motif, num_motifs=80)[0]
    d.x = torch.ones((d.num_nodes, 10))
    d.edge_mask = d.edge_mask.bool()
    return d


def add_noise(d, frac, seed):
    d = d.clone()
    if frac <= 0:
        return d
    n_add = int(frac * d.num_edges / 2)
    existing = set(zip(d.edge_index[0].tolist(), d.edge_index[1].tolist()))
    g = torch.Generator().manual_seed(seed)
    new = []
    while len(new) < n_add:
        a, b = torch.randint(0, d.num_nodes, (2,), generator=g).tolist()
        if a != b and (a, b) not in existing:
            existing.add((a, b))
            existing.add((b, a))
            new.append((a, b))
    add = torch.tensor(new, dtype=torch.long).t()
    add = torch.cat([add, add.flip(0)], dim=1)
    d.edge_index = torch.cat([d.edge_index, add], dim=1)
    d.edge_mask = torch.cat([d.edge_mask, torch.zeros(add.size(1), dtype=torch.bool)])
    return d


# ---------- models ----------
class Net(torch.nn.Module):
    def __init__(self, arch, in_dim, hid, out_dim):
        super().__init__()
        dims = [in_dim, hid, hid, hid]
        convs = []
        for i in range(3):
            if arch == "GCN":
                convs.append(GCNConv(dims[i], dims[i + 1]))
            elif arch == "GIN":
                mlp = torch.nn.Sequential(torch.nn.Linear(dims[i], dims[i + 1]), torch.nn.ReLU(),
                                          torch.nn.Linear(dims[i + 1], dims[i + 1]))
                convs.append(GINConv(mlp))
            else:
                convs.append(GraphConv(dims[i], dims[i + 1], aggr="add"))
        self.convs = torch.nn.ModuleList(convs)
        self.lin = torch.nn.Linear(hid, out_dim)

    def forward(self, x, edge_index):
        for c in self.convs:
            x = F.relu(c(x, edge_index))
        return self.lin(x)


def train_model(arch, d, seed, num_classes):
    set_seed(seed)
    model = Net(arch, d.x.size(1), HID, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    n = d.num_nodes
    perm = torch.randperm(n)
    n_tr, n_va, n_ca = int(0.6 * n), int(0.1 * n), int(0.15 * n)
    idx = {"train": perm[:n_tr], "val": perm[n_tr:n_tr + n_va],
           "cal": perm[n_tr + n_va:n_tr + n_va + n_ca], "test": perm[n_tr + n_va + n_ca:]}
    x, ei, y = d.x.to(device), d.edge_index.to(device), d.y.to(device)
    best_val, best_state = -1.0, None

    def acc(split):
        model.eval()
        with torch.no_grad():
            p = model(x, ei).argmax(1)
        return (p[idx[split].to(device)] == y[idx[split].to(device)]).float().mean().item()

    for ep in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(x, ei)[idx["train"].to(device)], y[idx["train"].to(device)])
        loss.backward()
        opt.step()
        if ep % 10 == 0:
            va = acc("val")
            if va >= best_val:
                best_val = va
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, idx, acc("train"), acc("val"), acc("test")


# ---------- explanations + features ----------
def explain_nodes(model, d, nodes):
    x, ei = d.x.to(device), d.edge_index.to(device)
    with torch.no_grad():
        preds = model(x, ei).argmax(1)
    gt_full = d.edge_mask.numpy()
    src, dst = d.edge_index[0].numpy(), d.edge_index[1].numpy()
    deg = np.bincount(src, minlength=d.num_nodes).astype(np.float64)
    epn = d.num_edges / d.num_nodes
    out = []
    for v in nodes:
        mask = torch.nn.Parameter(torch.ones(ei.size(1), device=device))
        set_masks(model, mask, ei, apply_sigmoid=False)
        model(x, ei)[v, preds[v]].backward()
        clear_masks(model)
        imp_full = mask.grad.detach().abs().cpu().numpy()
        _, _, _, sub = k_hop_subgraph(int(v), N_HOPS, d.edge_index, num_nodes=d.num_nodes)
        sub = sub.numpy()
        imp = imp_full[sub].astype(np.float64)
        gt = gt_full[sub]
        if gt.sum() == 0:
            continue
        n = len(imp)
        order = np.argsort(-imp, kind="stable")
        rank = np.empty(n, dtype=np.int64)
        rank[order] = np.arange(1, n + 1)
        p = imp / imp.sum() if imp.sum() > 0 else np.full(n, 1.0 / n)
        ent = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(n)) if n > 1 else 0.0
        topm = float(np.sort(p)[::-1][:TOPK_MASS].sum())
        nbrs = dst[src == v]
        feat = [math.log(n), math.log(deg[v] + 1),
                math.log(deg[nbrs].mean() + 1) if len(nbrs) else 0.0, epn, ent, topm]
        out.append({"rank": rank, "gt": gt, "score": int(rank[gt].max()),
                    "correct": bool(preds[v].item() == int(d.y[v])), "feat": feat})
    return out


def sample_motif_nodes(d, k, seed):
    rng = np.random.default_rng(seed)
    motif = np.where(d.y.numpy() > 0)[0]
    return [int(v) for v in rng.choice(motif, size=min(k, len(motif)), replace=False)]


# ---------- conformal ----------
def q_hat(scores, alpha):
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    return float("inf") if k > n else float(np.sort(scores)[k - 1])


def fit_lr(X, y, lam=LR_LAMBDA, iters=50):
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    w = np.zeros(Z.shape[1])
    R = lam * np.eye(Z.shape[1])
    R[0, 0] = 0.0
    for _ in range(iters):
        pr = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        g = Z.T @ (pr - y) + R @ w
        H = (Z * (pr * (1 - pr))[:, None]).T @ Z + R + 1e-8 * np.eye(Z.shape[1])
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break

    def logit(Xn):
        Zn = np.hstack([np.ones((len(Xn), 1)), (Xn - mu) / sd])
        return np.clip(Zn @ w, -30, 30)
    return logit


def weighted_q(scores_sorted, w_sorted, w_test, alpha):
    tot = w_sorted.sum() + w_test
    cum = np.cumsum(w_sorted) / tot
    idx = int(np.searchsorted(cum, 1 - alpha - 1e-12, side="left"))
    return float("inf") if idx >= len(scores_sorted) else float(scores_sorted[idx])




METHODS = ["standard", "pooled", "weighted", "dro_kl", "ot", "fewshot20"]
OT_STEPS, OT_LR, FEWSHOT_N = 200, 0.1, 20


def kl_binary(z, b):
    z = min(max(z, 1e-12), 1 - 1e-12)
    b = min(max(b, 1e-12), 1 - 1e-12)
    return z * math.log(z / b) + (1 - z) * math.log((1 - z) / (1 - b))


def g_kl(beta, rho):
    """g(beta) = inf{z in [0,1]: KL(Bern(z)||Bern(beta)) <= rho} (Cauchois et al.)."""
    if beta <= 0:
        return 0.0
    lo, hi = 0.0, beta          # KL is decreasing in z on [0, beta]
    if kl_binary(0.0, beta) <= rho:
        return 0.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if kl_binary(mid, beta) <= rho:
            hi = mid
        else:
            lo = mid
    return hi


def g_inv_kl(tau, rho):
    """g^{-1}(tau) = sup{beta in [0,1]: g(beta) <= tau}."""
    lo, hi = tau, 1.0
    if g_kl(1.0 - 1e-12, rho) <= tau:
        return 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if g_kl(mid, rho) <= tau:
            lo = mid
        else:
            hi = mid
    return lo


def level_quantile(scores, level):
    """Smallest score s with empirical CDF >= level, using the (n+1) correction."""
    n = len(scores)
    k = math.ceil((n + 1) * level)
    return float("inf") if k > n else float(np.sort(scores)[k - 1])


def ot_weights(cal_scores, s_low, s_high):
    """Minimise the label-free coverage-gap bound (Correia & Louizos 2025, eq. 7)."""
    S = torch.tensor(cal_scores, dtype=torch.float64)
    smax = int(max(cal_scores.max(), s_high.max())) + 1
    grid = torch.arange(1, smax + 1, dtype=torch.float64)
    ind = (S[:, None] <= grid[None, :]).double()                  # n x G
    F_dn = torch.tensor([(s_low <= g).mean() for g in range(1, smax + 1)], dtype=torch.float64)
    F_up = torch.tensor([(s_high <= g).mean() for g in range(1, smax + 1)], dtype=torch.float64)
    h = max(1.06 * float(S.std()) * len(S) ** (-0.2), 1.0)
    kern = torch.exp(-0.5 * ((grid[None, :] - S[:, None]) / h) ** 2) / (h * math.sqrt(2 * math.pi))
    theta = torch.zeros(len(S), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=OT_LR)
    for _ in range(OT_STEPS):
        w = torch.softmax(theta, 0)
        Fw = w @ ind
        pw = w @ kern
        loss = 0.5 * (pw * ((Fw - F_up).abs() + (Fw - F_dn).abs() + F_dn - F_up)).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return torch.softmax(theta, 0).detach().numpy()


def weighted_quantile_plain(scores_sorted, w_sorted, alpha):
    cum = np.cumsum(w_sorted) / w_sorted.sum()
    idx = int(np.searchsorted(cum, 1 - alpha - 1e-12, side="left"))
    return float("inf") if idx >= len(scores_sorted) else float(scores_sorted[idx])


ARCH_INDEX = {"GCN": 0, "GIN": 1, "GraphConv": 3}




METHODS = ["standard", "pooled", "graphmatch", "weighted", "weighted_split", "weighted_indep"]


def epn(d):
    return d.edge_index.size(1) / d.num_nodes


def run_one(di, ds, arch, si, base_offset, train_param):
    ai = ARCH_INDEX[arch]
    base = base_offset + 100000 * (di + 1) + 1000 * si
    train_g = make_graph(ds, base, train_param)
    C = int(train_g.y.max()) + 1
    model, idx, tr, va, te = train_model(arch, train_g, seed=base + 10 * ai, num_classes=C)
    std_nodes = [int(v) for v in idx["cal"].tolist() if int(train_g.y[v]) > 0]
    std_scores = np.array([p["score"] for p in explain_nodes(model, train_g, std_nodes)], float)

    # calibration family (identical to Weeks 16/17/20) and an independent fitting family
    family, cell_of, cell_epn = [], [], []
    fit_family = []
    for mi, m in enumerate(CAL_M):
        for ni, nz in enumerate(CAL_NOISE):
            cs = base + 100 + 10 * mi + ni
            g = add_noise(make_graph(ds, cs, m), nz, seed=cs + 500)
            new = explain_nodes(model, g, sample_motif_nodes(g, CAL_NODES, cs))
            family += new
            cell_of += [len(cell_epn)] * len(new)
            cell_epn.append(epn(g))
            fs = cs + 20000
            gfit = add_noise(make_graph(ds, fs, m), nz, seed=fs + 500)
            fit_family += explain_nodes(model, gfit, sample_motif_nodes(gfit, CAL_NODES, fs))
    cell_of = np.array(cell_of)
    Xcal = np.array([p["feat"] for p in family])
    Xfit = np.array([p["feat"] for p in fit_family])
    sc = np.array([p["score"] for p in family], dtype=np.float64)
    order = np.argsort(sc)
    s_sorted = sc[order]

    rng = np.random.default_rng(base + 7)
    perm = rng.permutation(len(family))
    c1, c2 = perm[: len(family) // 2], perm[len(family) // 2:]
    sc2 = sc[c2]
    order2 = np.argsort(sc2)
    s2_sorted = sc2[order2]

    run = {"dataset": ds, "arch": arch, "seed": si, "train_acc": round(tr, 4),
           "test_acc": round(te, 4), "rows": []}
    for kind, m, nz, ti in TESTS:
        per = {(me, a): [] for me in METHODS for a in ALPHAS}
        # independent unlabelled graph of this setting, used only to fit weighted_indep
        us = base + 30000 + 100 * ti
        gu = add_noise(make_graph(ds, us, m), nz, seed=us + 50)
        Xunl = np.array([t["feat"] for t in explain_nodes(model, gu, sample_motif_nodes(gu, N_TEST, us))])
        lg_ind = fit_lr(np.vstack([Xfit, Xunl]), np.r_[np.zeros(len(Xfit)), np.ones(len(Xunl))])
        r_ind = len(Xfit) / len(Xunl)
        w_ind_sorted = (np.exp(lg_ind(Xcal)) * r_ind)[order]
        for gi in range(N_TEST_GRAPHS):
            ts = base + 500 + 100 * ti + gi
            g = add_noise(make_graph(ds, ts, m), nz, seed=ts + 50)
            test = explain_nodes(model, g, sample_motif_nodes(g, N_TEST, ts))
            Xte = np.array([t["feat"] for t in test])
            logit = fit_lr(np.vstack([Xcal, Xte]), np.r_[np.zeros(len(Xcal)), np.ones(len(Xte))])
            ratio = len(Xcal) / len(Xte)
            w_sorted = (np.exp(logit(Xcal)) * ratio)[order]
            w_te = np.exp(logit(Xte)) * ratio
            w_te_ind = np.exp(lg_ind(Xte)) * r_ind
            q_rng = np.random.default_rng(ts + 9)
            qp = q_rng.permutation(len(test))
            halves = [qp[: len(test) // 2], qp[len(test) // 2:]]
            w_te_split = np.zeros(len(test))
            w_cal_split = [None, None]
            for h in range(2):
                fit_q = halves[1 - h]
                lg = fit_lr(np.vstack([Xcal[c1], Xte[fit_q]]), np.r_[np.zeros(len(c1)), np.ones(len(fit_q))])
                r2 = len(c1) / len(fit_q)
                w_cal_split[h] = (np.exp(lg(Xcal[c2])) * r2)[order2]
                w_te_split[halves[h]] = np.exp(lg(Xte[halves[h]])) * r2
            half_of = np.zeros(len(test), dtype=int)
            half_of[halves[1]] = 1
            gm_cell = int(np.argmin(np.abs(np.array(cell_epn) - epn(g))))
            gm_scores = sc[cell_of == gm_cell]
            for a in ALPHAS:
                q_fixed = {"standard": level_quantile(std_scores, 1 - a), "pooled": level_quantile(sc, 1 - a),
                           "graphmatch": level_quantile(gm_scores, 1 - a)}
                for me in METHODS:
                    cov, size, abst, whole, frac = [], [], [], [], []
                    for j, t in enumerate(test):
                        if me == "weighted":
                            q = weighted_q(s_sorted, w_sorted, w_te[j], a)
                        elif me == "weighted_split":
                            q = weighted_q(s2_sorted, w_cal_split[half_of[j]], w_te_split[j], a)
                        elif me == "weighted_indep":
                            q = weighted_q(s_sorted, w_ind_sorted, w_te_ind[j], a)
                        else:
                            q = q_fixed[me]
                        n_v = len(t["rank"])
                        k = int((t["rank"] <= q).sum())
                        cov.append(bool((t["rank"][t["gt"]] <= q).all()))
                        size.append(k)
                        abst.append(math.isinf(q))
                        whole.append(k >= n_v)
                        frac.append(k / n_v)
                    per[(me, a)].append((np.mean(cov), np.mean(size), np.mean(abst), np.mean(whole), np.mean(frac)))
        for a in ALPHAS:
            for me in METHODS:
                arr = np.array(per[(me, a)], dtype=float)
                run["rows"].append({"type": kind, "ba_m": m, "noise": nz, "method": me, "alpha": a,
                                    "coverage": round(float(arr[:, 0].mean()), 4),
                                    "avg_set_size": round(float(arr[:, 1].mean()), 2),
                                    "abstention_rate": round(float(arr[:, 2].mean()), 4),
                                    "whole_subgraph_rate": round(float(arr[:, 3].mean()), 4),
                                    "fraction_selected": round(float(arr[:, 4].mean()), 4)})
    return run


def boot_ci(vals, reps=10000, seed=0):
    vals = np.asarray(vals, float)
    g = np.random.default_rng(seed)
    means = vals[g.integers(0, len(vals), size=(reps, len(vals)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


t_start = time.time()
out = {}
ref = {}
for path in (W16_JSON, W17_JSON):
    if os.path.exists(path):
        with open(path) as f:
            for run in json.load(f)["runs"]:
                for r in run["rows"]:
                    if r["method"] == "weighted":
                        ref[(path, run["dataset"], run["arch"], run["seed"], r["ba_m"], r["noise"], r["alpha"])] = r

for fam, path_ref, datasets, cal_m, tests, offset, tparam in [
        ("ba", W16_JSON, ["house", "cycle", "grid"], [4, 5, 6],
         [("in-support", 5, 0.0, 0), ("calibrated cell", 4, 0.05, 2), ("held-out level", 5, 0.0625, 1),
          ("held-out level, other base", 4, 0.075, 4), ("beyond range", 5, 0.15, 3)], 0, 5),
        ("tree", W17_JSON, ["cycle", "grid"], [7, 8, 9],
         [("in-support", 8, 0.0, 0), ("calibrated cell", 7, 0.05, 2), ("held-out level", 8, 0.0625, 1),
          ("held-out level, other base", 7, 0.075, 4), ("beyond range", 8, 0.15, 3)], 500000, 8)]:
    FAMILY, CAL_M, TESTS = fam, cal_m, tests
    runs, same, compared = [], 0, 0
    n_total = len(datasets) * len(ARCHS) * len(SEEDS)
    print(f"\n{fam.upper()} graphs, {n_total} runs")
    for di, ds in enumerate(datasets):
        for arch in ARCHS:
            for si in SEEDS:
                t0 = time.time()
                run = run_one(di, ds, arch, si, offset, tparam)
                runs.append(run)
                for r in run["rows"]:
                    if r["method"] == "weighted":
                        k = (path_ref, ds, arch, si, r["ba_m"], r["noise"], r["alpha"])
                        if k in ref:
                            compared += 1
                            same += int(ref[k]["coverage"] == r["coverage"] and ref[k]["avg_set_size"] == r["avg_set_size"])
                out[fam] = {"runs": runs, "weighted_rows_identical_to_saved": same, "weighted_rows_compared": compared}
                with open(OUT_JSON, "w") as f:
                    json.dump(out, f, indent=2)
                print(f"[{len(runs):2d}/{n_total}] {fam}-{ds:5} {arch:9} seed {si} | test acc {run['test_acc']:.3f} | {time.time() - t0:.0f} s", flush=True)
    print(f"Check: weighted rows identical to saved Week {'16' if fam == 'ba' else '17'} results: {same} of {compared}")
    for a in ALPHAS:
        print(f"\n{fam.upper()} | TARGET {int(round((1 - a) * 100))}% | coverage mean (worst) [95% CI] / set size / abstain / whole-subgraph / fraction")
        print("method          | " + " | ".join(f"{k[0]} ({k[1]},{k[2]})" for k in TESTS))
        for me in METHODS:
            txt = []
            for kind, m, nz, ti in TESTS:
                rows = [r for run in runs for r in run["rows"] if r["type"] == kind and r["method"] == me and r["alpha"] == a]
                c = np.array([r["coverage"] for r in rows])
                lo, hi = boot_ci(c)
                f = lambda key: np.mean([r[key] for r in rows])
                txt.append("%.3f (%.3f) [%.3f,%.3f] / %.1f / %.2f / %.2f / %.2f" % (
                    c.mean(), c.min(), lo, hi, f("avg_set_size"), f("abstention_rate"), f("whole_subgraph_rate"), f("fraction_selected")))
            print(f"{me:15} | " + " | ".join(txt))
print(f"\nTotal time: {time.time() - t_start:.0f} s")
print("Saved:", OUT_JSON)

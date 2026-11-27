# week11_rich_calibration.py
# Week 11: weighted conformal with a RICHER calibration family + held-out shifts.
#
# Calibration family (graphs with ground truth, synthetic):
#   original graph cal nodes  +  fresh BA-Shapes graphs with
#   BA density m in {4, 5, 6}  x  extra-edge noise in {0, 2.5, 5, 10%},
#   CAL_GRAPHS graphs per cell, CAL_NODES motif nodes per graph.
# Test (5 graphs x 100 motif nodes each), including held-out combinations:
#   in-support : (5, 0), (5, 10%), (6, 0)
#   held-out   : (5, 6.25%), (4, 5% ), (6, 5%) -> mixes/levels never calibrated
#   beyond     : (5, 15%), (7, 0)               -> outside the calibration range
# Methods: standard (original graph only), pooled (whole family),
#          weighted (density-ratio weights on label-free features).
# New metric: abstention rate = share of test nodes whose set is the whole
# neighbourhood because no calibration data supports them (threshold = inf).
#
# Place in:  E:\conformal_gnn_project
# Run:       python week11_rich_calibration.py

import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.nn import GCNConv
from torch_geometric.utils import k_hop_subgraph

SEED = 42
ALPHAS = [0.05, 0.10, 0.20]
CAL_M = [4, 5, 6]
CAL_NOISE = [0.0, 0.025, 0.05, 0.10]
CAL_GRAPHS = 2
CAL_NODES = 60
TESTS = [("in-support", 5, 0.0), ("in-support", 5, 0.10), ("in-support", 6, 0.0),
         ("held-out", 5, 0.0625), ("held-out", 4, 0.05), ("held-out", 6, 0.05),
         ("beyond", 5, 0.15), ("beyond", 7, 0.0)]
N_GRAPHS = 5
N_TEST = 100
METHODS = ["standard", "pooled", "weighted"]
LR_LAMBDA = 1.0
TOPK_MASS = 12


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(CODE_DIR, "data")
RES_DIR = os.path.join(CODE_DIR, "results")
MODEL_DIR = os.path.join(RES_DIR, "models")


class GCN(torch.nn.Module):
    def __init__(self, in_dim, hid, out_dim):
        super().__init__()
        self.c1 = GCNConv(in_dim, hid)
        self.c2 = GCNConv(hid, hid)
        self.c3 = GCNConv(hid, hid)
        self.lin = torch.nn.Linear(hid, out_dim)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.c1(x, edge_index, edge_weight))
        x = F.relu(self.c2(x, edge_index, edge_weight))
        x = F.relu(self.c3(x, edge_index, edge_weight))
        return self.lin(x)


orig = torch.load(os.path.join(DATA_DIR, "bashapes_data.pt"),
                  map_location="cpu", weights_only=False)
orig.edge_mask = orig.edge_mask.bool()
splits = torch.load(os.path.join(DATA_DIR, "bashapes_splits.pt"),
                    map_location="cpu", weights_only=False)
model = GCN(orig.x.size(1), 20, int(orig.y.max()) + 1).to(device)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "gcn_bashapes.pt"),
                                 map_location=device, weights_only=True))
model.eval()


def new_graph(seed, m):
    set_seed(seed)
    d = ExplainerDataset(graph_generator=BAGraph(num_nodes=300, num_edges=m),
                         motif_generator="house", num_motifs=80)[0]
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


def explain_nodes(d, nodes):
    x, ei = d.x.to(device), d.edge_index.to(device)
    with torch.no_grad():
        preds = model(x, ei).argmax(1)
    gt_full = d.edge_mask.numpy()
    src = d.edge_index[0].numpy()
    dst = d.edge_index[1].numpy()
    deg = np.bincount(src, minlength=d.num_nodes).astype(np.float64)
    epn = d.num_edges / d.num_nodes
    out = []
    for v in nodes:
        w = torch.ones(ei.size(1), device=device, requires_grad=True)
        model(x, ei, w)[v, preds[v]].backward()
        imp_full = w.grad.detach().abs().cpu().numpy()
        _, _, _, sub = k_hop_subgraph(int(v), 3, d.edge_index, num_nodes=d.num_nodes)
        sub = sub.numpy()
        imp = imp_full[sub].astype(np.float64)
        gt = gt_full[sub]
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
        out.append({"rank": rank, "gt": gt, "n": n, "score": int(rank[gt].max()),
                    "correct": bool(preds[v].item() == int(d.y[v])), "feat": feat})
    return out


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


t0 = time.time()

# ---------- 1. Calibration family ----------
cal_nodes_orig = [v for v in splits["cal"].tolist() if int(orig.y[v]) > 0]
standard_pts = explain_nodes(orig, cal_nodes_orig)
family = list(standard_pts)
for m in CAL_M:
    for ni, nz in enumerate(CAL_NOISE):
        for gi in range(CAL_GRAPHS):
            cseed = 70000 + 1000 * m + 100 * ni + gi
            d = add_noise(new_graph(cseed, m), nz, seed=cseed + 50)
            rng = np.random.default_rng(cseed)
            motif = np.where(d.y.numpy() > 0)[0]
            nodes = rng.choice(motif, size=min(CAL_NODES, len(motif)), replace=False)
            family += explain_nodes(d, [int(v) for v in nodes])
print(f"Calibration: standard {len(standard_pts)} points, family {len(family)} points "
      f"({time.time() - t0:.0f} s)")
Xcal = np.array([p["feat"] for p in family])
cal_scores = np.array([p["score"] for p in family], dtype=np.float64)
order = np.argsort(cal_scores)
s_sorted = cal_scores[order]
thr = {a: {"standard": q_hat(np.array([p["score"] for p in standard_pts]), a),
           "pooled": q_hat(cal_scores, a)} for a in ALPHAS}

# ---------- 2. Test settings ----------
results = {"dataset": "BA-Shapes", "explainer": "Saliency", "score": "rankabs", "seed": SEED,
           "cal_m": CAL_M, "cal_noise": CAL_NOISE, "cal_graphs_per_cell": CAL_GRAPHS,
           "cal_nodes_per_graph": CAL_NODES, "n_family_points": len(family),
           "n_graphs": N_GRAPHS, "n_test_per_graph": N_TEST, "tests": []}
print("\ntype       | m | noise  | method   | alpha | coverage (mean+/-std) | correct-only | "
      "avg size | abstain | size (non-abstained)")
for ti, (kind, m, nz) in enumerate(TESTS):
    per = {(me, a): [] for me in METHODS for a in ALPHAS}
    stats = []
    for gi in range(N_GRAPHS):
        gseed = 90000 + 100 * ti + gi
        d = add_noise(new_graph(gseed, m), nz, seed=gseed + 50)
        rng = np.random.default_rng(gseed)
        motif = np.where(d.y.numpy() > 0)[0]
        tnodes = rng.choice(motif, size=min(N_TEST, len(motif)), replace=False)
        test = explain_nodes(d, [int(v) for v in tnodes])
        Xte = np.array([t["feat"] for t in test])
        logit = fit_lr(np.vstack([Xcal, Xte]), np.r_[np.zeros(len(Xcal)), np.ones(len(Xte))])
        ratio = len(Xcal) / len(Xte)
        w_cal = np.exp(logit(Xcal)) * ratio
        w_te = np.exp(logit(Xte)) * ratio
        w_sorted = w_cal[order]
        ess = float(w_cal.sum() ** 2 / (w_cal ** 2).sum())
        stats.append((np.mean([t["n"] for t in test]), np.mean([t["correct"] for t in test]),
                      d.num_edges / d.num_nodes, ess))
        corr = np.array([t["correct"] for t in test])
        for a in ALPHAS:
            for me in METHODS:
                cov, size, abst = [], [], []
                for j, t in enumerate(test):
                    q = (thr[a][me] if me != "weighted"
                         else weighted_q(s_sorted, w_sorted, w_te[j], a))
                    cov.append(bool((t["rank"][t["gt"]] <= q).all()))
                    size.append(int((t["rank"] <= q).sum()))
                    abst.append(math.isinf(q))
                cov, size, abst = np.array(cov), np.array(size, float), np.array(abst)
                per[(me, a)].append((cov.mean(), cov[corr].mean() if corr.any() else np.nan,
                                     size.mean(), abst.mean(),
                                     size[~abst].mean() if (~abst).any() else np.nan))
    nsub, acc, epn, ess = (float(np.mean([s[i] for s in stats])) for i in range(4))
    for a in ALPHAS:
        for me in METHODS:
            arr = np.array(per[(me, a)], dtype=float)
            row = {"type": kind, "ba_m": m, "noise": nz, "method": me, "alpha": a, "target": 1 - a,
                   "coverage_mean": round(float(arr[:, 0].mean()), 4),
                   "coverage_std": round(float(arr[:, 0].std()), 4),
                   "coverage_correct_only": round(float(np.nanmean(arr[:, 1])), 4),
                   "avg_set_size": round(float(arr[:, 2].mean()), 2),
                   "abstention_rate": round(float(arr[:, 3].mean()), 4),
                   "avg_size_non_abstained": (None if np.all(np.isnan(arr[:, 4]))
                                              else round(float(np.nanmean(arr[:, 4])), 2)),
                   "avg_subgraph_edges": round(nsub, 2), "model_acc": round(acc, 4),
                   "edges_per_node": round(epn, 3), "weighted_ess": round(ess, 1)}
            results["tests"].append(row)
            nab = "-" if row["avg_size_non_abstained"] is None else f"{row['avg_size_non_abstained']:.2f}"
            print(f"{kind:10} | {m} | {nz:6.4f} | {me:8} | {a:5.2f} | "
                  f"{row['coverage_mean']:.4f} +/- {row['coverage_std']:.4f}     | "
                  f"{row['coverage_correct_only']:.4f}       | {row['avg_set_size']:8.2f} | "
                  f"{row['abstention_rate']:.2f}    | {nab}")
    print(f"        test graphs: edges/node {epn:.3f}, avg subgraph {nsub:.1f}, model acc {acc:.4f}, "
          f"weighted ESS {ess:.1f} of {len(family)}  ({time.time() - t0:.0f} s)")

with open(os.path.join(RES_DIR, "week11_rich_calibration.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nTotal time: {time.time() - t0:.0f} s")
print("Saved:", os.path.join(RES_DIR, "week11_rich_calibration.json"))

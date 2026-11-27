# week10_weighted_conformal.py
# Week 10: weighted split conformal prediction (Tibshirani et al., 2019) for
# GNN explanation sets under covariate shift.
#
# Each calibration node gets a weight w(x) = P(test | x) / P(cal | x), estimated
# by a logistic-regression classifier on LABEL-FREE node features:
#   log subgraph size, log node degree, log mean neighbour degree,
#   graph edges-per-node, saliency entropy, saliency mass in top-12 edges.
# The test-node threshold is the weighted (1 - alpha) quantile of calibration
# scores (with the test point's own weight placed at +infinity).
#
# Same calibration family, test settings and seeds as Week 9, so the
# "standard" and "graphmatch" rows should reproduce Week 9 exactly.
#
# Place in:  E:\conformal_gnn_project
# Run:       python week10_weighted_conformal.py

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
CAL_LEVELS = [0.0, 0.025, 0.05, 0.10]
CAL_DRAWS = 3
SETTINGS = ([("noise", lv, 5) for lv in [0.0, 0.025, 0.05, 0.0625, 0.10, 0.15]] +
            [("density", 0.0, 4), ("density", 0.0, 6)])
N_GRAPHS = 5
N_TEST = 100
METHODS = ["standard", "graphmatch", "weighted"]
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
        # label-free features
        p = imp / imp.sum() if imp.sum() > 0 else np.full(n, 1.0 / n)
        ent = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(n)) if n > 1 else 0.0
        topm = float(np.sort(p)[::-1][:TOPK_MASS].sum())
        nbrs = d.edge_index[1].numpy()[src == v]
        feat = [math.log(n), math.log(deg[v] + 1), math.log(deg[nbrs].mean() + 1) if len(nbrs) else 0.0,
                epn, ent, topm]
        out.append({"rank": rank, "gt": gt, "n": n, "score": int(rank[gt].max()),
                    "correct": bool(preds[v].item() == int(d.y[v])), "feat": feat})
    return out, epn


def q_hat(scores, alpha):
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    return float("inf") if k > n else float(np.sort(scores)[k - 1])


def fit_lr(X, y, lam=LR_LAMBDA, iters=50):
    """L2-regularised logistic regression (Newton / IRLS), no sklearn needed."""
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
    return lambda Xn: Z_logit(Xn, w, mu, sd)


def Z_logit(Xn, w, mu, sd):
    Zn = np.hstack([np.ones((len(Xn), 1)), (Xn - mu) / sd])
    return np.clip(Zn @ w, -30, 30)


def weighted_q(scores_sorted, w_sorted, w_test, alpha):
    tot = w_sorted.sum() + w_test
    cum = np.cumsum(w_sorted) / tot
    idx = int(np.searchsorted(cum, 1 - alpha - 1e-12, side="left"))
    return float("inf") if idx >= len(scores_sorted) else float(scores_sorted[idx])


t0 = time.time()

# ---------- 1. Calibration family ----------
cal_nodes = [v for v in splits["cal"].tolist() if int(orig.y[v]) > 0]
cal_by_level, epn_by_level = {}, {}
for li, lvl in enumerate(CAL_LEVELS):
    pts, epns = [], []
    for dr in range(1 if lvl == 0 else CAL_DRAWS):
        g = add_noise(orig, lvl, seed=5000 + 10 * li + dr)
        p, e = explain_nodes(g, cal_nodes)
        pts += p
        epns.append(e)
    cal_by_level[lvl], epn_by_level[lvl] = pts, float(np.mean(epns))
    print(f"Calibration level {lvl:.3f}: {len(pts)} points, edges/node {epn_by_level[lvl]:.3f}")
pooled = [p for l in CAL_LEVELS for p in cal_by_level[l]]
Xcal = np.array([p["feat"] for p in pooled])
cal_scores = np.array([p["score"] for p in pooled], dtype=np.float64)

thr = {a: {"standard": q_hat(np.array([p["score"] for p in cal_by_level[0.0]]), a),
           "level": {l: q_hat(np.array([p["score"] for p in cal_by_level[l]]), a)
                     for l in CAL_LEVELS}} for a in ALPHAS}


def choose_level(epn):
    for l in CAL_LEVELS:
        if epn_by_level[l] >= epn - 1e-9:
            return l
    return CAL_LEVELS[-1]


# ---------- 2. Test settings ----------
results = {"dataset": "BA-Shapes", "explainer": "Saliency", "score": "rankabs", "seed": SEED,
           "cal_levels": CAL_LEVELS, "n_cal_points": len(pooled), "lr_lambda": LR_LAMBDA,
           "features": ["log_subgraph_edges", "log_degree", "log_mean_nbr_degree",
                        "graph_edges_per_node", "saliency_entropy", "saliency_top12_mass"],
           "n_graphs": N_GRAPHS, "n_test_per_graph": N_TEST, "settings": []}
print("\nshift   | level | m | method     | alpha | coverage (mean+/-std) | cov. correct-only | avg size")
for si, (kind, lvl, m) in enumerate(SETTINGS):
    per_graph = {(meth, a): [] for meth in METHODS for a in ALPHAS}
    stats = []
    for gi in range(N_GRAPHS):
        gseed = 20000 + 100 * si + gi                     # same seeds as Week 9
        d = add_noise(new_graph(gseed, m), lvl, seed=gseed + 50)
        rng = np.random.default_rng(gseed)
        motif = np.where(d.y.numpy() > 0)[0]
        tnodes = rng.choice(motif, size=min(N_TEST, len(motif)), replace=False)
        test, epn = explain_nodes(d, [int(v) for v in tnodes])
        pick = choose_level(epn)

        # density-ratio weights (label-free): calibration = 0, test = 1
        Xte = np.array([t["feat"] for t in test])
        logit = fit_lr(np.vstack([Xcal, Xte]),
                       np.r_[np.zeros(len(Xcal)), np.ones(len(Xte))])
        ratio = len(Xcal) / len(Xte)
        w_cal = np.exp(logit(Xcal)) * ratio
        w_te = np.exp(logit(Xte)) * ratio
        order = np.argsort(cal_scores)
        s_sorted, w_sorted = cal_scores[order], w_cal[order]
        ess = float(w_cal.sum() ** 2 / (w_cal ** 2).sum())
        stats.append((epn, np.mean([t["n"] for t in test]),
                      np.mean([t["correct"] for t in test]), pick, ess))

        corr = np.array([t["correct"] for t in test])
        for a in ALPHAS:
            for meth in METHODS:
                cov, size = [], []
                for j, t in enumerate(test):
                    if meth == "standard":
                        q = thr[a]["standard"]
                    elif meth == "graphmatch":
                        q = thr[a]["level"][pick]
                    else:
                        q = weighted_q(s_sorted, w_sorted, w_te[j], a)
                    cov.append(bool((t["rank"][t["gt"]] <= q).all()))
                    size.append(int((t["rank"] <= q).sum()))
                cov = np.array(cov)
                per_graph[(meth, a)].append((cov.mean(),
                                             cov[corr].mean() if corr.any() else np.nan,
                                             float(np.mean(size))))
    epn_m, nsub_m, acc_m, ess_m = (float(np.mean([s[i] for s in stats])) for i in (0, 1, 2, 4))
    picks = sorted(set(s[3] for s in stats))
    for a in ALPHAS:
        for meth in METHODS:
            arr = np.array(per_graph[(meth, a)], dtype=float)
            row = {"shift": kind, "noise": lvl, "ba_m": m, "method": meth, "alpha": a,
                   "target": 1 - a,
                   "coverage_mean": round(float(arr[:, 0].mean()), 4),
                   "coverage_std": round(float(arr[:, 0].std()), 4),
                   "coverage_correct_only_mean": round(float(np.nanmean(arr[:, 1])), 4),
                   "avg_set_size_mean": round(float(arr[:, 2].mean()), 2),
                   "avg_subgraph_edges": round(nsub_m, 2), "edges_per_node": round(epn_m, 3),
                   "model_acc": round(acc_m, 4), "graphmatch_levels": picks,
                   "weighted_ess": round(ess_m, 1)}
            results["settings"].append(row)
            print(f"{kind:7} | {lvl:5.3f} | {m} | {meth:10} | {a:5.2f} | "
                  f"{row['coverage_mean']:.4f} +/- {row['coverage_std']:.4f}     | "
                  f"{row['coverage_correct_only_mean']:.4f}            | {row['avg_set_size_mean']:.2f}")
    print(f"        test graphs: edges/node {epn_m:.3f}, avg subgraph {nsub_m:.1f}, model acc {acc_m:.4f}, "
          f"graphmatch level(s) {picks}, weighted ESS {ess_m:.1f} of {len(pooled)}  ({time.time() - t0:.0f} s)")

with open(os.path.join(RES_DIR, "week10_weighted_conformal.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nTotal time: {time.time() - t0:.0f} s")
print("Saved:", os.path.join(RES_DIR, "week10_weighted_conformal.json"))

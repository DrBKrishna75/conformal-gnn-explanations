# week8_shift_aware_calibration.py
# Week 8: shift-aware conformal calibration for GNN explanation sets.
#
# Calibration family: perturbed copies of the ORIGINAL graph with 0/10/20/40%
# extra random edges (3 random draws per level). Ground-truth house edges are
# unchanged by the perturbation, so calibration scores stay valid.
#
# Methods compared (Saliency explainer, conformal top-k "rankabs" score):
#   standard  : calibrate on the original graph only (Week 6 baseline)
#   pooled    : one threshold from all perturbed calibration copies
#   graphmatch: pick the calibration level whose edges-per-node matches the
#               test graph (label-free; rounds UP to be conservative)
#   mondrian  : thresholds per bin of the node's own 3-hop subgraph size
#               (label-free, per node)  <-- proposed
#
# Test: NEW BA-Shapes graphs with 0/10/20/30/40% extra edges
#       (30% was never used in calibration).
# All randomness is seeded (python, numpy, torch) so runs are reproducible.
#
# Place in:  E:\conformal_gnn_project
# Run:       python week8_shift_aware_calibration.py

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
CAL_LEVELS = [0.0, 0.10, 0.20, 0.40]
CAL_DRAWS = 3
TEST_LEVELS = [0.0, 0.10, 0.20, 0.30, 0.40]
N_TEST = 100
N_BINS = 5
METHODS = ["standard", "pooled", "graphmatch", "mondrian"]


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


# ---------- graph helpers ----------
def new_graph(seed):
    set_seed(seed)
    d = ExplainerDataset(graph_generator=BAGraph(num_nodes=300, num_edges=5),
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


def edges_per_node(d):
    return d.num_edges / d.num_nodes


def explain_nodes(d, nodes):
    """Saliency explanation + rank info for each node (motif nodes only)."""
    x, ei = d.x.to(device), d.edge_index.to(device)
    with torch.no_grad():
        preds = model(x, ei).argmax(1)
    gt_full = d.edge_mask.numpy()
    out = []
    for v in nodes:
        w = torch.ones(ei.size(1), device=device, requires_grad=True)
        model(x, ei, w)[v, preds[v]].backward()
        imp_full = w.grad.detach().abs().cpu().numpy()
        _, _, _, sub = k_hop_subgraph(int(v), 3, d.edge_index, num_nodes=d.num_nodes)
        sub = sub.numpy()
        imp = imp_full[sub].astype(np.float64)
        gt = gt_full[sub]
        order = np.argsort(-imp, kind="stable")
        rank = np.empty(len(imp), dtype=np.int64)
        rank[order] = np.arange(1, len(imp) + 1)
        out.append({"rank": rank, "gt": gt, "n": len(imp),
                    "score": int(rank[gt].max()),
                    "correct": bool(preds[v].item() == int(d.y[v]))})
    return out


def q_hat(scores, alpha):
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    return float("inf") if k > n else float(np.sort(scores)[k - 1])


t0 = time.time()

# ---------- 1. Calibration family (perturbed copies of the original graph) ----------
cal_nodes = [v for v in splits["cal"].tolist() if int(orig.y[v]) > 0]
cal_by_level = {}
epn_by_level = {}
for li, lvl in enumerate(CAL_LEVELS):
    pts, epns = [], []
    draws = 1 if lvl == 0 else CAL_DRAWS
    for dr in range(draws):
        g = add_noise(orig, lvl, seed=5000 + 10 * li + dr)
        pts += explain_nodes(g, cal_nodes)
        epns.append(edges_per_node(g))
    cal_by_level[lvl] = pts
    epn_by_level[lvl] = float(np.mean(epns))
    print(f"Calibration level {lvl:.2f}: {len(pts)} points, edges/node {epn_by_level[lvl]:.2f}, "
          f"avg subgraph {np.mean([p['n'] for p in pts]):.1f}")

pooled = [p for lvl in CAL_LEVELS for p in cal_by_level[lvl]]
standard = cal_by_level[0.0]

# Mondrian bins on log subgraph size (equal-count bins over the pooled set)
log_n = np.log([p["n"] for p in pooled])
edges = np.quantile(log_n, np.linspace(0, 1, N_BINS + 1))
edges[0], edges[-1] = -np.inf, np.inf
bin_of = lambda n: int(np.searchsorted(edges, np.log(n), side="right") - 1)
bins = {b: [p for p in pooled if bin_of(p["n"]) == b] for b in range(N_BINS)}
max_cal_n = max(p["n"] for p in pooled)
print("Mondrian bins (subgraph edges):",
      [f"{int(np.exp(edges[b])) if b else 0}-{int(np.exp(edges[b + 1])) if b < N_BINS - 1 else 'max'} "
       f"(n={len(bins[b])})" for b in range(N_BINS)])
print(f"Calibration done in {time.time() - t0:.0f} s")

thr = {a: {
    "standard": q_hat(np.array([p["score"] for p in standard]), a),
    "pooled": q_hat(np.array([p["score"] for p in pooled]), a),
    "level": {l: q_hat(np.array([p["score"] for p in cal_by_level[l]]), a) for l in CAL_LEVELS},
    "bin": {b: q_hat(np.array([p["score"] for p in bins[b]]), a) for b in range(N_BINS)},
} for a in ALPHAS}


def choose_level(epn):
    for l in CAL_LEVELS:                       # smallest level with >= density
        if epn_by_level[l] >= epn - 1e-9:
            return l
    return CAL_LEVELS[-1]


# ---------- 2. Test on new shifted graphs ----------
results = {"dataset": "BA-Shapes", "explainer": "Saliency", "score": "rankabs",
           "seed": SEED, "cal_levels": CAL_LEVELS, "cal_draws": CAL_DRAWS,
           "n_cal_nodes": len(cal_nodes), "n_bins": N_BINS,
           "cal_edges_per_node": epn_by_level, "test": []}
print("\nnoise | method     | alpha | coverage | avg size | (test: edges/node, avg subgraph, model acc)")
for ti, lvl in enumerate(TEST_LEVELS):
    base = new_graph(seed=2000 + ti)
    d = add_noise(base, lvl, seed=3000 + ti)
    rng = np.random.default_rng(SEED + ti)
    motif = np.where(d.y.numpy() > 0)[0]
    tnodes = rng.choice(motif, size=min(N_TEST, len(motif)), replace=False)
    test = explain_nodes(d, [int(v) for v in tnodes])
    epn = edges_per_node(d)
    nsub = float(np.mean([t["n"] for t in test]))
    acc = float(np.mean([t["correct"] for t in test]))
    beyond = float(np.mean([t["n"] > max_cal_n for t in test]))
    lvl_pick = choose_level(epn)
    for a in ALPHAS:
        for m in METHODS:
            cov, size = [], []
            for t in test:
                if m == "standard":
                    q = thr[a]["standard"]
                elif m == "pooled":
                    q = thr[a]["pooled"]
                elif m == "graphmatch":
                    q = thr[a]["level"][lvl_pick]
                else:
                    q = thr[a]["bin"][bin_of(t["n"])]
                cov.append(bool((t["rank"][t["gt"]] <= q).all()))
                size.append(int((t["rank"] <= q).sum()))
            c, s = float(np.mean(cov)), float(np.mean(size))
            results["test"].append({"noise": lvl, "method": m, "alpha": a, "target": 1 - a,
                                    "coverage": round(c, 4), "avg_set_size": round(s, 2),
                                    "test_edges_per_node": round(epn, 3),
                                    "avg_subgraph_edges": round(nsub, 2),
                                    "model_acc_test_nodes": round(acc, 4),
                                    "frac_beyond_cal_size": round(beyond, 4),
                                    "graphmatch_level": lvl_pick})
            print(f"{lvl:5.2f} | {m:10} | {a:5.2f} | {c:8.4f} | {s:8.2f}")
    print(f"      test graph: edges/node {epn:.2f}, avg subgraph {nsub:.1f}, model acc {acc:.4f}, "
          f"graphmatch level {lvl_pick:.2f}, beyond cal size {beyond:.2f}")

print(f"\nTotal time: {time.time() - t0:.0f} s")
with open(os.path.join(RES_DIR, "week8_shift_aware_calibration.json"), "w") as f:
    json.dump(results, f, indent=2)
print("Saved:", os.path.join(RES_DIR, "week8_shift_aware_calibration.json"))

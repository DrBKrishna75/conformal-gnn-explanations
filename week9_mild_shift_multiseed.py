# week9_mild_shift_multiseed.py
# Week 9: shift-aware calibration under MILD, realistic shifts, multiple seeds.
#
# Shift type A: extra random edges 0 / 2.5 / 5 / 6.25 (unseen) / 10 / 15% (beyond)
# Shift type B: BA density change (num_edges = 4 or 6 instead of 5), no noise
# Calibration family: perturbed copies of the ORIGINAL graph at 0/2.5/5/10%
# (3 draws per level). Methods: standard, pooled, graphmatch (edges-per-node).
# 5 test graphs per setting, 100 motif test nodes each -> mean +/- std.
# Coverage is also reported on correctly classified test nodes only.
#
# Place in:  E:\conformal_gnn_project
# Run:       python week9_mild_shift_multiseed.py

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
METHODS = ["standard", "pooled", "graphmatch"]


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

# ---------- 1. Calibration family ----------
cal_nodes = [v for v in splits["cal"].tolist() if int(orig.y[v]) > 0]
cal_by_level, epn_by_level = {}, {}
for li, lvl in enumerate(CAL_LEVELS):
    pts, epns = [], []
    for dr in range(1 if lvl == 0 else CAL_DRAWS):
        g = add_noise(orig, lvl, seed=5000 + 10 * li + dr)
        pts += explain_nodes(g, cal_nodes)
        epns.append(g.num_edges / g.num_nodes)
    cal_by_level[lvl], epn_by_level[lvl] = pts, float(np.mean(epns))
    print(f"Calibration level {lvl:.3f}: {len(pts)} points, edges/node {epn_by_level[lvl]:.3f}, "
          f"avg subgraph {np.mean([p['n'] for p in pts]):.1f}, "
          f"model acc {np.mean([p['correct'] for p in pts]):.4f}")
pooled = [p for l in CAL_LEVELS for p in cal_by_level[l]]

thr = {a: {"standard": q_hat(np.array([p["score"] for p in cal_by_level[0.0]]), a),
           "pooled": q_hat(np.array([p["score"] for p in pooled]), a),
           "level": {l: q_hat(np.array([p["score"] for p in cal_by_level[l]]), a)
                     for l in CAL_LEVELS}} for a in ALPHAS}


def choose_level(epn):
    for l in CAL_LEVELS:
        if epn_by_level[l] >= epn - 1e-9:
            return l
    return CAL_LEVELS[-1]


# ---------- 2. Test settings, N_GRAPHS graphs each ----------
results = {"dataset": "BA-Shapes", "explainer": "Saliency", "score": "rankabs", "seed": SEED,
           "cal_levels": CAL_LEVELS, "cal_edges_per_node": {str(k): v for k, v in epn_by_level.items()},
           "n_graphs": N_GRAPHS, "n_test_per_graph": N_TEST, "settings": []}
print("\nshift   | level | m | method     | alpha | coverage (mean+/-std) | cov. correct-only | avg size")
for si, (kind, lvl, m) in enumerate(SETTINGS):
    per_graph = {(meth, a): [] for meth in METHODS for a in ALPHAS}
    stats = []
    for gi in range(N_GRAPHS):
        gseed = 20000 + 100 * si + gi
        d = add_noise(new_graph(gseed, m), lvl, seed=gseed + 50)
        rng = np.random.default_rng(gseed)
        motif = np.where(d.y.numpy() > 0)[0]
        tnodes = rng.choice(motif, size=min(N_TEST, len(motif)), replace=False)
        test = explain_nodes(d, [int(v) for v in tnodes])
        epn = d.num_edges / d.num_nodes
        pick = choose_level(epn)
        stats.append((epn, np.mean([t["n"] for t in test]),
                      np.mean([t["correct"] for t in test]), pick))
        for a in ALPHAS:
            for meth in METHODS:
                q = (thr[a]["standard"] if meth == "standard" else
                     thr[a]["pooled"] if meth == "pooled" else thr[a]["level"][pick])
                cov = np.array([(t["rank"][t["gt"]] <= q).all() for t in test])
                corr = np.array([t["correct"] for t in test])
                size = np.mean([(t["rank"] <= q).sum() for t in test])
                per_graph[(meth, a)].append((cov.mean(),
                                             cov[corr].mean() if corr.any() else np.nan, size))
    epn_m, nsub_m, acc_m = (float(np.mean([s[i] for s in stats])) for i in range(3))
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
                   "model_acc": round(acc_m, 4), "graphmatch_levels": picks}
            results["settings"].append(row)
            print(f"{kind:7} | {lvl:5.3f} | {m} | {meth:10} | {a:5.2f} | "
                  f"{row['coverage_mean']:.4f} +/- {row['coverage_std']:.4f}     | "
                  f"{row['coverage_correct_only_mean']:.4f}            | {row['avg_set_size_mean']:.2f}")
    print(f"        test graphs: edges/node {epn_m:.3f}, avg subgraph {nsub_m:.1f}, "
          f"model acc {acc_m:.4f}, graphmatch level(s) {picks}  ({time.time() - t0:.0f} s)")

with open(os.path.join(RES_DIR, "week9_mild_shift_multiseed.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nTotal time: {time.time() - t0:.0f} s")
print("Saved:", os.path.join(RES_DIR, "week9_mild_shift_multiseed.json"))

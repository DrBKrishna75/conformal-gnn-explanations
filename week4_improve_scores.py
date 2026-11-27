# week4_improve_scores.py
# Week 4: make the conformal explanation sets smaller while keeping coverage.
# Uses the saved Week 2 GNNExplainer masks (no retraining, runs in seconds).
#
# Compares:
#   Ground truth (GT):
#     "all_motif" = every house edge inside the node's 3-hop subgraph (Week 3)
#     "own_house" = only the edges of the node's OWN house motif (correct target)
#   Nonconformity scores:
#     "minmax"   = 1 - min-max normalised importance (Week 3)
#     "rankfrac" = rank of edge / number of subgraph edges
#     "rankabs"  = rank of edge (conformal top-k: set = top-k edges)
#
# Place in:  E:\conformal_gnn_project
# Run:       python week4_improve_scores.py

import json
import math
import os

import numpy as np
import torch
from torch_geometric.utils import k_hop_subgraph

SEED = 42
ALPHAS = [0.05, 0.10, 0.20]
N_RESPLITS = 200

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(CODE_DIR, "data")
RES_DIR = os.path.join(CODE_DIR, "results")

data = torch.load(os.path.join(DATA_DIR, "bashapes_data.pt"),
                  map_location="cpu", weights_only=False)
w2 = torch.load(os.path.join(RES_DIR, "week2_explanations_bashapes.pt"),
                map_location="cpu", weights_only=False)
records, edge_masks = w2["records"], w2["edge_masks"]

ei = data.edge_index.cpu().numpy()
gt_all = data.edge_mask.bool().cpu().numpy()

# ---------- 1. Find each node's own house (connected parts of motif edges) ----------
parent = list(range(data.num_nodes))


def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


for e in np.where(gt_all)[0]:
    ra, rb = find(int(ei[0, e])), find(int(ei[1, e]))
    if ra != rb:
        parent[ra] = rb
comp = np.array([find(i) for i in range(data.num_nodes)])

# ---------- 2. Per-node data ----------
nodes = []
for r, em in zip(records, edge_masks):
    if r["label"] == 0:
        continue
    v = r["node"]
    _, _, _, sub = k_hop_subgraph(v, 3, data.edge_index, num_nodes=data.num_nodes)
    sub = sub.cpu().numpy()
    imp = em.cpu().numpy()[sub].astype(np.float64)
    src, dst = ei[0, sub], ei[1, sub]
    g_all = gt_all[sub]
    g_own = g_all & (comp[src] == comp[v]) & (comp[dst] == comp[v])
    if g_own.sum() == 0:
        continue
    n = len(imp)
    rng_ = imp.max() - imp.min()
    norm = (imp - imp.min()) / rng_ if rng_ > 0 else np.zeros(n)
    order = np.argsort(-imp, kind="stable")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(1, n + 1)
    nodes.append({
        "split": r["split"], "n": n,
        "gt": {"all_motif": g_all, "own_house": g_own},
        "val": {"minmax": 1.0 - norm, "rankfrac": rank / n,
                "rankabs": rank.astype(np.float64)},
    })

cal = [x for x in nodes if x["split"] == "cal"]
test = [x for x in nodes if x["split"] == "test"]
pool = cal + test

print(f"Motif nodes: calibration {len(cal)}, test {len(test)}")
print(f"Avg subgraph edges: {np.mean([x['n'] for x in nodes]):.2f}")
print(f"Avg GT edges per node: all_motif {np.mean([x['gt']['all_motif'].sum() for x in nodes]):.2f}, "
      f"own_house {np.mean([x['gt']['own_house'].sum() for x in nodes]):.2f}")
print(f"Nodes whose subgraph contains OTHER houses' edges: "
      f"{sum(int(x['gt']['all_motif'].sum() > x['gt']['own_house'].sum()) for x in nodes)} of {len(nodes)}")


def q_hat(scores, alpha):
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return float("inf")
    return float(np.sort(scores)[k - 1])


def node_score(x, s, g):
    return float(x["val"][s][x["gt"][g]].max())


def evaluate(ts, q, s, g):
    cov, size = [], []
    for x in ts:
        in_set = x["val"][s] <= q + 1e-12
        cov.append(bool(in_set[x["gt"][g]].all()))
        size.append(int(in_set.sum()))
    return float(np.mean(cov)), float(np.mean(size))


# ---------- 3. Compare all combinations ----------
results = {"dataset": "BA-Shapes", "explainer": "GNNExplainer", "seed": SEED,
           "n_cal": len(cal), "n_test": len(test), "runs": []}
print("\nGT        | score    | alpha | coverage | avg size | re-split coverage | re-split size")
for g in ["all_motif", "own_house"]:
    for s in ["minmax", "rankfrac", "rankabs"]:
        rng = np.random.default_rng(SEED)
        cal_scores = np.array([node_score(x, s, g) for x in cal])
        for a in ALPHAS:
            q = q_hat(cal_scores, a)
            cov, size = evaluate(test, q, s, g)
            rc, rs = [], []
            for _ in range(N_RESPLITS):
                perm = rng.permutation(len(pool))
                c = [pool[i] for i in perm[:len(cal)]]
                t = [pool[i] for i in perm[len(cal):]]
                qq = q_hat(np.array([node_score(x, s, g) for x in c]), a)
                cv, sz = evaluate(t, qq, s, g)
                rc.append(cv)
                rs.append(sz)
            results["runs"].append({
                "gt": g, "score": s, "alpha": a, "target": 1 - a,
                "q_hat": None if math.isinf(q) else round(q, 4),
                "coverage": round(cov, 4), "avg_set_size": round(size, 2),
                "resplit_coverage_mean": round(float(np.mean(rc)), 4),
                "resplit_coverage_std": round(float(np.std(rc)), 4),
                "resplit_size_mean": round(float(np.mean(rs)), 2),
            })
            print(f"{g:9} | {s:8} | {a:5.2f} | {cov:8.4f} | {size:8.2f} | "
                  f"{np.mean(rc):.4f} +/- {np.std(rc):.4f} | {np.mean(rs):.2f}")

with open(os.path.join(RES_DIR, "week4_improve_scores.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved:", os.path.join(RES_DIR, "week4_improve_scores.json"))

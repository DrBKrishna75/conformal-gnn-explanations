# week3_conformal_bashapes.py
# Week 3: split conformal prediction on GNNExplainer edge scores (BA-Shapes).
# For each motif node we build an "explanation set" of edges such that ALL
# ground-truth motif edges are inside the set with probability >= 1 - alpha.
#
# Nonconformity score of a calibration node = the threshold needed to cover
# all its ground-truth edges:  s_i = max over GT edges of (1 - normalised importance).
# Test set for a node = all edges in its 3-hop subgraph with (1 - importance) <= q_hat.
#
# Place in:  E:\conformal_gnn_project
# Run:       python week3_conformal_bashapes.py

import json
import math
import os

import numpy as np
import torch
from torch_geometric.utils import k_hop_subgraph

SEED = 42
ALPHAS = [0.05, 0.10, 0.20]
N_RESPLITS = 200          # repeated random cal/test splits for a coverage check
TOPK_BASELINE = 12        # a house motif has 6 undirected = 12 directed edges

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(CODE_DIR, "data")
RES_DIR = os.path.join(CODE_DIR, "results")

# ---------- 1. Load data and Week 2 explanations ----------
data = torch.load(os.path.join(DATA_DIR, "bashapes_data.pt"),
                  map_location="cpu", weights_only=False)
w2 = torch.load(os.path.join(RES_DIR, "week2_explanations_bashapes.pt"),
                map_location="cpu", weights_only=False)
records, edge_masks = w2["records"], w2["edge_masks"]
gt_edge = data.edge_mask.bool()

# ---------- 2. Per-node edge scores (motif nodes only) ----------
nodes = []   # each: dict(split, ncs (array per edge), gt (bool array), score)
for r, em in zip(records, edge_masks):
    if r["label"] == 0:
        continue                       # base nodes have no ground truth
    _, _, _, sub = k_hop_subgraph(r["node"], 3, data.edge_index,
                                  num_nodes=data.num_nodes)
    imp = em[sub].numpy().astype(np.float64)
    gt = gt_edge[sub].numpy()
    if gt.sum() == 0:
        continue
    rng_ = imp.max() - imp.min()
    norm = (imp - imp.min()) / rng_ if rng_ > 0 else np.zeros_like(imp)
    ncs = 1.0 - norm                   # low = important
    nodes.append({
        "node": r["node"], "split": r["split"], "ncs": ncs, "gt": gt,
        "imp": imp, "score": float(ncs[gt].max()),
    })

cal = [n for n in nodes if n["split"] == "cal"]
test = [n for n in nodes if n["split"] == "test"]
print(f"Motif nodes: calibration {len(cal)}, test {len(test)}")


def q_hat(scores, alpha):
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return 1.0                     # include everything
    return float(np.sort(scores)[k - 1])


def evaluate(test_nodes, q):
    cov, recall, size, frac = [], [], [], []
    for t in test_nodes:
        in_set = t["ncs"] <= q + 1e-12
        cov.append(bool(in_set[t["gt"]].all()))
        recall.append(float(in_set[t["gt"]].mean()))
        size.append(int(in_set.sum()))
        frac.append(float(in_set.mean()))
    return (float(np.mean(cov)), float(np.mean(recall)),
            float(np.mean(size)), float(np.mean(frac)))


def topk_baseline(test_nodes, k):
    cov, size = [], []
    for t in test_nodes:
        order = np.argsort(-t["imp"])[:k]
        in_set = np.zeros_like(t["gt"])
        in_set[order] = True
        cov.append(bool(in_set[t["gt"]].all()))
        size.append(int(in_set.sum()))
    return float(np.mean(cov)), float(np.mean(size))


# ---------- 3. Original split ----------
results = {"dataset": "BA-Shapes", "explainer": "GNNExplainer", "seed": SEED,
           "n_cal": len(cal), "n_test": len(test), "alphas": {}}
cal_scores = np.array([c["score"] for c in cal])
base_cov, base_size = topk_baseline(test, TOPK_BASELINE)
results["topk_baseline"] = {"k": TOPK_BASELINE, "coverage": round(base_cov, 4),
                            "avg_set_size": round(base_size, 2)}

print(f"\nBaseline top-{TOPK_BASELINE}: coverage {base_cov:.4f}, avg size {base_size:.2f}")
print("\nalpha | target | coverage | GT recall | avg set size | frac of subgraph")

rng = np.random.default_rng(SEED)
pool = cal + test
for a in ALPHAS:
    q = q_hat(cal_scores, a)
    cov, rec, size, frac = evaluate(test, q)

    # repeated random re-splits of the same pool (same sizes)
    rs_cov, rs_size = [], []
    for _ in range(N_RESPLITS):
        perm = rng.permutation(len(pool))
        c = [pool[i] for i in perm[:len(cal)]]
        t = [pool[i] for i in perm[len(cal):]]
        qq = q_hat(np.array([x["score"] for x in c]), a)
        cv, _, sz, _ = evaluate(t, qq)
        rs_cov.append(cv)
        rs_size.append(sz)

    results["alphas"][str(a)] = {
        "target_coverage": 1 - a, "q_hat": round(q, 4),
        "coverage": round(cov, 4), "gt_edge_recall": round(rec, 4),
        "avg_set_size": round(size, 2), "avg_frac_subgraph": round(frac, 4),
        "resplit_coverage_mean": round(float(np.mean(rs_cov)), 4),
        "resplit_coverage_std": round(float(np.std(rs_cov)), 4),
        "resplit_set_size_mean": round(float(np.mean(rs_size)), 2),
        "n_resplits": N_RESPLITS,
    }
    print(f"{a:5.2f} | {1-a:6.2f} | {cov:8.4f} | {rec:9.4f} | {size:12.2f} | {frac:.4f}")
    print(f"        re-splits ({N_RESPLITS}): coverage {np.mean(rs_cov):.4f} "
          f"+/- {np.std(rs_cov):.4f}, avg size {np.mean(rs_size):.2f}")

with open(os.path.join(RES_DIR, "week3_conformal_bashapes.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved:", os.path.join(RES_DIR, "week3_conformal_bashapes.json"))

# week5_compare_explainers.py
# Week 5: compare explainers by conformal explanation-set size at equal coverage.
# Explainers: GNNExplainer (saved from Week 2), Saliency, Integrated Gradients, Random.
# Conformal score: "rankabs" (conformal top-k), best in Week 4.
#
# Place in:  E:\conformal_gnn_project
# Run:       python week5_compare_explainers.py

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import k_hop_subgraph

SEED = 42
ALPHAS = [0.05, 0.10, 0.20]
N_RESPLITS = 200
IG_STEPS = 50

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(CODE_DIR, "data")
RES_DIR = os.path.join(CODE_DIR, "results")
MODEL_DIR = os.path.join(RES_DIR, "models")

data = torch.load(os.path.join(DATA_DIR, "bashapes_data.pt"),
                  map_location="cpu", weights_only=False)
w2 = torch.load(os.path.join(RES_DIR, "week2_explanations_bashapes.pt"),
                map_location="cpu", weights_only=False)
records, gnnexp_masks = w2["records"], w2["edge_masks"]
num_classes = int(data.y.max()) + 1


class GCN(torch.nn.Module):  # same layers as Week 1; edge_weight added for gradients
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


model = GCN(data.x.size(1), 20, num_classes).to(device)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "gcn_bashapes.pt"),
                                 map_location=device, weights_only=True))
model.eval()
x = data.x.to(device)
ei = data.edge_index.to(device)
E = ei.size(1)

with torch.no_grad():
    preds = model(x, ei).argmax(dim=1)


def saliency(v):
    w = torch.ones(E, device=device, requires_grad=True)
    out = model(x, ei, w)[v, preds[v]]
    out.backward()
    return w.grad.detach().abs().cpu()


def integrated_gradients(v):
    total = torch.zeros(E, device=device)
    for k in range(1, IG_STEPS + 1):
        w = torch.full((E,), k / IG_STEPS, device=device, requires_grad=True)
        out = model(x, ei, w)[v, preds[v]]
        out.backward()
        total += w.grad.detach()
    return (total / IG_STEPS).abs().cpu()   # (input - baseline) = 1


# ---------- 1. Compute explanations ----------
gen = torch.Generator().manual_seed(SEED)
masks = {"GNNExplainer": [], "Saliency": [], "IntegratedGrad": [], "Random": []}
t0 = time.time()
for r, gm in zip(records, gnnexp_masks):
    v = r["node"]
    masks["GNNExplainer"].append(gm)
    masks["Saliency"].append(saliency(v))
    masks["IntegratedGrad"].append(integrated_gradients(v))
    masks["Random"].append(torch.rand(E, generator=gen))
print(f"Explanations computed for {len(records)} nodes in {time.time() - t0:.1f} s")

ei_np = data.edge_index.numpy()
gt_all = data.edge_mask.bool().numpy()
subs = {}
for r in records:
    if r["label"] > 0:
        _, _, _, s = k_hop_subgraph(r["node"], 3, data.edge_index, num_nodes=data.num_nodes)
        subs[r["node"]] = s.numpy()


def auc_score(imp, gt):
    pos, neg = imp[gt], imp[~gt]
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def q_hat(scores, alpha):
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    return float("inf") if k > n else float(np.sort(scores)[k - 1])


def evaluate(ts, q):
    cov = [bool((t["rank"][t["gt"]] <= q).all()) for t in ts]
    size = [int((t["rank"] <= q).sum()) for t in ts]
    return float(np.mean(cov)), float(np.mean(size))


# ---------- 2. Conformal top-k for each explainer ----------
results = {"dataset": "BA-Shapes", "seed": SEED, "score": "rankabs", "explainers": {}}
print("\nexplainer      | test AUC | alpha | coverage | avg size | re-split coverage | re-split size")
for name, mlist in masks.items():
    nodes = []
    for r, m in zip(records, mlist):
        if r["label"] == 0:
            continue
        sub = subs[r["node"]]
        imp = m.numpy()[sub].astype(np.float64)
        gt = gt_all[sub]
        order = np.argsort(-imp, kind="stable")
        rank = np.empty(len(imp), dtype=np.int64)
        rank[order] = np.arange(1, len(imp) + 1)
        nodes.append({"split": r["split"], "rank": rank, "gt": gt,
                      "score": int(rank[gt].max()), "auc": auc_score(imp, gt)})
    cal = [t for t in nodes if t["split"] == "cal"]
    test = [t for t in nodes if t["split"] == "test"]
    pool = cal + test
    test_auc = float(np.mean([t["auc"] for t in test if t["auc"] is not None]))
    rng = np.random.default_rng(SEED)
    rows = []
    for a in ALPHAS:
        q = q_hat(np.array([t["score"] for t in cal]), a)
        cov, size = evaluate(test, q)
        rc, rs = [], []
        for _ in range(N_RESPLITS):
            perm = rng.permutation(len(pool))
            c = [pool[i] for i in perm[:len(cal)]]
            t = [pool[i] for i in perm[len(cal):]]
            cv, sz = evaluate(t, q_hat(np.array([z["score"] for z in c]), a))
            rc.append(cv)
            rs.append(sz)
        rows.append({"alpha": a, "target": 1 - a,
                     "k_hat": None if math.isinf(q) else int(q),
                     "coverage": round(cov, 4), "avg_set_size": round(size, 2),
                     "resplit_coverage_mean": round(float(np.mean(rc)), 4),
                     "resplit_coverage_std": round(float(np.std(rc)), 4),
                     "resplit_size_mean": round(float(np.mean(rs)), 2)})
        print(f"{name:14} | {test_auc:8.4f} | {a:5.2f} | {cov:8.4f} | {size:8.2f} | "
              f"{np.mean(rc):.4f} +/- {np.std(rc):.4f} | {np.mean(rs):.2f}")
    results["explainers"][name] = {"test_auc": round(test_auc, 4), "rows": rows}

with open(os.path.join(RES_DIR, "week5_compare_explainers.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved:", os.path.join(RES_DIR, "week5_compare_explainers.json"))

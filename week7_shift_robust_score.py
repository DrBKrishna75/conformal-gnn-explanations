# week7_shift_robust_score.py
# Week 7: a size-normalised conformal score that adapts to neighbourhood growth.
#   normrank: s = rank / n^beta, where n = number of subgraph edges and beta is
#             fitted on TRAINING motif nodes (not calibration), so the split
#             conformal guarantee is kept on the original distribution.
# Compared with rankabs, rankfrac, minmax on the SAME shifted graphs as Week 6.
# Explainer: Saliency (best in Week 5).
#
# Place in:  E:\conformal_gnn_project
# Run:       python week7_shift_robust_score.py

import json
import math
import os
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
NOISE_LEVELS = [0.0, 0.10, 0.20, 0.40]
N_TEST_PER_LEVEL = 80
N_RESPLITS = 200
SCORES = ["rankabs", "rankfrac", "minmax", "normrank"]

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


def make_shifted_graph(seed, noise_frac):  # identical to Week 6
    torch.manual_seed(seed)
    d = ExplainerDataset(graph_generator=BAGraph(num_nodes=300, num_edges=5),
                         motif_generator="house", num_motifs=80)[0]
    d.x = torch.ones((d.num_nodes, 10))
    d.edge_mask = d.edge_mask.bool()
    if noise_frac > 0:
        n_add = int(noise_frac * d.num_edges / 2)
        existing = set(zip(d.edge_index[0].tolist(), d.edge_index[1].tolist()))
        g = torch.Generator().manual_seed(seed + 1)
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


def saliency(d, v):
    x, ei = d.x.to(device), d.edge_index.to(device)
    with torch.no_grad():
        c = model(x, ei)[v].argmax()
    w = torch.ones(ei.size(1), device=device, requires_grad=True)
    model(x, ei, w)[v, c].backward()
    return w.grad.detach().abs().cpu().numpy()


def base_info(d, v):
    _, _, _, sub = k_hop_subgraph(v, 3, d.edge_index, num_nodes=d.num_nodes)
    sub = sub.numpy()
    imp = saliency(d, v)[sub].astype(np.float64)
    gt = d.edge_mask.numpy()[sub]
    n = len(imp)
    order = np.argsort(-imp, kind="stable")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(1, n + 1)
    rng_ = imp.max() - imp.min()
    norm = (imp - imp.min()) / rng_ if rng_ > 0 else np.zeros(n)
    return {"rank": rank.astype(np.float64), "gt": gt, "n": n, "norm": norm}


def add_values(t, beta):
    t["val"] = {"rankabs": t["rank"], "rankfrac": t["rank"] / t["n"],
                "minmax": 1.0 - t["norm"], "normrank": t["rank"] / (t["n"] ** beta)}
    t["score"] = {s: float(t["val"][s][t["gt"]].max()) for s in SCORES}
    return t


def q_hat(scores, alpha):
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    return float("inf") if k > n else float(np.sort(scores)[k - 1])


def evaluate(ts, q, s):
    cov = float(np.mean([(t["val"][s][t["gt"]] <= q + 1e-12).all() for t in ts]))
    size = float(np.mean([(t["val"][s] <= q + 1e-12).sum() for t in ts]))
    return cov, size


t0 = time.time()
# ---------- 1. Fit beta on TRAINING motif nodes ----------
train_nodes = [v for v in splits["train"].tolist() if int(orig.y[v]) > 0]
train = [base_info(orig, v) for v in train_nodes]
logn = np.log([t["n"] for t in train])
logr = np.log([t["rank"][t["gt"]].max() for t in train])
beta = float(np.clip(np.polyfit(logn, logr, 1)[0], 0.0, 1.0))
print(f"beta fitted on {len(train)} training motif nodes: {beta:.4f}")

# ---------- 2. Calibration and in-distribution check ----------
cal_nodes = [v for v in splits["cal"].tolist() if int(orig.y[v]) > 0]
test0_nodes = [v for v in splits["test"].tolist() if int(orig.y[v]) > 0]
cal = [add_values(base_info(orig, v), beta) for v in cal_nodes]
test0 = [add_values(base_info(orig, v), beta) for v in test0_nodes]
print(f"Calibration motif nodes: {len(cal)}, original test motif nodes: {len(test0)}")

results = {"dataset": "BA-Shapes", "explainer": "Saliency", "seed": SEED,
           "beta": round(beta, 4), "n_train_fit": len(train), "n_cal": len(cal),
           "in_distribution": [], "shift": []}

print("\nIN-DISTRIBUTION (original graph, 200 re-splits)")
print("score    | alpha | coverage | avg size")
pool = cal + test0
for s in SCORES:
    rng = np.random.default_rng(SEED)
    for a in ALPHAS:
        rc, rs = [], []
        for _ in range(N_RESPLITS):
            perm = rng.permutation(len(pool))
            c = [pool[i] for i in perm[:len(cal)]]
            t = [pool[i] for i in perm[len(cal):]]
            cv, sz = evaluate(t, q_hat(np.array([z["score"][s] for z in c]), a), s)
            rc.append(cv)
            rs.append(sz)
        results["in_distribution"].append({"score": s, "alpha": a,
                                           "coverage": round(float(np.mean(rc)), 4),
                                           "coverage_std": round(float(np.std(rc)), 4),
                                           "avg_set_size": round(float(np.mean(rs)), 2)})
        print(f"{s:8} | {a:5.2f} | {np.mean(rc):.4f} +/- {np.std(rc):.4f} | {np.mean(rs):.2f}")

# ---------- 3. Shifted graphs (same graphs and test nodes as Week 6) ----------
print("\nSHIFT (calibrated on original graph)")
print("score    | noise | alpha | coverage | avg size | avg subgraph")
for lvl, noise in enumerate(NOISE_LEVELS):
    d = make_shifted_graph(1000 + lvl, noise)
    rng = np.random.default_rng(SEED + lvl)
    motif = np.where(d.y.numpy() > 0)[0]
    test_nodes = rng.choice(motif, size=min(N_TEST_PER_LEVEL, len(motif)), replace=False)
    test = [add_values(base_info(d, int(v)), beta) for v in test_nodes]
    nsub = float(np.mean([t["n"] for t in test]))
    for s in SCORES:
        cal_scores = np.array([c["score"][s] for c in cal])
        for a in ALPHAS:
            cov, size = evaluate(test, q_hat(cal_scores, a), s)
            results["shift"].append({"score": s, "noise": noise, "alpha": a,
                                     "coverage": round(cov, 4), "avg_set_size": round(size, 2),
                                     "avg_subgraph_edges": round(nsub, 2)})
            print(f"{s:8} | {noise:5.2f} | {a:5.2f} | {cov:8.4f} | {size:8.2f} | {nsub:.2f}")

print(f"\nTotal time: {time.time() - t0:.0f} s")
with open(os.path.join(RES_DIR, "week7_shift_robust_score.json"), "w") as f:
    json.dump(results, f, indent=2)
print("Saved:", os.path.join(RES_DIR, "week7_shift_robust_score.json"))

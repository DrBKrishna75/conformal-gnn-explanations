# week6_shift_bashapes.py
# Week 6: does the conformal guarantee survive distribution shift?
# Calibrate on the ORIGINAL graph's calibration nodes (Week 1 split),
# then test on NEW BA-Shapes graphs with 0%, 10%, 20%, 40% extra random edges.
# Explainers: Saliency and Integrated Gradients. Score: conformal top-k (rankabs).
#
# Place in:  E:\conformal_gnn_project
# Run:       python week6_shift_bashapes.py

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
IG_STEPS = 50

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
splits = torch.load(os.path.join(DATA_DIR, "bashapes_splits.pt"),
                    map_location="cpu", weights_only=False)
num_classes = int(orig.y.max()) + 1
model = GCN(orig.x.size(1), 20, num_classes).to(device)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "gcn_bashapes.pt"),
                                 map_location=device, weights_only=True))
model.eval()


def make_shifted_graph(seed, noise_frac):
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


def explain(d, v, method):
    x, ei = d.x.to(device), d.edge_index.to(device)
    E = ei.size(1)
    with torch.no_grad():
        c = model(x, ei)[v].argmax()
    if method == "Saliency":
        w = torch.ones(E, device=device, requires_grad=True)
        model(x, ei, w)[v, c].backward()
        return w.grad.detach().abs().cpu().numpy()
    total = torch.zeros(E, device=device)
    for k in range(1, IG_STEPS + 1):
        w = torch.full((E,), k / IG_STEPS, device=device, requires_grad=True)
        model(x, ei, w)[v, c].backward()
        total += w.grad.detach()
    return (total / IG_STEPS).abs().cpu().numpy()


def node_info(d, v, imp_full):
    _, _, _, sub = k_hop_subgraph(v, 3, d.edge_index, num_nodes=d.num_nodes)
    sub = sub.numpy()
    imp = imp_full[sub].astype(np.float64)
    gt = d.edge_mask.numpy()[sub]
    order = np.argsort(-imp, kind="stable")
    rank = np.empty(len(imp), dtype=np.int64)
    rank[order] = np.arange(1, len(imp) + 1)
    pos, neg = imp[gt], imp[~gt]
    auc = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
    return {"rank": rank, "gt": gt, "score": int(rank[gt].max()), "auc": auc, "n": len(imp)}


def q_hat(scores, alpha):
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    return float("inf") if k > n else float(np.sort(scores)[k - 1])


def accuracy(d):
    with torch.no_grad():
        p = model(d.x.to(device), d.edge_index.to(device)).argmax(1).cpu()
    return float((p == d.y).float().mean())


# ---------- 1. Calibration on the original graph ----------
orig.edge_mask = orig.edge_mask.bool()
cal_nodes = [v for v in splits["cal"].tolist() if int(orig.y[v]) > 0]
t0 = time.time()
cal = {m: [node_info(orig, v, explain(orig, v, m)) for v in cal_nodes]
       for m in ["Saliency", "IntegratedGrad"]}
print(f"Calibration: {len(cal_nodes)} motif nodes ({time.time() - t0:.0f} s)")

# ---------- 2. Test on shifted graphs ----------
results = {"dataset": "BA-Shapes", "seed": SEED, "shift": "random extra edges",
           "n_cal": len(cal_nodes), "levels": []}
print("\nexplainer      | noise | model acc | test AUC | avg subgraph | alpha | coverage | avg size")
for lvl, noise in enumerate(NOISE_LEVELS):
    d = make_shifted_graph(1000 + lvl, noise)
    acc = accuracy(d)
    rng = np.random.default_rng(SEED + lvl)
    motif = np.where(d.y.numpy() > 0)[0]
    test_nodes = rng.choice(motif, size=min(N_TEST_PER_LEVEL, len(motif)), replace=False)
    for m in ["Saliency", "IntegratedGrad"]:
        test = [node_info(d, int(v), explain(d, int(v), m)) for v in test_nodes]
        auc = float(np.mean([t["auc"] for t in test]))
        nsub = float(np.mean([t["n"] for t in test]))
        cal_scores = np.array([c["score"] for c in cal[m]])
        for a in ALPHAS:
            q = q_hat(cal_scores, a)
            cov = float(np.mean([(t["rank"][t["gt"]] <= q).all() for t in test]))
            size = float(np.mean([(t["rank"] <= q).sum() for t in test]))
            results["levels"].append({
                "explainer": m, "noise": noise, "model_acc": round(acc, 4),
                "test_auc": round(auc, 4), "avg_subgraph_edges": round(nsub, 2),
                "alpha": a, "target": 1 - a,
                "k_hat": None if math.isinf(q) else int(q),
                "coverage": round(cov, 4), "avg_set_size": round(size, 2),
            })
            print(f"{m:14} | {noise:5.2f} | {acc:9.4f} | {auc:8.4f} | {nsub:12.2f} | "
                  f"{a:5.2f} | {cov:8.4f} | {size:8.2f}")
    print(f"  (noise {noise:.2f} done, {time.time() - t0:.0f} s)")

with open(os.path.join(RES_DIR, "week6_shift_bashapes.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved:", os.path.join(RES_DIR, "week6_shift_bashapes.json"))

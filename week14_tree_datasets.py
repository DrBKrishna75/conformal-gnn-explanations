# week14_tree_datasets.py
# Week 14: does weighted conformal hold on a DIFFERENT graph family (trees)?
#
#   Datasets : Tree-Cycles (6-cycle motifs), Tree-Grids (3x3 grid motifs) on a
#              balanced binary tree (depth 8 = 511 nodes) + 80 motifs
#   Models   : GCN, GIN, GraphConv        Seeds: 3  => 18 runs
#   Calibration family: tree depth in {7, 8, 9} x added edges {0, 2.5, 5, 10%}
#   Tests (3 graphs x 100 motif nodes each):
#     in-support (depth 8, 0%), held-out (depth 8, 6.25%),
#     held-out (depth 7, 5%), beyond (depth 8, 15%)
#   Explainer: Saliency (edge-mask gradients). Methods: standard, pooled, weighted.
# Results are saved after every run.
#
# Place in:  E:\conformal_gnn_project
# Run:       python week14_tree_datasets.py
# Expected time: about 5-10 minutes on the RTX 3050.

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
from torch_geometric.datasets.graph_generator import GraphGenerator
from torch_geometric.datasets.motif_generator import CycleMotif, GridMotif, HouseMotif
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.nn import GCNConv, GINConv, GraphConv
from torch_geometric.utils import k_hop_subgraph

DATASETS = ["cycle", "grid"]
ARCHS = ["GCN", "GIN", "GraphConv"]
SEEDS = [0, 1, 2]
ALPHAS = [0.05, 0.10, 0.20]
CAL_M = [7, 8, 9]          # tree depth
CAL_NOISE = [0.0, 0.025, 0.05, 0.10]
CAL_NODES = 60
TRAIN_DEPTH = 8
TESTS = [("in-support", 8, 0.0), ("held-out", 8, 0.0625),
         ("held-out", 7, 0.05), ("beyond", 8, 0.15)]
N_TEST_GRAPHS = 3
N_TEST = 100
METHODS = ["standard", "pooled", "weighted"]
EPOCHS = 1000
HID = 20
LR_LAMBDA = 1.0
TOPK_MASS = 12
N_HOPS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR = os.path.join(CODE_DIR, "results")
os.makedirs(RES_DIR, exist_ok=True)
OUT_JSON = os.path.join(RES_DIR, "week14_tree_datasets.json")


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


# ---------- graphs ----------
class TreeGraph(GraphGenerator):
    """Balanced binary tree of the given depth (depth 8 = 511 nodes)."""
    def __init__(self, depth):
        super().__init__()
        self.depth = depth

    def __call__(self):
        n = 2 ** (self.depth + 1) - 1
        child = torch.arange(1, n)
        parent = (child - 1) // 2
        ei = torch.stack([torch.cat([parent, child]), torch.cat([child, parent])])
        return Data(edge_index=ei, num_nodes=n)


def make_graph(dataset, seed, depth):
    set_seed(seed)
    motif = {"house": HouseMotif(), "cycle": CycleMotif(6), "grid": GridMotif()}[dataset]
    d = ExplainerDataset(graph_generator=TreeGraph(depth),
                         motif_generator=motif, num_motifs=80)[0]
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


# ---------- main loop ----------
t_start = time.time()
all_runs = []
for di, ds in enumerate(DATASETS):
    for ai, arch in enumerate(ARCHS):
        for si in SEEDS:
            t0 = time.time()
            base = 500000 + 100000 * (di + 1) + 1000 * si          # graphs shared by models within a seed
            train_g = make_graph(ds, base, TRAIN_DEPTH)
            C = int(train_g.y.max()) + 1
            model, idx, tr, va, te = train_model(arch, train_g, seed=base + 10 * ai, num_classes=C)

            # calibration: standard (training graph's cal split) + family (fresh graphs)
            std_nodes = [int(v) for v in idx["cal"].tolist() if int(train_g.y[v]) > 0]
            standard_pts = explain_nodes(model, train_g, std_nodes)
            family = []
            for mi, m in enumerate(CAL_M):
                for ni, nz in enumerate(CAL_NOISE):
                    cs = base + 100 + 10 * mi + ni
                    g = add_noise(make_graph(ds, cs, m), nz, seed=cs + 500)
                    family += explain_nodes(model, g, sample_motif_nodes(g, CAL_NODES, cs))
            Xcal = np.array([p["feat"] for p in family])
            cal_scores = np.array([p["score"] for p in family], dtype=np.float64)
            order = np.argsort(cal_scores)
            s_sorted = cal_scores[order]
            thr = {a: {"standard": q_hat(np.array([p["score"] for p in standard_pts]), a),
                       "pooled": q_hat(cal_scores, a)} for a in ALPHAS}

            run = {"dataset": ds, "arch": arch, "seed": si, "train_acc": round(tr, 4),
                   "val_acc": round(va, 4), "test_acc": round(te, 4),
                   "n_standard_cal": len(standard_pts), "n_family": len(family), "tests": []}
            for ti, (kind, m, nz) in enumerate(TESTS):
                per = {(me, a): [] for me in METHODS for a in ALPHAS}
                accs = []
                for gi in range(N_TEST_GRAPHS):
                    ts = base + 500 + 100 * ti + gi
                    g = add_noise(make_graph(ds, ts, m), nz, seed=ts + 50)
                    test = explain_nodes(model, g, sample_motif_nodes(g, N_TEST, ts))
                    accs.append(np.mean([t["correct"] for t in test]))
                    Xte = np.array([t["feat"] for t in test])
                    logit = fit_lr(np.vstack([Xcal, Xte]),
                                   np.r_[np.zeros(len(Xcal)), np.ones(len(Xte))])
                    ratio = len(Xcal) / len(Xte)
                    w_sorted = (np.exp(logit(Xcal)) * ratio)[order]
                    w_te = np.exp(logit(Xte)) * ratio
                    for a in ALPHAS:
                        for me in METHODS:
                            cov, size, abst = [], [], []
                            for j, t in enumerate(test):
                                q = (thr[a][me] if me != "weighted"
                                     else weighted_q(s_sorted, w_sorted, w_te[j], a))
                                cov.append(bool((t["rank"][t["gt"]] <= q).all()))
                                size.append(int((t["rank"] <= q).sum()))
                                abst.append(math.isinf(q))
                            per[(me, a)].append((np.mean(cov), np.mean(size), np.mean(abst)))
                for a in ALPHAS:
                    for me in METHODS:
                        arr = np.array(per[(me, a)], dtype=float)
                        run["tests"].append({"type": kind, "base_param": m, "noise": nz, "method": me,
                                             "alpha": a, "coverage": round(float(arr[:, 0].mean()), 4),
                                             "avg_set_size": round(float(arr[:, 1].mean()), 2),
                                             "abstention_rate": round(float(arr[:, 2].mean()), 4),
                                             "model_acc_test_nodes": round(float(np.mean(accs)), 4)})
            all_runs.append(run)
            with open(OUT_JSON, "w") as f:
                json.dump({"runs": all_runs}, f, indent=2)
            w10 = [r for r in run["tests"] if r["method"] == "weighted" and r["alpha"] == 0.10]
            s10 = [r for r in run["tests"] if r["method"] == "standard" and r["alpha"] == 0.10]
            w_txt = " ".join("%.2f" % r["coverage"] for r in w10)
            s_txt = " ".join("%.2f" % r["coverage"] for r in s10)
            print(f"[{len(all_runs):2d}/{len(DATASETS) * len(ARCHS) * len(SEEDS)}] {ds:5} {arch:3} seed {si} | acc train {tr:.3f} test {te:.3f} | "
                  f"90% cov weighted {w_txt} | standard {s_txt} | {time.time() - t0:.0f} s")

# ---------- summary over all 27 runs ----------
print("\nSUMMARY over all runs (mean coverage / mean set size / min coverage across runs)")
print("type       | depth | noise  | method   | alpha | coverage | set size | min coverage | abstain")
summary = []
for kind, m, nz in TESTS:
    for a in ALPHAS:
        for me in METHODS:
            rows = [r for run in all_runs for r in run["tests"]
                    if r["type"] == kind and r["base_param"] == m and r["noise"] == nz
                    and r["method"] == me and r["alpha"] == a]
            c = np.array([r["coverage"] for r in rows])
            s = np.array([r["avg_set_size"] for r in rows])
            ab = np.array([r["abstention_rate"] for r in rows])
            summary.append({"type": kind, "base_param": m, "noise": nz, "method": me, "alpha": a,
                            "coverage_mean": round(float(c.mean()), 4),
                            "coverage_min": round(float(c.min()), 4),
                            "set_size_mean": round(float(s.mean()), 2),
                            "abstention_mean": round(float(ab.mean()), 4)})
            print(f"{kind:10} | {m:5d} | {nz:6.4f} | {me:8} | {a:5.2f} | {c.mean():8.4f} | "
                  f"{s.mean():8.2f} | {c.min():12.4f} | {ab.mean():.2f}")

print("\nPER DATASET x MODEL, weighted at 90% target (mean over 3 seeds): coverage per test setting")
for ds in DATASETS:
    for arch in ARCHS:
        runs = [r for r in all_runs if r["dataset"] == ds and r["arch"] == arch]
        accs = np.mean([r["test_acc"] for r in runs])
        covs = []
        for kind, m, nz in TESTS:
            covs.append(np.mean([t["coverage"] for r in runs for t in r["tests"]
                                 if t["type"] == kind and t["base_param"] == m and t["noise"] == nz
                                 and t["method"] == "weighted" and t["alpha"] == 0.10]))
        print(f"{ds:5} {arch:3} | model test acc {accs:.3f} | " + " ".join(f"{c:.3f}" for c in covs))

with open(OUT_JSON, "w") as f:
    json.dump({"runs": all_runs, "summary": summary}, f, indent=2)
print(f"\nTotal time: {time.time() - t_start:.0f} s")
print("Saved:", OUT_JSON)

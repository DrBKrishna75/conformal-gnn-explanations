# week19_explainers.py
# Week 19: does the method work for other explainers?
#
#   Same BA protocol as Week 16 (CPU, deterministic), now with four explainers:
#     Saliency, Integrated Gradients (50 steps), GNNExplainer (PyG, 100 epochs), Random.
#   For every explainer: rank-depth scores, its own label-free features (entropy and
#   top-12 mass are computed from that explainer's importances), and the methods
#   standard / pooled / weighted.  Also records explainer AUC without shift, which
#   repeats the Table 6 comparison over many models and seeds.
#
# Place in:  E:\conformal_gnn_project
# Run:       python week19_explainers.py
# Expected time: about 2 hours (CPU). Progress is printed after every run (18 runs).

import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.datasets.motif_generator import CycleMotif, GridMotif, HouseMotif
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.nn import GCNConv, GINConv, GraphConv
from torch_geometric.utils import k_hop_subgraph

DATASETS = ["house", "grid"]
ARCHS = ["GCN", "GIN", "GraphConv"]
ARCH_INDEX = {"GCN": 0, "GIN": 1, "GraphConv": 3}     # same model seeds as Week 16
SEEDS = [0, 1, 2]
EXPLAINERS = ["Saliency", "IntegratedGradients", "GNNExplainer", "Random"]
ALPHAS = [0.05, 0.10, 0.20]
CAL_M = [4, 5, 6]
CAL_NOISE = [0.0, 0.025, 0.05, 0.10]
CAL_NODES = 30          # 30 per cell (Week 15: 15 per cell already suffices)
TESTS = [("in-support", 5, 0.0), ("held-out", 5, 0.0625),
         ("held-out", 4, 0.05), ("beyond", 5, 0.15)]
N_TEST_GRAPHS = 2
N_TEST = 60
METHODS = ["standard", "pooled", "weighted"]
EPOCHS = 1000
HID = 20
LR_LAMBDA = 1.0
TOPK_MASS = 12
N_HOPS = 3
IG_STEPS = 50
GNNEXP_EPOCHS = 100

QUICK = len(sys.argv) > 1 and sys.argv[1] == "quick"
if QUICK:
    DATASETS, ARCHS, SEEDS, CAL_NODES, N_TEST_GRAPHS, N_TEST, EPOCHS = ["house"], ["GCN"], [0], 8, 1, 15, 300

device = torch.device("cpu")
torch.use_deterministic_algorithms(True, warn_only=True)
print("Running on:", device, "| quick test" if QUICK else "")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR = os.path.join(ROOT, "results")
os.makedirs(RES_DIR, exist_ok=True)
OUT_JSON = os.path.join(RES_DIR, "week19_quick.json" if QUICK else "week19_explainers.json")


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


# ---------- graphs (identical to Week 16) ----------
def make_graph(dataset, seed, m):
    set_seed(seed)
    motif = {"house": HouseMotif(), "cycle": CycleMotif(6), "grid": GridMotif()}[dataset]
    d = ExplainerDataset(graph_generator=BAGraph(num_nodes=300, num_edges=m),
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


# ---------- models (identical to Week 16) ----------
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
    x, ei, y = d.x, d.edge_index, d.y
    best_val, best_state = -1.0, None

    def acc(split):
        model.eval()
        with torch.no_grad():
            p = model(x, ei).argmax(1)
        return (p[idx[split]] == y[idx[split]]).float().mean().item()

    for ep in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()
        F.cross_entropy(model(x, ei)[idx["train"]], y[idx["train"]]).backward()
        opt.step()
        if ep % 10 == 0:
            va = acc("val")
            if va >= best_val:
                best_val = va
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, idx, acc("train"), acc("val"), acc("test")


# ---------- explainers (computed on the exact 3-hop computation subgraph) ----------
def edge_grad(model, xs, sei, c, yhat, scale):
    mask = torch.nn.Parameter(torch.full((sei.size(1),), float(scale)))
    set_masks(model, mask, sei, apply_sigmoid=False)
    model(xs, sei)[c, yhat].backward()
    clear_masks(model)
    return mask.grad.detach().double()


def importances(name, model, xs, sei, c, yhat, node_seed):
    if name == "Saliency":
        return edge_grad(model, xs, sei, c, yhat, 1.0).abs().numpy()
    if name == "IntegratedGradients":
        tot = torch.zeros(sei.size(1), dtype=torch.float64)
        for k in range(1, IG_STEPS + 1):
            tot += edge_grad(model, xs, sei, c, yhat, k / IG_STEPS)
        return (tot / IG_STEPS).abs().numpy()
    if name == "GNNExplainer":
        set_seed(node_seed)
        exp = Explainer(model=model, algorithm=GNNExplainer(epochs=GNNEXP_EPOCHS),
                        explanation_type="model", edge_mask_type="object", node_mask_type=None,
                        model_config=dict(mode="multiclass_classification", task_level="node",
                                          return_type="raw"))
        return exp(xs, sei, index=c).edge_mask.detach().double().numpy()
    g = np.random.default_rng(node_seed)
    return g.random(sei.size(1))


def explain_nodes(model, d, nodes, seed):
    src, dst = d.edge_index[0].numpy(), d.edge_index[1].numpy()
    deg = np.bincount(src, minlength=d.num_nodes).astype(np.float64)
    epn = d.num_edges / d.num_nodes
    out = []
    for i, v in enumerate(nodes):
        subset, sei, mapping, emask = k_hop_subgraph(int(v), N_HOPS, d.edge_index, relabel_nodes=True,
                                                     num_nodes=d.num_nodes)
        gt = d.edge_mask[emask].numpy()
        if gt.sum() == 0:
            continue
        xs, c = d.x[subset], int(mapping)
        with torch.no_grad():
            yhat = int(model(xs, sei)[c].argmax())
        nbrs = dst[src == v]
        base_feat = [math.log(len(gt)), math.log(deg[v] + 1),
                     math.log(deg[nbrs].mean() + 1) if len(nbrs) else 0.0, epn]
        rec = {"gt": gt, "correct": bool(yhat == int(d.y[v]))}
        for name in EXPLAINERS:
            imp = importances(name, model, xs, sei, c, yhat, seed * 1000 + i)
            n = len(imp)
            order = np.argsort(-imp, kind="stable")
            rank = np.empty(n, dtype=np.int64)
            rank[order] = np.arange(1, n + 1)
            p = imp / imp.sum() if imp.sum() > 0 else np.full(n, 1.0 / n)
            ent = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(n)) if n > 1 else 0.0
            topm = float(np.sort(p)[::-1][:TOPK_MASS].sum())
            pos, neg = imp[gt], imp[~gt]
            auc = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()) \
                if len(neg) else None
            rec[name] = {"rank": rank, "score": int(rank[gt].max()), "feat": base_feat + [ent, topm], "auc": auc}
        out.append(rec)
    return out


def sample_motif_nodes(d, k, seed):
    rng = np.random.default_rng(seed)
    motif = np.where(d.y.numpy() > 0)[0]
    return [int(v) for v in rng.choice(motif, size=min(k, len(motif)), replace=False)]


# ---------- conformal (identical to Week 16) ----------
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


def level_quantile(scores, level):
    n = len(scores)
    k = math.ceil((n + 1) * level)
    return float("inf") if k > n else float(np.sort(scores)[k - 1])


# ---------- main loop ----------
t_start = time.time()
all_runs = []
n_total = len(DATASETS) * len(ARCHS) * len(SEEDS)
ds_index = {"house": 0, "cycle": 1, "grid": 2}
for ds in DATASETS:
    di = ds_index[ds]
    for arch in ARCHS:
        ai = ARCH_INDEX[arch]
        for si in SEEDS:
            t0 = time.time()
            base = 100000 * (di + 1) + 1000 * si
            train_g = make_graph(ds, base, 5)
            C = int(train_g.y.max()) + 1
            model, idx, tr, va, te = train_model(arch, train_g, seed=base + 10 * ai, num_classes=C)
            std_nodes = [int(v) for v in idx["cal"].tolist() if int(train_g.y[v]) > 0]
            std_raw = explain_nodes(model, train_g, std_nodes, base + 1)
            fam_raw = []
            for mi, m in enumerate(CAL_M):
                for ni, nz in enumerate(CAL_NOISE):
                    cs = base + 100 + 10 * mi + ni
                    g = add_noise(make_graph(ds, cs, m), nz, seed=cs + 500)
                    fam_raw += explain_nodes(model, g, sample_motif_nodes(g, CAL_NODES, cs), cs)
            tests = []
            for ti, (kind, m, nz) in enumerate(TESTS):
                graphs = []
                for gi in range(N_TEST_GRAPHS):
                    ts = base + 500 + 100 * ti + gi
                    g = add_noise(make_graph(ds, ts, m), nz, seed=ts + 50)
                    graphs.append(explain_nodes(model, g, sample_motif_nodes(g, N_TEST, ts), ts))
                tests.append((kind, m, nz, graphs))

            run = {"dataset": ds, "arch": arch, "seed": si, "train_acc": round(tr, 4), "test_acc": round(te, 4),
                   "n_cal": len(fam_raw), "rows": [], "auc": {}}
            for name in EXPLAINERS:
                std = np.array([r[name]["score"] for r in std_raw], float)
                Xcal = np.array([r[name]["feat"] for r in fam_raw])
                sc = np.array([r[name]["score"] for r in fam_raw], float)
                order = np.argsort(sc, kind="stable")
                s_sorted = sc[order]
                for kind, m, nz, graphs in tests:
                    per = {(me, a): [] for me in METHODS for a in ALPHAS}
                    aucs = []
                    for test in graphs:
                        aucs += [t[name]["auc"] for t in test if t[name]["auc"] is not None]
                        Xte = np.array([t[name]["feat"] for t in test])
                        logit = fit_lr(np.vstack([Xcal, Xte]), np.r_[np.zeros(len(Xcal)), np.ones(len(Xte))])
                        ratio = len(Xcal) / len(Xte)
                        w_cal = np.exp(logit(Xcal)) * ratio
                        w_te = np.exp(logit(Xte)) * ratio
                        w_sorted = w_cal[order]
                        for a in ALPHAS:
                            q_fixed = {"standard": level_quantile(std, 1 - a), "pooled": level_quantile(sc, 1 - a)}
                            for me in METHODS:
                                cov, size, abst = [], [], []
                                for j, t in enumerate(test):
                                    q = weighted_q(s_sorted, w_sorted, w_te[j], a) if me == "weighted" else q_fixed[me]
                                    rk = t[name]["rank"]
                                    cov.append(bool((rk[t["gt"]] <= q).all()))
                                    size.append(int((rk <= q).sum()))
                                    abst.append(math.isinf(q))
                                per[(me, a)].append((np.mean(cov), np.mean(size), np.mean(abst)))
                    run["auc"][f"{name}|{kind}|{m}|{nz}"] = round(float(np.mean(aucs)), 4)
                    for a in ALPHAS:
                        for me in METHODS:
                            arr = np.array(per[(me, a)], dtype=float)
                            run["rows"].append({"explainer": name, "type": kind, "ba_m": m, "noise": nz,
                                                "method": me, "alpha": a,
                                                "coverage": round(float(arr[:, 0].mean()), 4),
                                                "avg_set_size": round(float(arr[:, 1].mean()), 2),
                                                "abstention_rate": round(float(arr[:, 2].mean()), 4)})
            all_runs.append(run)
            with open(OUT_JSON, "w") as f:
                json.dump({"runs": all_runs}, f, indent=2)
            print(f"[{len(all_runs):2d}/{n_total}] {ds:5} {arch:9} seed {si} | acc train {tr:.3f} test {te:.3f} | "
                  f"{time.time() - t0:.0f} s", flush=True)

# ---------- summary ----------
summary = []
keys = [(k, m, nz) for k, m, nz in TESTS]
print("\nEXPLAINER AUC (mean over runs; 'in-support' = no shift)")
print("explainer            | " + " | ".join(f"{k} ({m},{nz})" for k, m, nz in keys))
for name in EXPLAINERS:
    vals = []
    for k, m, nz in keys:
        a = [run["auc"][f"{name}|{k}|{m}|{nz}"] for run in all_runs]
        vals.append("%.3f (min %.3f)" % (np.mean(a), np.min(a)))
        summary.append({"explainer": name, "type": k, "ba_m": m, "noise": nz, "auc_mean": round(float(np.mean(a)), 4),
                        "auc_min": round(float(np.min(a)), 4)})
    print(f"{name:20} | " + " | ".join(vals))
for a in ALPHAS:
    print(f"\nTARGET {int(round((1 - a) * 100))}% | coverage mean (worst run) / set size / abstain")
    print("explainer            method   | " + " | ".join(f"{k} ({m},{nz})" for k, m, nz in keys))
    for name in EXPLAINERS:
        for me in METHODS:
            txt = []
            for k, m, nz in keys:
                rows = [r for run in all_runs for r in run["rows"] if r["explainer"] == name and r["type"] == k
                        and r["ba_m"] == m and r["noise"] == nz and r["method"] == me and r["alpha"] == a]
                c = np.array([r["coverage"] for r in rows])
                s = np.array([r["avg_set_size"] for r in rows])
                ab = np.array([r["abstention_rate"] for r in rows])
                summary.append({"explainer": name, "method": me, "type": k, "ba_m": m, "noise": nz, "alpha": a,
                                "coverage_mean": round(float(c.mean()), 4), "coverage_min": round(float(c.min()), 4),
                                "set_size_mean": round(float(s.mean()), 2), "abstention_mean": round(float(ab.mean()), 4)})
                txt.append("%.3f (%.3f) / %.1f / %.2f" % (c.mean(), c.min(), s.mean(), ab.mean()))
            print(f"{name:20} {me:8} | " + " | ".join(txt))
print("\nModel test accuracy: mean %.4f, min %.4f" % (np.mean([r["test_acc"] for r in all_runs]),
                                                       np.min([r["test_acc"] for r in all_runs])))
with open(OUT_JSON, "w") as f:
    json.dump({"runs": all_runs, "summary": summary}, f, indent=2)
print(f"\nTotal time: {time.time() - t_start:.0f} s")
print("Saved:", OUT_JSON)

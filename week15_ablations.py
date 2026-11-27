# week15_ablations.py
# Week 15: ablations of weighted conformal explanation sets.
#
#   Same 27 runs as Weeks 12-13 (BA-Shapes, BA-Cycles, BA-Grids x GCN, GIN,
#   GraphConv x 3 seeds; same graphs, seeds and models), so the "all features /
#   full family" rows should reproduce Week 13 exactly.
#   A. Feature ablation: which label-free features the density-ratio weights need
#      (all 6; drop each one; structure only; explanation only).
#   B. Calibration-family ablation: how much and which calibration data is needed
#      (full; noise-only cells; density-only cells; 50% and 25% of nodes per cell).
#   Tests: in-support (5,0), held-out (5,6.25%), held-out (4,5%), beyond (5,15%).
#   Targets 95% and 90%. Saved after every run.
#
# Place in:  E:\conformal_gnn_project
# Run:       python week15_ablations.py
# Expected time: about 8-12 minutes on the RTX 3050.

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
from torch_geometric.datasets.motif_generator import CycleMotif, GridMotif, HouseMotif
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.nn import GCNConv, GINConv, GraphConv
from torch_geometric.utils import k_hop_subgraph

DATASETS = ["house", "cycle", "grid"]
ARCHS = ["GCN", "GIN", "GraphConv"]
ARCH_INDEX = {"GCN": 0, "GIN": 1, "GraphConv": 3}   # model seed offset (3 = new)
SEEDS = [0, 1, 2]
ALPHAS = [0.05, 0.10]
CAL_M = [4, 5, 6]
CAL_NOISE = [0.0, 0.025, 0.05, 0.10]
CAL_NODES = 60
TESTS = [("in-support", 5, 0.0), ("held-out", 5, 0.0625),
         ("held-out", 4, 0.05), ("beyond", 5, 0.15)]
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
OUT_JSON = os.path.join(RES_DIR, "week15_ablations.json")


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


# ---------- graphs ----------
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



FEATURE_NAMES = ["log_subgraph", "log_degree", "log_nbr_degree", "edges_per_node",
                 "sal_entropy", "sal_top12"]
FEATURE_SETS = {"all": [0, 1, 2, 3, 4, 5]}
for fi, fn in enumerate(FEATURE_NAMES):
    FEATURE_SETS["no_" + fn] = [j for j in range(6) if j != fi]
FEATURE_SETS["structure_only"] = [0, 1, 2, 3]
FEATURE_SETS["explanation_only"] = [4, 5]
FAMILY_SETS = ["full", "noise_only", "density_only", "half", "quarter"]


def family_mask(fam, cells, rng):
    keep = np.ones(len(cells), dtype=bool)
    if fam == "noise_only":
        keep = np.array([c[0] == 5 for c in cells])
    elif fam == "density_only":
        keep = np.array([c[1] == 0.0 for c in cells])
    elif fam in ("half", "quarter"):
        frac = 0.5 if fam == "half" else 0.25
        keep = np.zeros(len(cells), dtype=bool)
        for cell in sorted(set(cells)):
            idx = [i for i, c in enumerate(cells) if c == cell]
            k = max(1, int(round(frac * len(idx))))
            keep[rng.choice(idx, size=k, replace=False)] = True
    return keep


def eval_weighted(Xcal, cal_scores, tests, a):
    """Weighted conformal on given calibration features/scores; tests = list of (Xte, points)."""
    order = np.argsort(cal_scores)
    s_sorted = cal_scores[order]
    cov, size, abst = [], [], []
    for Xte, test in tests:
        logit = fit_lr(np.vstack([Xcal, Xte]), np.r_[np.zeros(len(Xcal)), np.ones(len(Xte))])
        ratio = len(Xcal) / len(Xte)
        w_sorted = (np.exp(logit(Xcal)) * ratio)[order]
        w_te = np.exp(logit(Xte)) * ratio
        for j, t in enumerate(test):
            q = weighted_q(s_sorted, w_sorted, w_te[j], a)
            cov.append(bool((t["rank"][t["gt"]] <= q).all()))
            size.append(int((t["rank"] <= q).sum()))
            abst.append(math.isinf(q))
    return float(np.mean(cov)), float(np.mean(size)), float(np.mean(abst))


ARCH_INDEX = {"GCN": 0, "GIN": 1, "GraphConv": 3}
t_start = time.time()
all_runs = []
n_total = len(DATASETS) * len(ARCHS) * len(SEEDS)
for di, ds in enumerate(DATASETS):
    for arch in ARCHS:
        ai = ARCH_INDEX[arch]
        for si in SEEDS:
            t0 = time.time()
            base = 100000 * (di + 1) + 1000 * si          # same as Weeks 12-13
            train_g = make_graph(ds, base, 5)
            C = int(train_g.y.max()) + 1
            model, idx, tr, va, te = train_model(arch, train_g, seed=base + 10 * ai, num_classes=C)

            family, cells = [], []
            for mi, m in enumerate(CAL_M):
                for ni, nz in enumerate(CAL_NOISE):
                    cs = base + 100 + 10 * mi + ni
                    g = add_noise(make_graph(ds, cs, m), nz, seed=cs + 500)
                    pts = explain_nodes(model, g, sample_motif_nodes(g, CAL_NODES, cs))
                    family += pts
                    cells += [(m, nz)] * len(pts)
            Xall = np.array([p["feat"] for p in family])
            sall = np.array([p["score"] for p in family], dtype=np.float64)

            test_sets = {}
            for ti, (kind, m, nz) in enumerate(TESTS):
                lst = []
                for gi in range(N_TEST_GRAPHS):
                    ts = base + 500 + 100 * ti + gi
                    g = add_noise(make_graph(ds, ts, m), nz, seed=ts + 50)
                    test = explain_nodes(model, g, sample_motif_nodes(g, N_TEST, ts))
                    lst.append((np.array([t["feat"] for t in test]), test))
                test_sets[(kind, m, nz)] = lst

            run = {"dataset": ds, "arch": arch, "seed": si, "test_acc": round(te, 4), "rows": []}
            for key, lst in test_sets.items():
                for a in ALPHAS:
                    for fname, cols in FEATURE_SETS.items():
                        c, s, ab = eval_weighted(Xall[:, cols], sall,
                                                 [(X[:, cols], t) for X, t in lst], a)
                        run["rows"].append({"ablation": "features", "variant": fname, "type": key[0],
                                            "ba_m": key[1], "noise": key[2], "alpha": a,
                                            "coverage": round(c, 4), "avg_set_size": round(s, 2),
                                            "abstention_rate": round(ab, 4)})
                    for fam in FAMILY_SETS:
                        rng = np.random.default_rng(base + 7)
                        keep = family_mask(fam, cells, rng)
                        c, s, ab = eval_weighted(Xall[keep], sall[keep], lst, a)
                        run["rows"].append({"ablation": "family", "variant": fam, "type": key[0],
                                            "ba_m": key[1], "noise": key[2], "alpha": a,
                                            "n_cal": int(keep.sum()),
                                            "coverage": round(c, 4), "avg_set_size": round(s, 2),
                                            "abstention_rate": round(ab, 4)})
            all_runs.append(run)
            with open(OUT_JSON, "w") as f:
                json.dump({"runs": all_runs}, f, indent=2)
            print(f"[{len(all_runs):2d}/{n_total}] {ds:5} {arch:9} seed {si} | test acc {te:.3f} | "
                  f"{time.time() - t0:.0f} s")

# ---------- summaries ----------
summary = []
for abl, variants in (("features", list(FEATURE_SETS)), ("family", FAMILY_SETS)):
    for a in ALPHAS:
        print(f"\nABLATION: {abl} | target {int(round((1 - a) * 100))}% | "
              "coverage mean (min) / set size / abstain, per test setting")
        print("variant              | " + " | ".join(f"{k[0]} ({k[1]},{k[2]})" for k in TESTS))
        for v in variants:
            cells_txt = []
            for kind, m, nz in TESTS:
                rows = [r for run in all_runs for r in run["rows"]
                        if r["ablation"] == abl and r["variant"] == v and r["type"] == kind
                        and r["ba_m"] == m and r["noise"] == nz and r["alpha"] == a]
                c = np.array([r["coverage"] for r in rows])
                s = np.array([r["avg_set_size"] for r in rows])
                ab = np.array([r["abstention_rate"] for r in rows])
                summary.append({"ablation": abl, "variant": v, "type": kind, "ba_m": m, "noise": nz,
                                "alpha": a, "coverage_mean": round(float(c.mean()), 4),
                                "coverage_min": round(float(c.min()), 4),
                                "set_size_mean": round(float(s.mean()), 2),
                                "abstention_mean": round(float(ab.mean()), 4)})
                cells_txt.append("%.3f (%.3f) / %.1f / %.2f" % (c.mean(), c.min(), s.mean(), ab.mean()))
            print(f"{v:20} | " + " | ".join(cells_txt))

with open(OUT_JSON, "w") as f:
    json.dump({"runs": all_runs, "summary": summary}, f, indent=2)
print(f"\nTotal time: {time.time() - t_start:.0f} s")
print("Saved:", OUT_JSON)

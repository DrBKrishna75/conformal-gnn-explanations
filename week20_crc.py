# week20_crc.py
# Week 20: conformal RISK CONTROL for explanation sets (partial coverage).
#
#   Full coverage (Weeks 16-17) asks the set to contain EVERY target edge.
#   Risk control asks the set to miss, on average, at most a fraction alpha of
#   the target edges:  E[ 1 - |E* in C_k| / |E*| ] <= alpha   (Angelopoulos et al. 2024).
#   Loss L_v(k) = fraction of target edges of v outside its top-k edges (non-increasing in k).
#     split CRC   : smallest k with ( sum_i L_i(k) + 1 ) / (n + 1) <= alpha
#     weighted CRC: smallest k with  sum_i p_i L_i(k) + p_test * 1 <= alpha  (same weights
#                   as Algorithm 1; abstains exactly when p_test > alpha, as before)
#   Same 27 BA runs (Week 16) and 18 tree runs (Week 17): same graphs, seeds, models (CPU).
#   Methods: standard / pooled / weighted (full coverage, must reproduce Weeks 16-17)
#            standard_crc / pooled_crc / weighted_crc (new).
#   Metrics: full coverage, recall (mean fraction of target edges inside the set),
#            set size, abstention, whole-subgraph rate, fraction of subgraph selected.
#
# Place in:  E:\conformal_gnn_project
# Run:       python week20_crc.py
# Expected time: about 10-20 minutes (CPU).

import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph, GraphGenerator
from torch_geometric.datasets.motif_generator import CycleMotif, GridMotif, HouseMotif
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.nn import GCNConv, GINConv, GraphConv
from torch_geometric.utils import k_hop_subgraph

ARCHS = ["GCN", "GIN", "GraphConv"]
ARCH_INDEX = {"GCN": 0, "GIN": 1, "GraphConv": 3}
SEEDS = [0, 1, 2]
ALPHAS = [0.05, 0.10, 0.20]
CAL_NOISE = [0.0, 0.025, 0.05, 0.10]
CAL_NODES = 60
N_TEST_GRAPHS = 3
N_TEST = 100
EPOCHS = 1000
HID = 20
LR_LAMBDA = 1.0
TOPK_MASS = 12
N_HOPS = 3
METHODS = ["standard", "pooled", "weighted", "standard_crc", "pooled_crc", "weighted_crc"]

FAMILIES = {
    "ba": {"datasets": ["house", "cycle", "grid"], "cal_m": [4, 5, 6], "offset": 0, "train_param": 5,
           "tests": [("in-support", 5, 0.0), ("held-out", 5, 0.0625), ("held-out", 4, 0.05), ("beyond", 5, 0.15)],
           "ref": "week16_baselines.json"},
    "tree": {"datasets": ["cycle", "grid"], "cal_m": [7, 8, 9], "offset": 500000, "train_param": 8,
             "tests": [("in-support", 8, 0.0), ("held-out", 8, 0.0625), ("held-out", 7, 0.05), ("beyond", 8, 0.15)],
             "ref": "week17_trees_all_methods.json"},
}

device = torch.device("cpu")
torch.use_deterministic_algorithms(True, warn_only=True)
print("Running on:", device)

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR = os.path.join(CODE_DIR, "results")
OUT_JSON = os.path.join(RES_DIR, "week20_crc.json")
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "week20_output.txt")
open(LOG_FILE, "w").close()


def log(*a):
    s = " ".join(str(t) for t in a)
    print(s, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(s + "\n")


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


# ---------- graphs (identical to Weeks 16-17) ----------
class TreeGraph(GraphGenerator):
    def __init__(self, depth):
        super().__init__()
        self.depth = depth

    def __call__(self):
        n = 2 ** (self.depth + 1) - 1
        child = torch.arange(1, n)
        parent = (child - 1) // 2
        ei = torch.stack([torch.cat([parent, child]), torch.cat([child, parent])])
        return Data(edge_index=ei, num_nodes=n)


def make_graph(family, dataset, seed, m):
    set_seed(seed)
    motif = {"house": HouseMotif(), "cycle": CycleMotif(6), "grid": GridMotif()}[dataset]
    gen = BAGraph(num_nodes=300, num_edges=m) if family == "ba" else TreeGraph(m)
    d = ExplainerDataset(graph_generator=gen, motif_generator=motif, num_motifs=80)[0]
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


# ---------- models (identical to Weeks 16-17) ----------
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


# ---------- explanations + features (identical to Weeks 16-17) ----------
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


# ---------- conformal: full coverage (identical to Weeks 16-17) ----------
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


# ---------- conformal: risk control (new) ----------
def crc_curve(recs, w):
    """Positions where calibration losses drop, and the cumulative weighted drop.
    Node i with m_i target edges at ranks g_i1..g_im loses w_i/m_i of weighted loss
    at each of those ranks, so sum_i w_i L_i(k) = sum_i w_i - cum_drop(k)."""
    pos = np.concatenate([r["rank"][r["gt"]] for r in recs]).astype(np.float64)
    val = np.concatenate([np.full(int(r["gt"].sum()), w[i] / r["gt"].sum()) for i, r in enumerate(recs)])
    o = np.argsort(pos, kind="stable")
    return pos[o], np.cumsum(val[o])


def crc_k(pos, cum, W, w_test, alpha):
    """Smallest k with (sum_i w_i L_i(k) + w_test * 1) / (W + w_test) <= alpha."""
    tot = W + w_test
    need = (1 - alpha) * tot
    idx = int(np.searchsorted(cum, need - 1e-9 * tot, side="left"))
    return float("inf") if idx >= len(pos) else float(pos[idx])


def node_metrics(t, q):
    gr = t["rank"][t["gt"]]
    n = len(t["rank"])
    size = n if math.isinf(q) else int(min(q, n))
    return (bool((gr <= q).all()), float((gr <= q).mean()), size, math.isinf(q), size >= n, size / n)


# ---------- one run ----------
def run_one(family, di, ds, arch, si):
    F_ = FAMILIES[family]
    ai = ARCH_INDEX[arch]
    base = F_["offset"] + 100000 * (di + 1) + 1000 * si
    train_g = make_graph(family, ds, base, F_["train_param"])
    C = int(train_g.y.max()) + 1
    model, idx, tr, va, te = train_model(arch, train_g, seed=base + 10 * ai, num_classes=C)
    std_nodes = [int(v) for v in idx["cal"].tolist() if int(train_g.y[v]) > 0]
    std = explain_nodes(model, train_g, std_nodes)
    std_scores = np.array([p["score"] for p in std], float)

    fam = []
    for mi, m in enumerate(F_["cal_m"]):
        for ni, nz in enumerate(CAL_NOISE):
            cs = base + 100 + 10 * mi + ni
            g = add_noise(make_graph(family, ds, cs, m), nz, seed=cs + 500)
            fam += explain_nodes(model, g, sample_motif_nodes(g, CAL_NODES, cs))
    Xcal = np.array([p["feat"] for p in fam])
    sc = np.array([p["score"] for p in fam], dtype=np.float64)
    order = np.argsort(sc)
    s_sorted = sc[order]
    std_pos, std_cum = crc_curve(std, np.ones(len(std)))
    pool_pos, pool_cum = crc_curve(fam, np.ones(len(fam)))

    run = {"family": family, "dataset": ds, "arch": arch, "seed": si, "train_acc": round(tr, 4),
           "test_acc": round(te, 4), "rows": []}
    for ti, (kind, m, nz) in enumerate(F_["tests"]):
        per = {(me, a): [] for me in METHODS for a in ALPHAS}
        for gi in range(N_TEST_GRAPHS):
            ts = base + 500 + 100 * ti + gi
            g = add_noise(make_graph(family, ds, ts, m), nz, seed=ts + 50)
            test = explain_nodes(model, g, sample_motif_nodes(g, N_TEST, ts))
            Xte = np.array([t["feat"] for t in test])
            logit = fit_lr(np.vstack([Xcal, Xte]), np.r_[np.zeros(len(Xcal)), np.ones(len(Xte))])
            ratio = len(Xcal) / len(Xte)
            w_cal = np.exp(logit(Xcal)) * ratio
            w_te = np.exp(logit(Xte)) * ratio
            w_sorted = w_cal[order]
            W = float(w_cal.sum())
            wpos, wcum = crc_curve(fam, w_cal)
            for a in ALPHAS:
                fixed = {"standard": level_quantile(std_scores, 1 - a),
                         "pooled": level_quantile(sc, 1 - a),
                         "standard_crc": crc_k(std_pos, std_cum, float(len(std)), 1.0, a),
                         "pooled_crc": crc_k(pool_pos, pool_cum, float(len(fam)), 1.0, a)}
                for me in METHODS:
                    vals = []
                    for j, t in enumerate(test):
                        if me == "weighted":
                            q = weighted_q(s_sorted, w_sorted, w_te[j], a)
                        elif me == "weighted_crc":
                            q = crc_k(wpos, wcum, W, float(w_te[j]), a)
                        else:
                            q = fixed[me]
                        vals.append(node_metrics(t, q))
                    v = np.array(vals, dtype=float)
                    per[(me, a)].append(v.mean(0))
        for a in ALPHAS:
            for me in METHODS:
                arr = np.array(per[(me, a)], dtype=float)
                mean = arr.mean(0)
                run["rows"].append({"type": kind, "ba_m": m, "noise": nz, "method": me, "alpha": a,
                                    "coverage": round(float(mean[0]), 4), "recall": round(float(mean[1]), 4),
                                    "avg_set_size": round(float(mean[2]), 2),
                                    "abstention_rate": round(float(mean[3]), 4),
                                    "whole_rate": round(float(mean[4]), 4),
                                    "frac_selected": round(float(mean[5]), 4)})
    return run


# ---------- main ----------
t_start = time.time()
all_runs, repro = [], {}
for family in ["ba", "tree"]:
    F_ = FAMILIES[family]
    ref_path = os.path.join(RES_DIR, F_["ref"])
    ref_runs = json.load(open(ref_path))["runs"] if os.path.exists(ref_path) else []
    n_same = n_cmp = 0
    n_total = len(F_["datasets"]) * len(ARCHS) * len(SEEDS)
    k = 0
    for di, ds in enumerate(F_["datasets"]):
        for arch in ARCHS:
            for si in SEEDS:
                t0 = time.time()
                run = run_one(family, di, ds, arch, si)
                all_runs.append(run)
                k += 1
                # reproducibility: full-coverage rows must equal Weeks 16-17
                ref = [r for r in ref_runs if r["dataset"] == ds and r["arch"] == arch and r["seed"] == si]
                if ref:
                    for r in run["rows"]:
                        if r["method"] in ("standard", "pooled", "weighted"):
                            match = [x for x in ref[0]["rows"] if x["type"] == r["type"] and x["ba_m"] == r["ba_m"]
                                     and x["noise"] == r["noise"] and x["method"] == r["method"]
                                     and x["alpha"] == r["alpha"]]
                            if match:
                                n_cmp += 1
                                n_same += int(match[0]["coverage"] == r["coverage"]
                                              and match[0]["avg_set_size"] == r["avg_set_size"])
                with open(OUT_JSON, "w") as f:
                    json.dump({"runs": all_runs}, f, indent=2)
                log(f"[{family} {k:2d}/{n_total}] {ds:5} {arch:9} seed {si} | test acc {run['test_acc']:.3f} "
                    f"| {time.time() - t0:.0f} s")
    repro[family] = {"rows_compared": n_cmp, "rows_identical": n_same}
    log(f"REPRODUCIBILITY ({family}, full-coverage rows vs {F_['ref']}): {n_same} of {n_cmp} identical")

# ---------- summary ----------
summary = []
for family in ["ba", "tree"]:
    tests = FAMILIES[family]["tests"]
    for a in ALPHAS:
        log(f"\n{family.upper()} | target {int(round((1 - a) * 100))}% | full coverage / recall / set size / whole-subgraph rate")
        log("method        | " + " | ".join(f"{kd} ({m},{nz})" for kd, m, nz in tests))
        for me in METHODS:
            txt = []
            for kd, m, nz in tests:
                rows = [r for run in all_runs if run["family"] == family for r in run["rows"]
                        if r["type"] == kd and r["ba_m"] == m and r["noise"] == nz
                        and r["method"] == me and r["alpha"] == a]
                c = np.array([r["coverage"] for r in rows])
                rc = np.array([r["recall"] for r in rows])
                s = np.array([r["avg_set_size"] for r in rows])
                wh = np.array([r["whole_rate"] for r in rows])
                summary.append({"family": family, "method": me, "type": kd, "ba_m": m, "noise": nz, "alpha": a,
                                "coverage_mean": round(float(c.mean()), 4), "recall_mean": round(float(rc.mean()), 4),
                                "recall_min": round(float(rc.min()), 4), "set_size_mean": round(float(s.mean()), 2),
                                "whole_rate_mean": round(float(wh.mean()), 4)})
                txt.append("%.3f/%.3f/%.1f/%.2f" % (c.mean(), rc.mean(), s.mean(), wh.mean()))
            log(f"{me:13} | " + " | ".join(txt))
with open(OUT_JSON, "w") as f:
    json.dump({"reproducibility": repro, "runs": all_runs, "summary": summary}, f, indent=2)
log(f"\nTotal time: {time.time() - t_start:.0f} s")
log("Saved:", OUT_JSON)

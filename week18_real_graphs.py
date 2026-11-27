# week18_real_graphs.py
# Week 18: real citation graphs (Cora, CiteSeer) with a model-based proxy ground truth.
#
#   Real graphs have no planted explanation, so the target is a proxy computed from the
#   model itself by exhaustive occlusion: an undirected edge is "critical" for node v if
#   deleting it lowers the predicted-class probability of v by at least TAU.
#   E*_v = all critical edges of v (nodes with no critical edge are not used).
#   The explainer (Saliency, one backward pass) is cheap; occlusion needs one forward pass
#   per edge. Conformal sets certify how many top-Saliency edges contain all critical edges.
#
#   Edges are handled as undirected units (both directions together).
#   Shifts (applied to the real graph): remove a fraction delta of edges, then add a
#   fraction rho of random edges.  Calibration family = delta {0,10,20%} x rho {0,2.5,5,10%}.
#   Tests: in-support (0,0), held-out level (0,6.25%), held-out removal (15%,2.5%),
#          beyond range (0,15%).
#   Models: GCN, GAT, GIN, GraphConv (2 layers, real node features), 3 seeds, CPU deterministic.
#   Methods: standard, pooled, weighted (ours), dro_kl, ot, fewshot20 (same as Week 16).
#
# Place in:  E:\conformal_gnn_project
# Run:       python week18_real_graphs.py
# Needs internet the first time (downloads Cora and CiteSeer into E:\data\Planetoid).
# Expected time: about 1 to 2 hours (CPU). Progress is printed after every run.

import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.nn import GATConv, GCNConv, GINConv, GraphConv
from torch_geometric.utils import k_hop_subgraph, remove_self_loops, to_undirected

DATASETS = ["Cora", "CiteSeer"]
ARCHS = ["GCN", "GAT", "GIN", "GraphConv"]
SEEDS = [0, 1, 2]
ALPHAS = [0.05, 0.10, 0.20]
TAUS = [0.05, 0.10]
CAL_DEL = [0.0, 0.10, 0.20]
CAL_ADD = [0.0, 0.025, 0.05, 0.10]
CAL_NODES = 60
TESTS = [("in-support", 0.0, 0.0), ("held-out level", 0.0, 0.0625),
         ("held-out removal", 0.15, 0.025), ("beyond range", 0.0, 0.15)]
N_TEST_GRAPHS = 3
N_TEST = 100
N_STD = 200
FEWSHOT_N = 20
EPOCHS = 300
HID = 64
N_HOPS = 2
MAX_UNITS = 400          # skip nodes whose 2-hop subgraph has more undirected edges
LR_LAMBDA = 1.0
TOPK_MASS = 12
OT_STEPS, OT_LR = 200, 0.1
METHODS = ["standard", "pooled", "weighted", "dro_kl", "ot", "fewshot20"]

QUICK = len(sys.argv) > 1 and sys.argv[1] == "quick"     # smoke test: 1 dataset, 1 model, 1 seed
if QUICK:
    DATASETS, ARCHS, SEEDS, N_TEST_GRAPHS, N_TEST, CAL_NODES = ["Cora"], ["GCN"], [0], 1, 30, 15

device = torch.device("cpu")
torch.use_deterministic_algorithms(True, warn_only=True)
torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
print("Running on:", device, "| quick test" if QUICK else "")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR = os.path.join(ROOT, "results")
DATA_DIR = os.path.join(ROOT, "data", "Planetoid")
os.makedirs(RES_DIR, exist_ok=True)
OUT_JSON = os.path.join(RES_DIR, "week18_quick.json" if QUICK else "week18_real_graphs.json")


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


# ---------- graphs ----------
def load(name):
    d = Planetoid(DATA_DIR, name)[0]
    d.edge_index = to_undirected(remove_self_loops(d.edge_index)[0], num_nodes=d.num_nodes)
    return d


def undirected_pairs(ei):
    a, b = ei[0], ei[1]
    keep = a < b
    return torch.stack([a[keep], b[keep]])


def perturb(d, delete, add, seed):
    """Remove a fraction `delete` of undirected edges, then add a fraction `add` (of the
    original count) of random new undirected edges. Returns a new graph."""
    d = d.clone()
    if delete <= 0 and add <= 0:
        return d
    g = torch.Generator().manual_seed(seed)
    pairs = undirected_pairs(d.edge_index)
    m0 = pairs.size(1)
    if delete > 0:
        keep = torch.randperm(m0, generator=g)[: m0 - int(delete * m0)]
        pairs = pairs[:, torch.sort(keep).values]
    if add > 0:
        existing = set(zip(pairs[0].tolist(), pairs[1].tolist()))
        new = []
        while len(new) < int(add * m0):
            a, b = torch.randint(0, d.num_nodes, (2,), generator=g).tolist()
            a, b = min(a, b), max(a, b)
            if a != b and (a, b) not in existing:
                existing.add((a, b))
                new.append((a, b))
        pairs = torch.cat([pairs, torch.tensor(new, dtype=torch.long).t()], dim=1)
    d.edge_index = torch.cat([pairs, pairs.flip(0)], dim=1)
    return d


# ---------- models ----------
class Net(torch.nn.Module):
    def __init__(self, arch, in_dim, hid, out_dim):
        super().__init__()
        if arch == "GCN":
            self.c1, self.c2 = GCNConv(in_dim, hid), GCNConv(hid, out_dim)
        elif arch == "GAT":
            self.c1, self.c2 = GATConv(in_dim, hid // 8, heads=8), GATConv(hid, out_dim, heads=1)
        elif arch == "GIN":
            mlp1 = torch.nn.Sequential(torch.nn.Linear(in_dim, hid), torch.nn.ReLU(), torch.nn.Linear(hid, hid))
            mlp2 = torch.nn.Sequential(torch.nn.Linear(hid, hid), torch.nn.ReLU(), torch.nn.Linear(hid, out_dim))
            self.c1, self.c2 = GINConv(mlp1), GINConv(mlp2)
        else:
            self.c1, self.c2 = GraphConv(in_dim, hid, aggr="add"), GraphConv(hid, out_dim, aggr="add")

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.c1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.c2(x, edge_index)


def train_model(arch, d, seed):
    set_seed(seed)
    C = int(d.y.max()) + 1
    model = Net(arch, d.x.size(1), HID, C).to(device)
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
        if ep % 5 == 0:
            va = acc("val")
            if va > best_val:
                best_val = va
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, idx, acc("train"), acc("val"), acc("test")


# ---------- explanation, proxy ground truth, features ----------
SKIPPED = {"large": 0, "seen": 0}


def explain_nodes(model, d, nodes):
    """For each node: Saliency ranks over undirected edges of its 2-hop subgraph and the
    occlusion drop of every edge. Ground truth per TAU is derived later."""
    deg_full = torch.bincount(d.edge_index[0], minlength=d.num_nodes).double().numpy()
    epn = d.edge_index.size(1) / d.num_nodes
    out = []
    for v in nodes:
        SKIPPED["seen"] += 1
        subset, sei, mapping, _ = k_hop_subgraph(int(v), N_HOPS, d.edge_index, relabel_nodes=True,
                                                 num_nodes=d.num_nodes)
        a, b = sei
        und = (a < b).nonzero().flatten()
        n_units = int(und.numel())
        if n_units == 0 or n_units > MAX_UNITS:
            SKIPPED["large"] += int(n_units > MAX_UNITS)
            continue
        # unit id for every directed edge
        key = {(int(a[i]), int(b[i])): u for u, i in enumerate(und.tolist())}
        unit = torch.tensor([key[(min(int(p), int(q)), max(int(p), int(q)))] for p, q in zip(a, b)])
        xs, c = d.x[subset], int(mapping)
        with torch.no_grad():
            prob = F.softmax(model(xs, sei), 1)[c]
        yhat = int(prob.argmax())
        p0 = float(prob[yhat])
        # Saliency over edges
        mask = torch.nn.Parameter(torch.ones(sei.size(1)))
        set_masks(model, mask, sei, apply_sigmoid=False)
        model(xs, sei)[c, yhat].backward()
        clear_masks(model)
        imp = torch.zeros(n_units, dtype=torch.float64).index_add_(0, unit, mask.grad.detach().abs().double()).numpy()
        # occlusion: delete each undirected edge, record drop of predicted-class probability
        drop = np.zeros(n_units)
        with torch.no_grad():
            for u in range(n_units):
                keep = unit != u
                drop[u] = p0 - float(F.softmax(model(xs, sei[:, keep]), 1)[c, yhat])
        order = np.argsort(-imp, kind="stable")
        rank = np.empty(n_units, dtype=np.int64)
        rank[order] = np.arange(1, n_units + 1)
        p = imp / imp.sum() if imp.sum() > 0 else np.full(n_units, 1.0 / n_units)
        ent = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(n_units)) if n_units > 1 else 0.0
        topm = float(np.sort(p)[::-1][:TOPK_MASS].sum())
        nbrs = d.edge_index[1][d.edge_index[0] == v].numpy()
        feat = [math.log(n_units), math.log(deg_full[v] + 1),
                math.log(deg_full[nbrs].mean() + 1) if len(nbrs) else 0.0, epn, ent, topm]
        out.append({"rank": rank, "drop": drop, "feat": feat,
                    "correct": bool(yhat == int(d.y[v]))})
    return out


def with_gt(items, tau):
    """Keep nodes with at least one critical edge; attach ground truth and score."""
    res = []
    for t in items:
        gt = t["drop"] >= tau
        if gt.any():
            res.append({"rank": t["rank"], "gt": gt, "score": int(t["rank"][gt].max()), "feat": t["feat"]})
    return res


def sample(pool, k, seed):
    rng = np.random.default_rng(seed)
    pool = np.array(pool)
    return [int(v) for v in rng.choice(pool, size=min(k, len(pool)), replace=False)]


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


def kl_binary(z, b):
    z = min(max(z, 1e-12), 1 - 1e-12)
    b = min(max(b, 1e-12), 1 - 1e-12)
    return z * math.log(z / b) + (1 - z) * math.log((1 - z) / (1 - b))


def g_kl(beta, rho):
    if beta <= 0:
        return 0.0
    lo, hi = 0.0, beta
    if kl_binary(0.0, beta) <= rho:
        return 0.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if kl_binary(mid, beta) <= rho:
            hi = mid
        else:
            lo = mid
    return hi


def g_inv_kl(tau, rho):
    lo, hi = tau, 1.0
    if g_kl(1.0 - 1e-12, rho) <= tau:
        return 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if g_kl(mid, rho) <= tau:
            lo = mid
        else:
            hi = mid
    return lo


def level_quantile(scores, level):
    n = len(scores)
    if n == 0:
        return float("inf")
    k = math.ceil((n + 1) * level)
    return float("inf") if k > n else float(np.sort(scores)[k - 1])


def ot_weights(cal_scores, s_low, s_high):
    S = torch.tensor(cal_scores, dtype=torch.float64)
    smax = int(max(cal_scores.max(), s_high.max())) + 1
    grid = torch.arange(1, smax + 1, dtype=torch.float64)
    ind = (S[:, None] <= grid[None, :]).double()
    F_dn = torch.tensor([(s_low <= g).mean() for g in range(1, smax + 1)], dtype=torch.float64)
    F_up = torch.tensor([(s_high <= g).mean() for g in range(1, smax + 1)], dtype=torch.float64)
    h = max(1.06 * float(S.std()) * len(S) ** (-0.2), 1.0)
    kern = torch.exp(-0.5 * ((grid[None, :] - S[:, None]) / h) ** 2) / (h * math.sqrt(2 * math.pi))
    theta = torch.zeros(len(S), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=OT_LR)
    for _ in range(OT_STEPS):
        w = torch.softmax(theta, 0)
        loss = 0.5 * ((w @ kern) * ((w @ ind - F_up).abs() + (w @ ind - F_dn).abs() + F_dn - F_up)).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return torch.softmax(theta, 0).detach().numpy()


def weighted_quantile_plain(scores_sorted, w_sorted, alpha):
    cum = np.cumsum(w_sorted) / w_sorted.sum()
    idx = int(np.searchsorted(cum, 1 - alpha - 1e-12, side="left"))
    return float("inf") if idx >= len(scores_sorted) else float(scores_sorted[idx])


# ---------- main loop ----------
t_start = time.time()
all_runs = []
n_total = len(DATASETS) * len(ARCHS) * len(SEEDS)
for di, ds in enumerate(DATASETS):
    base_graph = load(ds)
    print(f"{ds}: {base_graph.num_nodes} nodes, {base_graph.edge_index.size(1) // 2} undirected edges, "
          f"{base_graph.x.size(1)} features, {int(base_graph.y.max()) + 1} classes")
    for ai, arch in enumerate(ARCHS):
        for si in SEEDS:
            t0 = time.time()
            base = 100000 * (di + 1) + 1000 * si
            model, idx, tr, va, te = train_model(arch, base_graph, seed=base + 10 * ai)
            cal_pool, test_pool, val_pool = idx["cal"].tolist(), idx["test"].tolist(), idx["val"].tolist()

            std_raw = explain_nodes(model, base_graph, sample(cal_pool, N_STD, base + 1))
            fam_raw = []
            for i, dl in enumerate(CAL_DEL):
                for j, ad in enumerate(CAL_ADD):
                    cs = base + 100 + 10 * i + j
                    g = perturb(base_graph, dl, ad, seed=cs + 500)
                    fam_raw += explain_nodes(model, g, sample(cal_pool, CAL_NODES, cs))
            test_raw = []
            for ti, (kind, dl, ad) in enumerate(TESTS):
                fs = base + 900 + 100 * ti
                gf = perturb(base_graph, dl, ad, seed=fs + 50)
                few_raw = explain_nodes(model, gf, sample(val_pool, 3 * FEWSHOT_N, fs))
                graphs = []
                for gi in range(N_TEST_GRAPHS):
                    ts = base + 500 + 100 * ti + gi
                    g = perturb(base_graph, dl, ad, seed=ts + 50)
                    graphs.append(explain_nodes(model, g, sample(test_pool, N_TEST, ts)))
                test_raw.append((kind, dl, ad, few_raw, graphs))

            run = {"dataset": ds, "arch": arch, "seed": si, "train_acc": round(tr, 4),
                   "val_acc": round(va, 4), "test_acc": round(te, 4), "rows": [], "counts": {}}
            for tau in TAUS:
                std = np.array([t["score"] for t in with_gt(std_raw, tau)], float)
                family = with_gt(fam_raw, tau)
                if len(family) < 20 or len(std) < 20:
                    print(f"  too few nodes with critical edges at TAU {tau}; skipped")
                    continue
                Xcal = np.array([p["feat"] for p in family])
                sc = np.array([p["score"] for p in family], float)
                order = np.argsort(sc, kind="stable")
                s_sorted = sc[order]
                gt_min = min(int(p["gt"].sum()) for p in family)
                run["counts"][str(tau)] = {"standard_cal": len(std), "family_cal": len(family),
                                          "family_raw": len(fam_raw),
                                          "mean_gt_size": round(float(np.mean([p["gt"].sum() for p in family])), 2),
                                          "mean_units": round(float(np.mean([len(p["rank"]) for p in family])), 2)}
                for kind, dl, ad, few_raw, graphs in test_raw:
                    few = np.array([t["score"] for t in with_gt(few_raw, tau)][:FEWSHOT_N], float)
                    per = {(me, a): [] for me in METHODS for a in ALPHAS}
                    kl_list, n_used = [], []
                    for graw in graphs:
                        test = with_gt(graw, tau)
                        n_used.append(len(test))
                        if len(test) < 5:
                            continue
                        Xte = np.array([t["feat"] for t in test])
                        logit = fit_lr(np.vstack([Xcal, Xte]), np.r_[np.zeros(len(Xcal)), np.ones(len(Xte))])
                        ratio = len(Xcal) / len(Xte)
                        w_cal = np.exp(logit(Xcal)) * ratio
                        w_te = np.exp(logit(Xte)) * ratio
                        w_sorted = w_cal[order]
                        rho = max(0.0, float(np.mean(np.log(w_te / w_cal.mean()))))
                        kl_list.append(rho)
                        s_low = np.array([min(gt_min, len(t["rank"])) for t in test], float)
                        s_high = np.array([len(t["rank"]) for t in test], float)
                        w_ot = ot_weights(sc, s_low, s_high)[order]
                        for a in ALPHAS:
                            q_fixed = {"standard": level_quantile(std, 1 - a),
                                       "pooled": level_quantile(sc, 1 - a),
                                       "dro_kl": level_quantile(sc, g_inv_kl(1 - a, rho)),
                                       "ot": weighted_quantile_plain(s_sorted, w_ot, a),
                                       "fewshot20": level_quantile(few, 1 - a)}
                            for me in METHODS:
                                cov, size, abst = [], [], []
                                for j, t in enumerate(test):
                                    q = weighted_q(s_sorted, w_sorted, w_te[j], a) if me == "weighted" else q_fixed[me]
                                    cov.append(bool((t["rank"][t["gt"]] <= q).all()))
                                    size.append(int((t["rank"] <= q).sum()))
                                    abst.append(math.isinf(q))
                                per[(me, a)].append((np.mean(cov), np.mean(size), np.mean(abst)))
                    for a in ALPHAS:
                        for me in METHODS:
                            if not per[(me, a)]:
                                continue
                            arr = np.array(per[(me, a)], dtype=float)
                            run["rows"].append({"tau": tau, "type": kind, "delete": dl, "add": ad, "method": me,
                                                "alpha": a, "coverage": round(float(arr[:, 0].mean()), 4),
                                                "avg_set_size": round(float(arr[:, 1].mean()), 2),
                                                "abstention_rate": round(float(arr[:, 2].mean()), 4),
                                                "kl_estimate": round(float(np.mean(kl_list)), 4),
                                                "test_nodes_per_graph": round(float(np.mean(n_used)), 1)})
            all_runs.append(run)
            with open(OUT_JSON, "w") as f:
                json.dump({"runs": all_runs, "skipped": SKIPPED}, f, indent=2)
            c = run["counts"].get(str(TAUS[0]), {"family_cal": 0, "family_raw": len(fam_raw), "mean_gt_size": 0, "mean_units": 0})
            print(f"[{len(all_runs):2d}/{n_total}] {ds:8} {arch:9} seed {si} | acc train {tr:.3f} test {te:.3f} | "
                  f"cal nodes {c['family_cal']}/{c['family_raw']} | GT size {c['mean_gt_size']} of "
                  f"{c['mean_units']} edges | {time.time() - t0:.0f} s", flush=True)

# ---------- summary ----------
summary = []
for tau in TAUS:
    for ds in DATASETS:
        for a in ALPHAS:
            print(f"\n{ds} | TAU {tau} | TARGET {int(round((1 - a) * 100))}% | coverage mean (worst run) / set size / abstain")
            print("method     | " + " | ".join(f"{k[0]} ({k[1]},{k[2]})" for k in TESTS))
            for me in METHODS:
                txt = []
                for kind, dl, ad in TESTS:
                    rows = [r for run in all_runs if run["dataset"] == ds for r in run["rows"]
                            if r["tau"] == tau and r["type"] == kind and r["method"] == me and r["alpha"] == a]
                    if not rows:
                        txt.append("n/a")
                        continue
                    cv = np.array([r["coverage"] for r in rows])
                    s = np.array([r["avg_set_size"] for r in rows])
                    ab = np.array([r["abstention_rate"] for r in rows])
                    summary.append({"dataset": ds, "tau": tau, "method": me, "type": kind, "alpha": a,
                                    "coverage_mean": round(float(cv.mean()), 4), "coverage_min": round(float(cv.min()), 4),
                                    "set_size_mean": round(float(s.mean()), 2), "abstention_mean": round(float(ab.mean()), 4),
                                    "n_runs": len(rows)})
                    txt.append("%.3f (%.3f) / %.1f / %.2f" % (cv.mean(), cv.min(), s.mean(), ab.mean()))
                print(f"{me:10} | " + " | ".join(txt))
print("\nModel test accuracy:")
for ds in DATASETS:
    for arch in ARCHS:
        accs = [run["test_acc"] for run in all_runs if run["dataset"] == ds and run["arch"] == arch]
        print(f"  {ds:8} {arch:9} mean {np.mean(accs):.4f}  min {np.min(accs):.4f}  max {np.max(accs):.4f}")
print("Nodes skipped for > %d edges: %d of %d explained" % (MAX_UNITS, SKIPPED["large"], SKIPPED["seen"]))
with open(OUT_JSON, "w") as f:
    json.dump({"runs": all_runs, "summary": summary, "skipped": SKIPPED}, f, indent=2)
print(f"\nTotal time: {time.time() - t_start:.0f} s")
print("Saved:", OUT_JSON)

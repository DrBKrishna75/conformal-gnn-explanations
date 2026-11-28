# Week 25: Twitch natural shift with further seeds (seed given on the command line).
# Identical to week19_twitch_conformal.py except that, for the given seed, it
# (1) draws a new DE split and retrains GCN and GAT exactly as week18_twitch_inspect.py, and
# (2) offsets every random seed of the calibration family and test sampling by 100000*SEED.
# With SEED = 0 every seed equals Week 19 (split and models of Week 18).
# Run:  python week25_twitch_seeds.py 1     then     python week25_twitch_seeds.py 2
# Based on Week 19: conformal explanation sets under NATURAL cross-country shift (Twitch)
# Source graph DE (models from Week 18, seed 0). Explanation domain: the edges incident
# to the node (its friendships); the model still predicts on the full 2-hop subgraph.
# Target: occlusion proxy (edge whose single removal lowers predicted-class prob >= tau).
# Calibration family: DE with edge removal delta x random addition rho (12 cells).
# Test: DE test split (in-support) and all nodes of EN, ES, FR, PT, RU (natural shift).
# Methods: standard, pooled, weighted (Algorithm 1), few-shot (20 labelled target nodes).
import os, sys, json, time, random, csv, math, copy
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import k_hop_subgraph, to_undirected, remove_self_loops, degree
from torch_geometric.explain.algorithm.utils import set_masks, clear_masks
from sklearn.metrics import roc_auc_score

DATA_DIR = r"E:\data\twitch"
RES_DIR = r"E:\results"
MODEL_DIR = os.path.join(RES_DIR, "models")
LOG_FILE = rf"E:\conformal_gnn_project\week25_output_seed{sys.argv[1]}.txt"
OUT_JSON = os.path.join(RES_DIR, f"week25_twitch_seed{sys.argv[1]}.json")
COUNTRIES = ["DE", "EN", "ES", "FR", "PT", "RU"]
TARGETS = ["EN", "ES", "FR", "PT", "RU"]
MODELS = ["GCN", "GAT"]
SEED = int(sys.argv[1])
OFF = 100000 * SEED
EPOCHS = 300
TAU = 0.05
MAX_DEG = 200
DELTAS = [0.0, 0.25, 0.50, 0.70]     # fraction of DE edges removed
RHOS = [0.0, 0.05, 0.10]             # fraction of random edges added
CAL_PER_CELL = 60
N_TEST = 150
N_FEW = 20
MAX_TRIES = 900
ALPHAS = [0.05, 0.10, 0.20]
TOPK_CONC = 3

open(LOG_FILE, "w").close()


def log(*a):
    s = " ".join(str(t) for t in a)
    print(s, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(s + "\n")


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)
T0 = time.time()
log(f"Week 25 - Twitch natural shift, seed {SEED} (CPU)")

# ---------- data (same loader as Week 18) ----------
FOLDER = {"DE": "DE", "EN": "ENGB", "ES": "ES", "FR": "FR", "PT": "PTBR", "RU": "RU"}
raw, max_feat = {}, 0
for c in COUNTRIES:
    tag = FOLDER[c]
    d = os.path.join(DATA_DIR, tag)
    fp = os.path.join(d, f"musae_{tag}_features.json")
    if not os.path.exists(fp):
        fp = os.path.join(d, f"musae_{tag}.json")
    with open(fp) as f:
        feats = json.load(f)
    lab = {}
    with open(os.path.join(d, f"musae_{tag}_target.csv")) as f:
        for r in csv.DictReader(f):
            lab[int(r["new_id"])] = 1 if r["mature"].strip() == "True" else 0
    edges = np.loadtxt(os.path.join(d, f"musae_{tag}_edges.csv"), delimiter=",", skiprows=1, dtype=np.int64)
    max_feat = max(max_feat, max((max(v) for v in feats.values() if v), default=0))
    raw[c] = (feats, lab, edges)
F_DIM = max_feat + 1
data = {}
for c in COUNTRIES:
    feats, lab, edges = raw[c]
    n = len(lab)
    x = torch.zeros(n, F_DIM)
    for k, v in feats.items():
        if v and int(k) < n:
            x[int(k), torch.tensor(sorted(set(v)))] = 1.0
    y = torch.tensor([lab[i] for i in range(n)], dtype=torch.long)
    ei, _ = remove_self_loops(torch.from_numpy(edges.T.copy()))
    data[c] = Data(x=x, y=y, edge_index=to_undirected(ei, num_nodes=n), num_nodes=n)
# ---------- DE split exactly as Week 18, with this seed ----------
_n = data["DE"].num_nodes
_perm = torch.randperm(_n, generator=torch.Generator().manual_seed(SEED))
_ntr, _nva, _nca = int(0.60 * _n), int(0.10 * _n), int(0.15 * _n)
split = {"train": _perm[:_ntr], "val": _perm[_ntr:_ntr + _nva],
         "cal": _perm[_ntr + _nva:_ntr + _nva + _nca], "test": _perm[_ntr + _nva + _nca:]}
torch.save(split, os.path.join(RES_DIR, f"week25_DE_split_seed{SEED}.pt"))
log(f"Loaded 6 graphs, feature dim {F_DIM}; DE split cal {len(split['cal'])}, test {len(split['test'])}")


# ---------- models (same definition as Week 18) ----------
class Net(torch.nn.Module):
    def __init__(self, kind, in_dim, hid=64, out=2):
        super().__init__()
        if kind == "GCN":
            self.c1, self.c2 = GCNConv(in_dim, hid), GCNConv(hid, out)
        else:
            self.c1, self.c2 = GATConv(in_dim, hid // 8, heads=8), GATConv(hid, out, heads=1)

    def forward(self, x, ei):
        x = F.dropout(x, 0.5, self.training)
        x = F.relu(self.c1(x, ei))
        x = F.dropout(x, 0.5, self.training)
        return self.c2(x, ei)


# ---------- speed-up: apply the first layer's input projection once per graph ----------
# GCNConv and GATConv first multiply x by a weight matrix and then pass messages.
# We compute x @ W (+ bias, if any) once for all nodes and replace that projection with
# the identity, so each occlusion pass costs only message passing. Checked for equality.
def make_fast(model):
    fast = copy.deepcopy(model)
    names, mods = [], {}
    for name, m in list(fast.c1._modules.items()):
        ind = None if m is None else getattr(m, "in_channels", getattr(m, "in_features", None))
        if ind == F_DIM:
            names.append(name)
            mods[id(m)] = m
    assert len(mods) == 1, "unexpected first-layer structure"
    proj = list(mods.values())[0]
    for name in names:
        setattr(fast.c1, name, torch.nn.Identity())
    fast.eval()
    return fast, proj


XW_CACHE = {}


def projected(proj, x):
    key = id(x)
    if key not in XW_CACHE:
        with torch.no_grad():
            XW_CACHE[key] = proj(x)
    return XW_CACHE[key]


# ---------- graph perturbation ----------
def perturb(g, delta, rho, seed):
    rng = np.random.default_rng(seed)
    ei = g.edge_index
    und = ei[:, ei[0] < ei[1]].numpy()
    m, n = und.shape[1], g.num_nodes
    keep = np.ones(m, dtype=bool)
    keep[rng.permutation(m)[:int(round(delta * m))]] = False
    und = und[:, keep]
    existing = set((und[0] * n + und[1]).tolist())
    n_add, added = int(round(rho * m)), []
    while len(added) < n_add:
        a, b = rng.integers(0, n, size=2)
        if a == b:
            continue
        a, b = min(a, b), max(a, b)
        key = int(a) * n + int(b)
        if key in existing:
            continue
        existing.add(key)
        added.append((a, b))
    if added:
        und = np.hstack([und, np.array(added, dtype=np.int64).T])
    new_ei = to_undirected(torch.from_numpy(und.copy()), num_nodes=n)
    return Data(x=g.x, y=g.y, edge_index=new_ei, num_nodes=n)


# ---------- one node: saliency ranking, occlusion target, score, features ----------
def node_record(model, g, xw, v, density, deg_all):
    subset, sei, mapping, _ = k_hop_subgraph(int(v), 2, g.edge_index, relabel_nodes=True,
                                             num_nodes=g.num_nodes)
    xs, c = xw[subset], int(mapping[0])
    idx_in = (sei[1] == c).nonzero().view(-1)
    idx_out = (sei[0] == c).nonzero().view(-1)
    idx_in = idx_in[torch.argsort(sei[0, idx_in], stable=True)]
    idx_out = idx_out[torch.argsort(sei[1, idx_out], stable=True)]
    nv = idx_in.numel()
    if nv == 0 or nv > MAX_DEG or not torch.equal(sei[0, idx_in], sei[1, idx_out]):
        return None
    # saliency: |d logit_yhat / d edge-mask|, summed over both directions of an incident edge
    mask = torch.ones(sei.size(1), requires_grad=True)
    set_masks(model, mask, sei, apply_sigmoid=False)
    out = model(xs, sei)
    yhat = int(out[c].argmax())
    out[c, yhat].backward()
    clear_masks(model)
    gabs = mask.grad.abs()
    sal = (gabs[idx_in] + gabs[idx_out]).numpy()
    # occlusion target
    with torch.no_grad():
        p0 = float(F.softmax(model(xs, sei), -1)[c, yhat])
        keep = torch.ones(sei.size(1), dtype=torch.bool)
        crit = np.zeros(nv, dtype=bool)
        for j in range(nv):
            keep[idx_in[j]] = False
            keep[idx_out[j]] = False
            pj = float(F.softmax(model(xs, sei[:, keep]), -1)[c, yhat])
            keep[idx_in[j]] = True
            keep[idx_out[j]] = True
            crit[j] = (p0 - pj) >= TAU
    if not crit.any():
        return None
    order = np.argsort(-sal, kind="stable")
    rank = np.empty(nv, dtype=np.int64)
    rank[order] = np.arange(1, nv + 1)
    S = int(rank[crit].max())
    tot = sal.sum()
    pe = sal / tot if tot > 0 else np.full(nv, 1.0 / nv)
    ent = 0.0 if nv == 1 else float(-(pe[pe > 0] * np.log(pe[pe > 0])).sum() / math.log(nv))
    conc = float(np.sort(pe)[::-1][:TOPK_CONC].sum())
    nbrs = sei[0, idx_in]
    feat = [math.log(nv),
            math.log1p(float(deg_all[subset[nbrs]].mean())),
            math.log1p(sei.size(1) / 2),
            density, ent, conc]
    auc = float("nan")
    if 0 < crit.sum() < nv:
        auc = float(roc_auc_score(crit.astype(int), sal))
    return {"node": int(v), "S": S, "nv": nv, "ncrit": int(crit.sum()), "feat": feat, "auc": auc}


def collect(model, proj, g, pool, need, seed, label):
    rng = np.random.default_rng(seed)
    deg_all = degree(g.edge_index[0], g.num_nodes)
    density = g.edge_index.size(1) / g.num_nodes
    xw = projected(proj, g.x)
    pool = np.array([u for u in pool if 1 <= int(deg_all[u]) <= MAX_DEG])
    pool = pool[rng.permutation(len(pool))]
    recs, tried, t1 = [], 0, time.time()
    for v in pool[:MAX_TRIES]:
        tried += 1
        r = node_record(model, g, xw, v, density, deg_all)
        if r is not None:
            recs.append(r)
        if len(recs) >= need:
            break
    log(f"    {label:22} eligible {len(recs):4}/{tried:4} tried, density {density:6.2f}, "
        f"{time.time() - t1:7.1f} s")
    return recs


# ---------- conformal methods ----------
def split_k(S, alpha):
    S = np.sort(np.asarray(S))
    r = math.ceil((len(S) + 1) * (1 - alpha))
    return float(S[r - 1]) if r <= len(S) else math.inf


def fit_logodds(Xa, ya, lam=1.0, iters=50):
    mu, sd = Xa.mean(0), Xa.std(0)
    sd[sd == 0] = 1.0
    Z = np.hstack([np.ones((len(Xa), 1)), (Xa - mu) / sd])
    w = np.zeros(Z.shape[1])
    R = lam * np.eye(Z.shape[1])
    R[0, 0] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -30, 30)))
        grad = Z.T @ (p - ya) + R @ w
        H = (Z * (p * (1 - p))[:, None]).T @ Z + R + 1e-9 * np.eye(Z.shape[1])
        step = np.linalg.solve(H, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return lambda X: np.clip(np.hstack([np.ones((len(X), 1)), (X - mu) / sd]) @ w, -30, 30)


def weighted_k(cal, test, alpha):
    Sc = np.array([r["S"] for r in cal], dtype=float)
    Xc = np.array([r["feat"] for r in cal])
    Xt = np.array([r["feat"] for r in test])
    n, q = len(Xc), len(Xt)
    eta = fit_logodds(np.vstack([Xc, Xt]), np.r_[np.zeros(n), np.ones(q)])
    wc, wt = (n / q) * np.exp(eta(Xc)), (n / q) * np.exp(eta(Xt))
    order = np.argsort(Sc, kind="stable")
    Ss, cum = Sc[order], np.cumsum(wc[order])
    W, ks = wc.sum(), []
    for w in wt:
        tot = W + w
        if w / tot > alpha:
            ks.append(math.inf)
        else:
            i = int(np.searchsorted(cum, (1 - alpha) * tot * (1 - 1e-12), side="left"))
            ks.append(float(Ss[min(i, len(Ss) - 1)]))
    return ks


def evaluate(test, ks):
    cov, size, abst, whole, frac = [], [], [], [], []
    for r, k in zip(test, ks):
        a = math.isinf(k)
        sz = r["nv"] if a else min(k, r["nv"])
        cov.append(a or r["S"] <= k)
        size.append(sz)
        abst.append(a)
        whole.append(sz >= r["nv"])
        frac.append(sz / r["nv"])
    return {"coverage": round(float(np.mean(cov)), 4), "set_size": round(float(np.mean(size)), 2),
            "abstain": round(float(np.mean(abst)), 3), "whole": round(float(np.mean(whole)), 3),
            "frac_selected": round(float(np.mean(frac)), 3), "n": len(test)}


# ---------- main ----------
# ---------- train GCN and GAT on DE exactly as Week 18, with this seed ----------
from sklearn.metrics import roc_auc_score as _auc
STATES, TRAIN_INFO = {}, {}
for kind in MODELS:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    t1 = time.time()
    src = data["DE"]
    model = Net(kind, F_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_val, best_state = -1.0, None
    for ep in range(EPOCHS):
        model.train()
        opt.zero_grad()
        out = model(src.x, src.edge_index)
        loss = F.cross_entropy(out[split["train"]], src.y[split["train"]])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            out = model(src.x, src.edge_index)
            va = float((out[split["val"]].argmax(-1) == src.y[split["val"]]).float().mean())
        if va > best_val:
            best_val, best_state = va, copy.deepcopy(model.state_dict())
    torch.save(best_state, os.path.join(MODEL_DIR, f"week25_{kind}_DE_seed{SEED}.pt"))
    STATES[kind] = best_state
    model.load_state_dict(best_state)
    model.eval()
    info = {"best_val_acc": round(best_val, 4)}
    with torch.no_grad():
        for c in COUNTRIES:
            g = data[c]
            o = model(g.x, g.edge_index)
            idx = split["test"] if c == "DE" else torch.arange(g.num_nodes)
            p = F.softmax(o[idx], -1)[:, 1].numpy()
            info[c] = {"acc": round(float((o[idx].argmax(-1) == g.y[idx]).float().mean()), 4),
                       "auc": round(float(_auc(g.y[idx].numpy(), p)), 4)}
    TRAIN_INFO[kind] = info
    log(f"  Trained {kind} (seed {SEED}) in {time.time() - t1:.1f} s: val {best_val:.4f}, "
        + ", ".join(f"{c} AUC {info[c]['auc']}" for c in COUNTRIES))

results = {"seed": SEED, "train": TRAIN_INFO}
for kind in MODELS:
    log(f"\n===== Model {kind} (trained on DE, seed {SEED}) =====")
    model = Net(kind, F_DIM)
    model.load_state_dict(STATES[kind])
    model.eval()
    fast, proj = make_fast(model)
    XW_CACHE.clear()
    with torch.no_grad():
        ref = model(data["DE"].x, data["DE"].edge_index)
        chk = fast(projected(proj, data["DE"].x), data["DE"].edge_index)
    diff = float((ref - chk).abs().max())
    log(f"  Speed-up check (max |difference| of outputs on DE): {diff:.2e}")
    assert diff < 1e-4, "fast first layer does not match the trained model"
    log("  Calibration family (DE, delta x rho):")
    family, cell_id = {}, 0
    for dl in DELTAS:
        for rh in RHOS:
            g = data["DE"] if (dl == 0 and rh == 0) else perturb(data["DE"], dl, rh, OFF + 1000 + cell_id)
            family[(dl, rh)] = collect(fast, proj, g, split["cal"].numpy(), CAL_PER_CELL,
                                       OFF + 2000 + cell_id, f"cell d={dl:.2f} r={rh:.2f}")
            cell_id += 1
    cal_all = [r for recs in family.values() for r in recs]
    cal_std = family[(0.0, 0.0)]
    log("  Test settings:")
    tests = {}
    for i, c in enumerate(COUNTRIES):
        pool = split["test"].numpy() if c == "DE" else np.arange(data[c].num_nodes)
        recs = collect(fast, proj, data[c], pool, N_TEST + N_FEW, OFF + 3000 + i, f"test {c}")
        tests[c] = (recs[:N_TEST], recs[N_TEST:])
    res = {"family_sizes": {f"{k[0]}_{k[1]}": len(v) for k, v in family.items()},
           "cal_density": {f"{k[0]}_{k[1]}": (v[0]["feat"][3] if v else None) for k, v in family.items()}}
    for c, (te, few) in tests.items():
        if len(te) == 0:
            continue
        res[c] = {"n_test": len(te), "n_few": len(few), "density": te[0]["feat"][3],
                  "mean_nv": round(float(np.mean([r["nv"] for r in te])), 2),
                  "mean_crit": round(float(np.mean([r["ncrit"] for r in te])), 2),
                  "mean_auc": round(float(np.nanmean([r["auc"] for r in te])), 4)}
        for a in ALPHAS:
            row = {"standard": evaluate(te, [split_k([r["S"] for r in cal_std], a)] * len(te)),
                   "pooled": evaluate(te, [split_k([r["S"] for r in cal_all], a)] * len(te)),
                   "weighted": evaluate(te, weighted_k(cal_all, te, a))}
            if len(few) >= N_FEW:
                row["fewshot20"] = evaluate(te, [split_k([r["S"] for r in few], a)] * len(te))
            res[c][f"target_{1 - a:.2f}"] = row
    results[kind] = res
    torch.save({"family": family, "tests": tests}, os.path.join(RES_DIR, f"week25_records_{kind}_seed{SEED}.pt"))
    json.dump(results, open(OUT_JSON, "w"), indent=2, default=str)

    for a in ALPHAS:
        tk = f"target_{1 - a:.2f}"
        log(f"\n  {kind}: coverage / mean set size (abstain rate) at target {1 - a:.2f}")
        log(f"  {'Setting':8}{'Dens':>7}{'n':>5}{'Standard':>20}{'Pooled':>20}{'Weighted':>26}{'Few-shot20':>20}")
        for c in COUNTRIES:
            if c not in res or tk not in res[c]:
                continue
            r = res[c][tk]
            def fmt(m, ab=False):
                if m not in r:
                    return "-"
                e = r[m]
                s = f"{e['coverage']:.3f} / {e['set_size']:.1f}"
                return s + (f" ({e['abstain']:.2f})" if ab else "")
            log(f"  {c:8}{res[c]['density']:7.2f}{res[c]['n_test']:5}{fmt('standard'):>20}{fmt('pooled'):>20}"
                f"{fmt('weighted', True):>26}{fmt('fewshot20'):>20}")
    log("\n  Mean incident edges / critical edges / saliency AUC per setting:")
    for c in COUNTRIES:
        if c in res:
            log(f"  {c:8} {res[c]['mean_nv']:7} {res[c]['mean_crit']:6} {res[c]['mean_auc']:8}")

total = round(time.time() - T0, 1)
results["total_seconds"] = total
json.dump(results, open(OUT_JSON, "w"), indent=2, default=str)
log(f"\nSaved {OUT_JSON}")
log(f"Total time: {total} s")

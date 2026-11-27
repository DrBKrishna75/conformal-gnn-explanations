# Week 18: Twitch cross-country NATURAL shift - feasibility check
# (1) load 6 Twitch country graphs, (2) train 4 models on DE (source),
# (3) accuracy/AUC on every country, (4) 2-hop subgraph sizes,
# Data: SNAP Twitch (https://snap.stanford.edu/data/twitch.zip), multi-hot features.
# (5) cost and size of the occlusion proxy target (tau = 0.05) with GCN.
import os, json, time, random, copy, csv
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, GINConv, GraphConv
from torch_geometric.utils import k_hop_subgraph, to_undirected, remove_self_loops, degree
from sklearn.metrics import roc_auc_score

DATA_DIR = r"E:\data\twitch"   # SNAP twitch.zip extracted here
RES_DIR = r"E:\results"
MODEL_DIR = os.path.join(RES_DIR, "models")
LOG_FILE = r"E:\conformal_gnn_project\week18_output.txt"
OUT_JSON = os.path.join(RES_DIR, "week18_twitch_inspect.json")
COUNTRIES = ["DE", "EN", "ES", "FR", "PT", "RU"]
SOURCE = "DE"
MODELS = ["GCN", "GAT", "GIN", "GraphConv"]
SEED = 0
EPOCHS = 300
TAU = 0.05
N_SIZE_SAMPLE = 300
N_OCC_NODES = 20
MAX_OCC_EDGES = 1000
CHUNK = 16

os.makedirs(MODEL_DIR, exist_ok=True)
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


torch.use_deterministic_algorithms(True)
T0 = time.time()
log("Week 18 - Twitch cross-country natural shift: feasibility check (CPU)")

# ---------- 1. load data ----------
FOLDER = {"DE": "DE", "EN": "ENGB", "ES": "ES", "FR": "FR", "PT": "PTBR", "RU": "RU"}
raw = {}
max_feat = 0
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
stats = {}
for c in COUNTRIES:
    feats, lab, edges = raw[c]
    n = len(lab)
    x = torch.zeros(n, F_DIM)
    for k, v in feats.items():
        if v and int(k) < n:
            x[int(k), torch.tensor(sorted(set(v)))] = 1.0
    y = torch.tensor([lab[i] for i in range(n)], dtype=torch.long)
    assert int(edges.max()) < n, f"{c}: edge index out of range"
    g = Data(x=x, y=y, edge_index=torch.from_numpy(edges.T.copy()), num_nodes=n)
    ei, _ = remove_self_loops(g.edge_index)
    ei = to_undirected(ei, num_nodes=g.num_nodes)
    g.edge_index = ei
    data[c] = g
    deg = degree(ei[0], g.num_nodes)
    stats[c] = {
        "nodes": int(g.num_nodes),
        "undirected_edges": int(ei.size(1) // 2),
        "features": int(g.x.size(1)),
        "positive_rate": round(float(g.y.float().mean()), 4),
        "mean_degree": round(float(deg.mean()), 2),
        "median_degree": float(deg.median()),
        "max_degree": int(deg.max()),
    }
log(f"\n[1] Dataset statistics (multi-hot feature dimension {F_DIM})")
log(f"{'Country':8}{'Nodes':>8}{'Edges':>10}{'Feat':>6}{'Pos':>8}{'MeanDeg':>9}{'MedDeg':>8}{'MaxDeg':>8}")
for c in COUNTRIES:
    s = stats[c]
    log(f"{c:8}{s['nodes']:>8}{s['undirected_edges']:>10}{s['features']:>6}{s['positive_rate']:>8}"
        f"{s['mean_degree']:>9}{s['median_degree']:>8.0f}{s['max_degree']:>8}")

# ---------- 2. split source graph ----------
src = data[SOURCE]
n_src = src.num_nodes
gen = torch.Generator().manual_seed(SEED)
perm = torch.randperm(n_src, generator=gen)
n_tr, n_va, n_ca = int(0.60 * n_src), int(0.10 * n_src), int(0.15 * n_src)
split = {
    "train": perm[:n_tr],
    "val": perm[n_tr:n_tr + n_va],
    "cal": perm[n_tr + n_va:n_tr + n_va + n_ca],
    "test": perm[n_tr + n_va + n_ca:],
}
torch.save(split, os.path.join(RES_DIR, "week18_DE_split_seed0.pt"))
log(f"\n[2] DE split: train {len(split['train'])}, val {len(split['val'])}, "
    f"cal {len(split['cal'])}, test {len(split['test'])}")


# ---------- 3. models ----------
class Net(torch.nn.Module):
    def __init__(self, kind, in_dim, hid=64, out=2):
        super().__init__()
        if kind == "GCN":
            self.c1, self.c2 = GCNConv(in_dim, hid), GCNConv(hid, out)
        elif kind == "GAT":
            self.c1, self.c2 = GATConv(in_dim, hid // 8, heads=8), GATConv(hid, out, heads=1)
        elif kind == "GIN":
            self.c1 = GINConv(torch.nn.Sequential(torch.nn.Linear(in_dim, hid), torch.nn.ReLU(),
                                                  torch.nn.Linear(hid, hid)))
            self.c2 = GINConv(torch.nn.Sequential(torch.nn.Linear(hid, hid), torch.nn.ReLU(),
                                                  torch.nn.Linear(hid, out)))
        else:
            self.c1, self.c2 = GraphConv(in_dim, hid), GraphConv(hid, out)

    def forward(self, x, ei):
        x = F.dropout(x, 0.5, self.training)
        x = F.relu(self.c1(x, ei))
        x = F.dropout(x, 0.5, self.training)
        return self.c2(x, ei)


def acc_auc(logits, y):
    prob = F.softmax(logits, -1)[:, 1]
    acc = float((logits.argmax(-1) == y).float().mean())
    try:
        auc = float(roc_auc_score(y.numpy(), prob.numpy()))
    except ValueError:
        auc = float("nan")
    return round(acc, 4), round(auc, 4)


results = {}
trained = {}
log("\n[3] Train on DE, evaluate on DE test and on all nodes of other countries")
for kind in MODELS:
    set_seed(SEED)
    t1 = time.time()
    model = Net(kind, src.x.size(1))
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
    model.load_state_dict(best_state)
    model.eval()
    torch.save(best_state, os.path.join(MODEL_DIR, f"week18_{kind}_DE_seed0.pt"))
    row = {"best_val_acc": round(best_val, 4), "train_seconds": round(time.time() - t1, 1)}
    with torch.no_grad():
        out = model(src.x, src.edge_index)
        row["DE_test"] = acc_auc(out[split["test"]], src.y[split["test"]])
        for c in COUNTRIES:
            if c == SOURCE:
                continue
            out = model(data[c].x, data[c].edge_index)
            row[c] = acc_auc(out, data[c].y)
    results[kind] = row
    trained[kind] = model
    log(f"  {kind:10} val {row['best_val_acc']:.4f} | DE test acc/AUC {row['DE_test']} | "
        + " | ".join(f"{c} {row[c]}" for c in COUNTRIES if c != SOURCE)
        + f" | {row['train_seconds']} s")

# ---------- 4. 2-hop subgraph sizes ----------
log(f"\n[4] 2-hop subgraph size (undirected edges), {N_SIZE_SAMPLE} random nodes per country")
log(f"{'Country':8}{'Median':>9}{'P90':>9}{'Max':>9}{'<=400':>8}{'<=1000':>8}")
rng = np.random.default_rng(SEED)
size_stats, sample_nodes = {}, {}
for c in COUNTRIES:
    g = data[c]
    pool = split["test"].numpy() if c == SOURCE else np.arange(g.num_nodes)
    nodes = rng.choice(pool, size=min(N_SIZE_SAMPLE, len(pool)), replace=False)
    sizes = []
    for v in nodes:
        _, sei, _, _ = k_hop_subgraph(int(v), 2, g.edge_index, relabel_nodes=True, num_nodes=g.num_nodes)
        sizes.append(sei.size(1) // 2)
    sizes = np.array(sizes)
    sample_nodes[c] = (nodes, sizes)
    size_stats[c] = {
        "median": float(np.median(sizes)), "p90": float(np.percentile(sizes, 90)),
        "max": int(sizes.max()),
        "frac_le_400": round(float((sizes <= 400).mean()), 3),
        "frac_le_1000": round(float((sizes <= 1000).mean()), 3),
    }
    s = size_stats[c]
    log(f"{c:8}{s['median']:>9.0f}{s['p90']:>9.0f}{s['max']:>9}{s['frac_le_400']:>8}{s['frac_le_1000']:>8}")


# ---------- 5. occlusion proxy target with GCN ----------
def occlusion(model, g, v):
    subset, sei, mapping, _ = k_hop_subgraph(int(v), 2, g.edge_index, relabel_nodes=True,
                                             num_nodes=g.num_nodes)
    xs, ns, c = g.x[subset], subset.numel(), int(mapping[0])
    with torch.no_grad():
        p = F.softmax(model(xs, sei), -1)[c]
        yhat = int(p.argmax())
        p0 = float(p[yhat])
        pairs = sei[:, sei[0] < sei[1]]
        key = sei[0] * ns + sei[1]
        kf = pairs[0] * ns + pairs[1]
        kb = pairs[1] * ns + pairs[0]
        drops = []
        for st in range(0, pairs.size(1), CHUNK):
            js = list(range(st, min(pairs.size(1), st + CHUNK)))
            eis = [sei[:, (key != kf[j]) & (key != kb[j])] + b * ns for b, j in enumerate(js)]
            out = F.softmax(model(xs.repeat(len(js), 1), torch.cat(eis, 1)), -1)
            drops.append(p0 - out[torch.arange(len(js)) * ns + c, yhat])
        drops = torch.cat(drops)
    return pairs.size(1), int((drops >= TAU).sum())


log(f"\n[5] Occlusion proxy target (GCN, tau = {TAU}), up to {N_OCC_NODES} nodes with "
    f"<= {MAX_OCC_EDGES} subgraph edges per country")
log(f"{'Country':8}{'Nodes':>7}{'MeanEdges':>11}{'MeanCrit':>10}{'Has>=1':>8}{'Sec/node':>10}{'Est300(min)':>13}")
occ_stats = {}
gcn = trained["GCN"]
for c in COUNTRIES:
    nodes, sizes = sample_nodes[c]
    chosen = nodes[sizes <= MAX_OCC_EDGES][:N_OCC_NODES]
    if len(chosen) == 0:
        occ_stats[c] = {"nodes": 0}
        log(f"{c:8}{0:>7}   (no sampled node has <= {MAX_OCC_EDGES} edges)")
        continue
    t1 = time.time()
    edges, crit = [], []
    for v in chosen:
        m, k = occlusion(gcn, data[c], v)
        edges.append(m)
        crit.append(k)
    sec = (time.time() - t1) / len(chosen)
    occ_stats[c] = {
        "nodes": int(len(chosen)), "mean_edges": round(float(np.mean(edges)), 1),
        "mean_critical": round(float(np.mean(crit)), 2),
        "frac_with_critical": round(float(np.mean(np.array(crit) > 0)), 3),
        "sec_per_node": round(sec, 3),
    }
    s = occ_stats[c]
    log(f"{c:8}{s['nodes']:>7}{s['mean_edges']:>11}{s['mean_critical']:>10}{s['frac_with_critical']:>8}"
        f"{s['sec_per_node']:>10}{sec * 300 / 60:>13.1f}")

total = round(time.time() - T0, 1)
json.dump({"dataset_stats": stats, "model_results": results, "subgraph_sizes": size_stats,
           "occlusion_gcn": occ_stats, "settings": {"seed": SEED, "epochs": EPOCHS, "tau": TAU,
           "source": SOURCE, "max_occ_edges": MAX_OCC_EDGES}, "total_seconds": total},
          open(OUT_JSON, "w"), indent=2)
log(f"\nSaved {OUT_JSON}")
log(f"Total time: {total} s")

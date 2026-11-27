# week1_train_bashapes.py
# Week 1 pilot: build BA-Shapes, train a 3-layer GCN, save the model + results.
# Place in:  ...\conformal_gnn\03_code\src
# Run:       python week1_train_bashapes.py

import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.nn import GCNConv

# ---------- 1. Fix random seeds (same results every run) ----------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

# Folder layout: this script lives in 03_code\src
# Outputs go to 03_code\data, 03_code\results, 03_code\results\models
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 03_code
DATA_DIR = os.path.join(CODE_DIR, "data")
RES_DIR = os.path.join(CODE_DIR, "results")
MODEL_DIR = os.path.join(RES_DIR, "models")
for d in (DATA_DIR, RES_DIR, MODEL_DIR):
    os.makedirs(d, exist_ok=True)

# ---------- 2. Build BA-Shapes (standard GNNExplainer setting) ----------
# 300 base nodes (Barabasi-Albert) + 80 "house" motifs (5 nodes each) = 700 nodes
dataset = ExplainerDataset(
    graph_generator=BAGraph(num_nodes=300, num_edges=5),
    motif_generator="house",
    num_motifs=80,
)
data = dataset[0]

# Node features: constant 1 for every node (standard for BA-Shapes)
data.x = torch.ones((data.num_nodes, 10))
num_classes = int(data.y.max()) + 1

# Train / validation / calibration / test split (60/10/15/15)
# The "calibration" part is kept aside for conformal prediction (Week 3+).
perm = torch.randperm(data.num_nodes)
n = data.num_nodes
n_tr, n_va, n_ca = int(0.60 * n), int(0.10 * n), int(0.15 * n)
idx_train = perm[:n_tr]
idx_val = perm[n_tr:n_tr + n_va]
idx_cal = perm[n_tr + n_va:n_tr + n_va + n_ca]
idx_test = perm[n_tr + n_va + n_ca:]

print(f"Nodes: {n}, Edges: {data.num_edges}, Classes: {num_classes}")
print(f"Split -> train {len(idx_train)}, val {len(idx_val)}, cal {len(idx_cal)}, test {len(idx_test)}")

data = data.to(device)


# ---------- 3. Define a simple 3-layer GCN ----------
class GCN(torch.nn.Module):
    def __init__(self, in_dim, hid, out_dim):
        super().__init__()
        self.c1 = GCNConv(in_dim, hid)
        self.c2 = GCNConv(hid, hid)
        self.c3 = GCNConv(hid, hid)
        self.lin = torch.nn.Linear(hid, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.c1(x, edge_index))
        x = F.relu(self.c2(x, edge_index))
        x = F.relu(self.c3(x, edge_index))
        return self.lin(x)


model = GCN(data.x.size(1), 20, num_classes).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


def accuracy(idx):
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)
    return (pred[idx] == data.y[idx]).float().mean().item()


# ---------- 4. Train ----------
best_val, best_state = 0.0, None
for epoch in range(1, 1001):
    model.train()
    opt.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[idx_train], data.y[idx_train])
    loss.backward()
    opt.step()

    if epoch % 50 == 0:
        tr, va = accuracy(idx_train), accuracy(idx_val)
        print(f"Epoch {epoch:4d} | loss {loss.item():.4f} | train {tr:.3f} | val {va:.3f}")
        if va >= best_val:
            best_val = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)

# ---------- 5. Final results ----------
res = {
    "dataset": "BA-Shapes",
    "seed": SEED,
    "device": str(device),
    "train_acc": round(accuracy(idx_train), 4),
    "val_acc": round(accuracy(idx_val), 4),
    "test_acc": round(accuracy(idx_test), 4),
}
print("\nFINAL:", res)

torch.save(model.state_dict(), os.path.join(MODEL_DIR, "gcn_bashapes.pt"))
torch.save(
    {"train": idx_train, "val": idx_val, "cal": idx_cal, "test": idx_test},
    os.path.join(DATA_DIR, "bashapes_splits.pt"),
)
torch.save(data.cpu(), os.path.join(DATA_DIR, "bashapes_data.pt"))
with open(os.path.join(RES_DIR, "week1_bashapes.json"), "w") as f:
    json.dump(res, f, indent=2)

print("Saved:")
print("  ", os.path.join(MODEL_DIR, "gcn_bashapes.pt"))
print("  ", os.path.join(DATA_DIR, "bashapes_splits.pt"))
print("  ", os.path.join(DATA_DIR, "bashapes_data.pt"))
print("  ", os.path.join(RES_DIR, "week1_bashapes.json"))

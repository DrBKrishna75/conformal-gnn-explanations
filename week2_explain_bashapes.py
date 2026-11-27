# week2_explain_bashapes.py
# Week 2: load the trained GCN from Week 1, run GNNExplainer on every
# calibration and test node, and save the edge-importance scores.
# These scores are the input for the conformal step (Week 3).
#
# Place in:  E:\conformal_gnn_project
# Quick test (10 nodes):   python week2_explain_bashapes.py --quick
# Full run (210 nodes):    python week2_explain_bashapes.py

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.nn import GCNConv
from torch_geometric.utils import k_hop_subgraph

# ---------- 0. Options ----------
parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true",
                    help="explain only 10 nodes, to test that everything works")
parser.add_argument("--epochs", type=int, default=200,
                    help="GNNExplainer optimisation epochs per node")
args = parser.parse_args()

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

# Same folder logic as Week 1, so the same files are found.
# (Script in E:\conformal_gnn_project -> files in E:\data and E:\results)
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(CODE_DIR, "data")
RES_DIR = os.path.join(CODE_DIR, "results")
MODEL_DIR = os.path.join(RES_DIR, "models")

# ---------- 1. Load data, splits and model from Week 1 ----------
data = torch.load(os.path.join(DATA_DIR, "bashapes_data.pt"),
                  map_location="cpu", weights_only=False)
splits = torch.load(os.path.join(DATA_DIR, "bashapes_splits.pt"),
                    map_location="cpu", weights_only=False)
data = data.to(device)
num_classes = int(data.y.max()) + 1


class GCN(torch.nn.Module):  # must match Week 1 exactly
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
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "gcn_bashapes.pt"),
                                 map_location=device, weights_only=True))
model.eval()

with torch.no_grad():
    logits = model(data.x, data.edge_index)
    probs = logits.softmax(dim=1)
    preds = logits.argmax(dim=1)

# Sanity check: test accuracy must match Week 1
test_acc = (preds[splits["test"].to(device)] == data.y[splits["test"].to(device)]).float().mean().item()
print(f"Reloaded model test accuracy: {test_acc:.4f} (should match Week 1)")

has_gt = hasattr(data, "edge_mask") and data.edge_mask is not None
if has_gt:
    gt_edge = data.edge_mask.bool()
    print(f"Ground-truth motif edges available: {int(gt_edge.sum())} of {data.num_edges}")
else:
    print("WARNING: no ground-truth edge mask in data; AUC will be skipped.")

# ---------- 2. Nodes to explain ----------
cal_nodes = splits["cal"].tolist()
test_nodes = splits["test"].tolist()
nodes = [(v, "cal") for v in cal_nodes] + [(v, "test") for v in test_nodes]
if args.quick:
    nodes = nodes[:5] + [(v, "test") for v in test_nodes[:5]]
print(f"Explaining {len(nodes)} nodes with GNNExplainer ({args.epochs} epochs each)")

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=args.epochs),
    explanation_type="model",
    node_mask_type=None,          # node features are constant, so only edges matter
    edge_mask_type="object",      # one importance score per edge
    model_config=dict(mode="multiclass_classification",
                      task_level="node", return_type="raw"),
)


def auc_score(scores, labels):
    """ROC-AUC without sklearn (Mann-Whitney). Returns None if undefined."""
    pos, neg = scores[labels], scores[~labels]
    if pos.numel() == 0 or neg.numel() == 0:
        return None
    greater = (pos[:, None] > neg[None, :]).float().mean().item()
    equal = (pos[:, None] == neg[None, :]).float().mean().item()
    return greater + 0.5 * equal


# ---------- 3. Run explanations ----------
edge_masks, records = [], []
t0 = time.time()
for i, (v, split) in enumerate(nodes, start=1):
    torch.manual_seed(SEED + v)  # reproducible per node
    expl = explainer(data.x, data.edge_index, index=v)
    em = expl.edge_mask.detach()
    edge_masks.append(em.cpu())

    # edges inside the node's 3-hop computation subgraph
    _, _, _, sub_edge = k_hop_subgraph(v, 3, data.edge_index,
                                       num_nodes=data.num_nodes)
    auc = None
    if has_gt and int(data.y[v]) > 0:   # only motif nodes have a ground truth
        auc = auc_score(em[sub_edge], gt_edge[sub_edge])

    records.append({
        "node": v, "split": split,
        "label": int(data.y[v]), "pred": int(preds[v]),
        "pred_prob": round(float(probs[v, preds[v]]), 4),
        "n_sub_edges": int(sub_edge.sum()),
        "auc": None if auc is None else round(auc, 4),
    })
    if i % 10 == 0 or i == len(nodes):
        print(f"  {i}/{len(nodes)} done  ({time.time() - t0:.0f} s)")

elapsed = time.time() - t0

# ---------- 4. Summary ----------
def mean_auc(split):
    vals = [r["auc"] for r in records if r["split"] == split and r["auc"] is not None]
    return (round(float(np.mean(vals)), 4), len(vals)) if vals else (None, 0)

cal_auc, cal_n = mean_auc("cal")
test_auc, test_n = mean_auc("test")
summary = {
    "dataset": "BA-Shapes", "seed": SEED, "device": str(device),
    "explainer": "GNNExplainer", "epochs": args.epochs, "quick": args.quick,
    "n_explained": len(nodes),
    "reloaded_test_acc": round(test_acc, 4),
    "mean_auc_cal_motif_nodes": cal_auc, "n_cal_motif_nodes": cal_n,
    "mean_auc_test_motif_nodes": test_auc, "n_test_motif_nodes": test_n,
    "seconds": round(elapsed, 1),
}
print("\nSUMMARY:", summary)

tag = "_quick" if args.quick else ""
torch.save({"records": records, "edge_masks": torch.stack(edge_masks)},
           os.path.join(RES_DIR, f"week2_explanations_bashapes{tag}.pt"))
with open(os.path.join(RES_DIR, f"week2_bashapes{tag}.json"), "w") as f:
    json.dump({"summary": summary, "records": records}, f, indent=2)

print("Saved:")
print("  ", os.path.join(RES_DIR, f"week2_explanations_bashapes{tag}.pt"))
print("  ", os.path.join(RES_DIR, f"week2_bashapes{tag}.json"))

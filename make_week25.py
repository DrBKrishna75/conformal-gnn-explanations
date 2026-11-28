# Creates week25_twitch_seeds.py from week19_twitch_conformal.py (only seed-related edits).
import hashlib
SRC = r"E:\conformal_gnn_project\week19_twitch_conformal.py"
DST = r"E:\conformal_gnn_project\week25_twitch_seeds.py"
s = open(SRC, encoding="utf-8-sig").read().replace("\r\n", "\n")


def rep(a, b):
    global s
    assert s.count(a) == 1, ("not found exactly once:", a[:60])
    s = s.replace(a, b)


header = '''# Week 25: Twitch natural shift with further seeds (seed given on the command line).
# Identical to week19_twitch_conformal.py except that, for the given seed, it
# (1) draws a new DE split and retrains GCN and GAT exactly as week18_twitch_inspect.py, and
# (2) offsets every random seed of the calibration family and test sampling by 100000*SEED.
# With SEED = 0 every seed equals Week 19 (split and models of Week 18).
# Run:  python week25_twitch_seeds.py 1     then     python week25_twitch_seeds.py 2
'''
rep("# Week 19: conformal explanation sets under NATURAL cross-country shift (Twitch)\n",
    header + "# Based on Week 19: conformal explanation sets under NATURAL cross-country shift (Twitch)\n")
rep("import os, json, time, random, csv, math, copy\n", "import os, sys, json, time, random, csv, math, copy\n")
rep("SEED = 0\n", "SEED = int(sys.argv[1])\nOFF = 100000 * SEED\nEPOCHS = 300\n")
rep('LOG_FILE = r"E:\\conformal_gnn_project\\week19_output.txt"',
    'LOG_FILE = rf"E:\\conformal_gnn_project\\week25_output_seed{SEED}.txt"')
rep('OUT_JSON = os.path.join(RES_DIR, "week19_twitch_conformal.json")',
    'OUT_JSON = os.path.join(RES_DIR, f"week25_twitch_seed{SEED}.json")')
rep('log("Week 19 - Twitch natural shift: conformal explanation sets (CPU)")',
    'log(f"Week 25 - Twitch natural shift, seed {SEED} (CPU)")')
rep('''split = torch.load(os.path.join(RES_DIR, "week18_DE_split_seed0.pt"))
log(f"Loaded 6 graphs, feature dim {F_DIM}; DE split cal {len(split['cal'])}, test {len(split['test'])}")''',
'''# ---------- DE split exactly as Week 18, with this seed ----------
_n = data["DE"].num_nodes
_perm = torch.randperm(_n, generator=torch.Generator().manual_seed(SEED))
_ntr, _nva, _nca = int(0.60 * _n), int(0.10 * _n), int(0.15 * _n)
split = {"train": _perm[:_ntr], "val": _perm[_ntr:_ntr + _nva],
         "cal": _perm[_ntr + _nva:_ntr + _nva + _nca], "test": _perm[_ntr + _nva + _nca:]}
torch.save(split, os.path.join(RES_DIR, f"week25_DE_split_seed{SEED}.pt"))
log(f"Loaded 6 graphs, feature dim {F_DIM}; DE split cal {len(split['cal'])}, test {len(split['test'])}")''')
rep('''results = {}
''', '''# ---------- train GCN and GAT on DE exactly as Week 18, with this seed ----------
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
''')
rep('model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"week18_{kind}_DE_seed0.pt")))',
    'model.load_state_dict(STATES[kind])')
rep('log(f"\\n===== Model {kind} (trained on DE, Week 18, seed 0) =====")',
    'log(f"\\n===== Model {kind} (trained on DE, seed {SEED}) =====")')
rep('perturb(data["DE"], dl, rh, 1000 + cell_id)', 'perturb(data["DE"], dl, rh, OFF + 1000 + cell_id)')
rep('2000 + cell_id, f"cell', 'OFF + 2000 + cell_id, f"cell')
rep("N_TEST + N_FEW, 3000 + i,", "N_TEST + N_FEW, OFF + 3000 + i,")
rep('f"week19_records_{kind}.pt"', 'f"week25_records_{kind}_seed{SEED}.pt"')
open(DST, "w", encoding="utf-8", newline="\n").write(s)
print("Created", DST, "| check code:", hashlib.sha256(s.encode("utf-8")).hexdigest()[:12])

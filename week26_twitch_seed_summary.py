# Week 26: combine Twitch results over seeds 0, 1, 2 (no new runs).
# Seed 0 = week19_twitch_conformal.json; seeds 1, 2 = week25_twitch_seed{1,2}.json.
import os, json
import numpy as np

RES_DIR = r"E:\results"
OUT_JSON = os.path.join(RES_DIR, "week26_twitch_seed_summary.json")
LOG_FILE = r"E:\conformal_gnn_project\week26_output.txt"
FILES = {0: "week19_twitch_conformal.json", 1: "week25_twitch_seed1.json", 2: "week25_twitch_seed2.json"}
MODELS = ["GCN", "GAT"]
COUNTRIES = ["DE", "EN", "ES", "FR", "PT", "RU"]
TARGETS = ["0.95", "0.90", "0.80"]
METHODS = ["standard", "pooled", "weighted", "fewshot20"]
open(LOG_FILE, "w").close()


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(s + "\n")


R = {s: json.load(open(os.path.join(RES_DIR, f))) for s, f in FILES.items()}
out = {}
for kind in MODELS:
    out[kind] = {}
    for t in TARGETS:
        key = f"target_{t}"
        tv = float(t)
        out[kind][t] = {}
        log(f"\n===== {kind} | target {t} | coverage per seed (0, 1, 2) -> mean [min, max]; mean set size; mean abstain; n per seed =====")
        log(f"  {'Ctry':5}{'Method':11}{'seed0':>8}{'seed1':>8}{'seed2':>8}{'mean':>8}{'min':>7}{'max':>7}{'size':>7}{'abst':>7}  n")
        for c in COUNTRIES:
            out[kind][t][c] = {}
            for m in METHODS:
                cov, size, ab, n = [], [], [], []
                for s in (0, 1, 2):
                    e = R[s].get(kind, {}).get(c, {}).get(key, {}).get(m)
                    if e is None:
                        cov.append(None); continue
                    cov.append(e["coverage"]); size.append(e["set_size"]); ab.append(e["abstain"]); n.append(e.get("n"))
                got = [x for x in cov if x is not None]
                if not got:
                    continue
                row = {"per_seed": cov, "mean": round(float(np.mean(got)), 4), "min": min(got), "max": max(got),
                       "n_seeds": len(got), "below_target": int(sum(x < tv for x in got)),
                       "set_size": round(float(np.mean(size)), 2), "abstain": round(float(np.mean(ab)), 3), "n": n}
                out[kind][t][c][m] = row
                ps = "".join(f"{x:8.3f}" if x is not None else f"{'-':>8}" for x in cov)
                log(f"  {c:5}{m:11}{ps}{row['mean']:8.3f}{row['min']:7.3f}{row['max']:7.3f}{row['set_size']:7.1f}{row['abstain']:7.3f}  {n}")

log("\n===== Summary over the five target countries (EN, ES, FR, PT, RU) and three seeds =====")
log(f"  {'Model':6}{'Target':8}{'Method':11}{'cells':>6}{'below':>7}{'min cov':>9}{'mean cov':>10}{'mean size':>11}")
summ = {}
for kind in MODELS:
    for t in TARGETS:
        for m in METHODS:
            rows = [out[kind][t][c][m] for c in COUNTRIES[1:] if m in out[kind][t][c]]
            vals = [x for r in rows for x in r["per_seed"] if x is not None]
            if not vals:
                continue
            below = sum(r["below_target"] for r in rows)
            sz = float(np.mean([r["set_size"] for r in rows]))
            summ[f"{kind}|{t}|{m}"] = {"cells": len(vals), "below_target": below, "min": min(vals),
                                       "mean": round(float(np.mean(vals)), 4), "mean_size": round(sz, 2)}
            log(f"  {kind:6}{t:8}{m:11}{len(vals):6}{below:7}{min(vals):9.3f}{np.mean(vals):10.3f}{sz:11.1f}")

log("\n===== Model AUC on each country (seeds 1, 2 from week25; seed 0 is in week18_twitch_inspect.json) =====")
for s in (1, 2):
    for kind in MODELS:
        tr = R[s]["train"][kind]
        log(f"  seed {s} {kind}: " + ", ".join(f"{c} {tr[c]['auc']}" for c in COUNTRIES))
json.dump({"per_cell": out, "summary": summ}, open(OUT_JSON, "w"), indent=1)
log(f"\nSaved {OUT_JSON}")

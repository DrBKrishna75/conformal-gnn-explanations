# Week 24: coverage by node degree on Twitch (no new model runs).
# Uses the records saved by week19_twitch_conformal.py and the same conformal code.
# Checks first that overall coverage reproduces week19_twitch_conformal.json exactly.
import os, json, math
import numpy as np
import torch

RES_DIR = r"E:\results"
OUT_JSON = os.path.join(RES_DIR, "week24_degree_coverage.json")
LOG_FILE = r"E:\conformal_gnn_project\week24_output.txt"
MODELS = ["GCN", "GAT"]
COUNTRIES = ["DE", "EN", "ES", "FR", "PT", "RU"]
TARGETS = ["EN", "ES", "FR", "PT", "RU"]
ALPHAS = [0.05, 0.10, 0.20]
BINS = [(1, 5, "deg 1-5"), (6, 20, "deg 6-20"), (21, 200, "deg 21-200")]
open(LOG_FILE, "w").close()


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(s + "\n")


# ---- copied unchanged from week19_twitch_conformal.py ----
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
# ---- end of copied code ----


def node_rows(te, ks):
    rows = []
    for r, k in zip(te, ks):
        a = math.isinf(k)
        sz = r["nv"] if a else min(k, r["nv"])
        rows.append({"nv": r["nv"], "cov": bool(a or r["S"] <= k), "whole": bool(sz >= r["nv"]),
                     "abst": bool(a), "size": float(sz)})
    return rows


def summ(rows):
    if not rows:
        return {"n": 0}
    return {"n": len(rows),
            "coverage": round(float(np.mean([x["cov"] for x in rows])), 4),
            "whole": round(float(np.mean([x["whole"] for x in rows])), 3),
            "abstain": round(float(np.mean([x["abst"] for x in rows])), 3),
            "set_size": round(float(np.mean([x["size"] for x in rows])), 2),
            "cov_selective": (round(float(np.mean([x["cov"] for x in rows if not x["whole"]])), 4)
                              if any(not x["whole"] for x in rows) else None),
            "n_selective": int(sum(not x["whole"] for x in rows))}


ref = json.load(open(os.path.join(RES_DIR, "week19_twitch_conformal.json")))
out, n_ok, n_all = {}, 0, 0
for kind in MODELS:
    rec = torch.load(os.path.join(RES_DIR, f"week19_records_{kind}.pt"), weights_only=False)
    family, tests = rec["family"], rec["tests"]
    cal_all = [r for recs in family.values() for r in recs]
    cal_std = family[(0.0, 0.0)]
    out[kind] = {}
    for a in ALPHAS:
        key = f"target_{1 - a:.2f}"
        k_std, k_pool = split_k([r["S"] for r in cal_std], a), split_k([r["S"] for r in cal_all], a)
        per = {"standard": {}, "pooled": {}, "weighted": {}}
        for c in COUNTRIES:
            te = tests[c][0]
            if len(te) == 0:
                continue
            per["standard"][c] = node_rows(te, [k_std] * len(te))
            per["pooled"][c] = node_rows(te, [k_pool] * len(te))
            per["weighted"][c] = node_rows(te, weighted_k(cal_all, te, a))
            for m in per:
                got = round(float(np.mean([x["cov"] for x in per[m][c]])), 4)
                want = ref[kind][c][key][m]["coverage"]
                n_all += 1
                n_ok += int(abs(got - want) < 1e-9)
        res = {}
        for m in per:
            res[m] = {}
            for scope, cs in (("DE (in-support)", ["DE"]), ("other countries (pooled)", TARGETS)):
                rows = [x for c in cs if c in per[m] for x in per[m][c]]
                res[m][scope] = {lab: summ([x for x in rows if lo <= x["nv"] <= hi]) for lo, hi, lab in BINS}
                res[m][scope]["all"] = summ(rows)
        out[kind][key] = res

log(f"REPRODUCIBILITY (overall coverage vs week19_twitch_conformal.json): {n_ok} of {n_all} identical")
for kind in MODELS:
    for a in ALPHAS:
        key = f"target_{1 - a:.2f}"
        log(f"\n===== {kind} | target {1 - a:.0%} | coverage / whole-set rate / n   [coverage of selective sets (n)] =====")
        for scope in ("DE (in-support)", "other countries (pooled)"):
            log(f"  {scope}")
            for m in ("standard", "pooled", "weighted"):
                cells = []
                for lab in [b[2] for b in BINS] + ["all"]:
                    s = out[kind][key][m][scope][lab]
                    if s["n"] == 0:
                        cells.append(f"{lab}: -")
                    else:
                        sel = "-" if s["cov_selective"] is None else f"{s['cov_selective']:.3f} ({s['n_selective']})"
                        cells.append(f"{lab}: {s['coverage']:.3f}/{s['whole']:.2f}/{s['n']} [{sel}]")
                log(f"    {m:9} " + " | ".join(cells))
json.dump(out, open(OUT_JSON, "w"), indent=1)
log(f"\nSaved {OUT_JSON}")

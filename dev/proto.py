#!/usr/bin/env python3
"""Prototype candidate forecasters, scored on practice p00..p04 (truth from next case)."""
import json, os, sys
import numpy as np

KIT = os.path.expanduser("~/workspace/makefun/polymarket-faa-attribution-01-kit")
IN = os.path.join(KIT, "input")
TAUS = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
Z = np.array([-1.644854, -0.67449, 0.0, 0.67449, 1.644854])

def load_cases():
    cases = {}
    for k in range(6):
        with open(os.path.join(IN, f"p0{k}.in")) as f:
            cases[f"p0{k}"] = json.load(f)
    return cases

CASES = load_cases()
# truth: {cid: {mid: np.array(240)}}
TRUTH = {}
for k in range(5):
    cid, nxt = f"p0{k}", f"p0{k+1}"
    TRUTH[cid] = {m["id"]: np.array(nm["prices"][-240:], dtype=float)
                  for m in CASES[cid]["markets"]
                  for nm in CASES[nxt]["markets"] if nm["id"] == m["id"]}

def score_forecasts(forecasts):
    """forecasts: {cid: {mid: (240,5) array}} -> dict of metrics."""
    tot, n = 0.0, 0
    med, mn = 0.0, 0
    cov, cn = 0, 0
    per = {}
    for cid, fmap in forecasts.items():
        cl, cn_ = 0.0, 0
        for mid, ys in TRUTH[cid].items():
            q = fmap[mid]
            d = ys[:, None] - q  # (240,5)
            loss = np.maximum(TAUS * d, (TAUS - 1.0) * d) * 100.0
            cl += loss.sum(); cn_ += loss.size
            med += np.abs(ys - q[:, 2]).sum() * 100.0; mn += 240
            cov += ((q[:, 0] <= ys) & (ys <= q[:, 4])).sum(); cn += 240
        per[cid] = cl / cn_
        tot += cl; n += cn_
    return {"loss": tot / n, "median_error": med / mn, "coverage_90": cov / cn, "per_case": per}

def price6(x):
    return np.round(np.clip(x, 0, 1) * 1e6) / 1e6

def run_model(model_fn, **kw):
    forecasts = {}
    for cid in [f"p0{k}" for k in range(5)]:
        fmap = {}
        for m in CASES[cid]["markets"]:
            p = np.array(m["prices"], dtype=float)
            fmap[m["id"]] = model_fn(p, **kw)
        forecasts[cid] = fmap
    return score_forecasts(forecasts)

# ---------------- candidate models: each returns (240,5) array ----------------
def m_baseline(p, sigma=0.0025):
    x = p[-1]
    h = np.arange(1, 241)
    return price6(x + np.outer(np.sqrt(h), Z * sigma))

def realized_sigma(p, window, min_n=30):
    d = np.diff(p[-window:]) if len(p) > 1 else np.array([0.0])
    if len(d) < min_n:
        d = np.diff(p)
    return float(np.std(d)) if len(d) else 0.0

def m_vol(p, window=720, mult=1.0, floor=0.0, cap=0.05, alpha=0.5):
    x = p[-1]
    s = realized_sigma(p, window)
    s = min(max(s * mult, floor), cap)
    h = np.arange(1, 241, dtype=float)
    return price6(x + np.outer(h ** alpha, Z * s))

def m_vol_mr(p, window=720, mult=1.0, floor=0.0, cap=0.05, alpha=0.5,
             mr_window=4320, mr_strength=0.0):
    """Mean-reverting median toward long-window mean; vol-scaled bands."""
    x = p[-1]
    s = min(max(realized_sigma(p, window) * mult, floor), cap)
    mu = float(np.mean(p[-mr_window:]))
    h = np.arange(1, 241, dtype=float)
    # median path: x + (mu - x) * (1 - exp(-h/tau)), tau from strength
    if mr_strength > 0:
        tau = 240.0 / mr_strength  # strength 1 -> ~63% reversion by h=240
        med = x + (mu - x) * (1.0 - np.exp(-h / tau))
    else:
        med = np.full(240, x)
    return price6(med[:, None] + np.outer(h ** alpha, Z * s))

def m_vol_drift(p, window=720, mult=1.0, floor=0.0, cap=0.05, alpha=0.5,
                drift_window=60, drift_shrink=0.0):
    x = p[-1]
    s = min(max(realized_sigma(p, window) * mult, floor), cap)
    h = np.arange(1, 241, dtype=float)
    if drift_shrink > 0 and len(p) > drift_window:
        slope = (p[-1] - p[-1 - drift_window]) / drift_window  # per-minute drift
        med = x + slope * drift_shrink * h
    else:
        med = np.full(240, x)
    return price6(med[:, None] + np.outer(h ** alpha, Z * s))

def m_empirical(p, window=4320, shrink=0.0):
    """Non-parametric: empirical quantiles of h-step changes over history."""
    n = len(p)
    h = np.arange(1, 241)
    q = np.empty((240, 5))
    x = p[-1]
    hist = p[-window:]
    N = len(hist)
    for j, hh in enumerate(h):
        if N - hh < 50:
            d = np.diff(hist)
            s = float(np.std(d)) if len(d) else 0.0025
            q[j] = x + Z * s * np.sqrt(hh)
            continue
        d = hist[hh:] - hist[:-hh]
        qs = np.quantile(d, TAUS)
        q[j] = x + qs
    # enforce ordering + range (quantile() already ordered)
    return price6(q)

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    def show(name, res):
        print(f"{name:38s} loss={res['loss']:.6f} med={res['median_error']:.6f} "
              f"cov90={res['coverage_90']:.4f} " +
              " ".join(f"{k}:{v:.4f}" for k, v in res["per_case"].items()))

    if which in ("all", "base"):
        show("A baseline sigma=0.0025", run_model(m_baseline))
    if which in ("all", "vol"):
        for w in (240, 720, 1440, 4320):
            for mult in (0.7, 1.0, 1.3):
                show(f"B vol w={w} mult={mult}", run_model(m_vol, window=w, mult=mult))
    if which in ("all", "alpha"):
        for alpha in (0.35, 0.4, 0.45, 0.5, 0.6):
            show(f"B vol w=720 mult=1 alpha={alpha}", run_model(m_vol, window=720, mult=1.0, alpha=alpha))
    if which in ("all", "mr"):
        for ms in (0.25, 0.5, 1.0, 2.0):
            show(f"C vol+MR strength={ms}", run_model(m_vol_mr, window=720, mult=1.0, mr_strength=ms))
    if which in ("all", "drift"):
        for ds in (0.25, 0.5, 1.0):
            show(f"D vol+drift shrink={ds}", run_model(m_vol_drift, window=720, mult=1.0, drift_shrink=ds))
    if which in ("all", "emp"):
        show("E empirical h-step quantiles", run_model(m_empirical))

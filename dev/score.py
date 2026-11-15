#!/usr/bin/env python3
"""Local practice scorer for polymarket-faa-attribution-01.

Truth for case p0k (k=0..4) is taken from case p0(k+1)'s input: its prices
cover minutes up to its own checkpoint, so the last 240 entries are the
actual prices at p0k.at+1 .. p0k.at+240. p05 has no successor, so it is
unscored (it is only used as truth for p04).

Pinball loss per RULES: 100 * max(tau*(y-q), (tau-1)*(y-q)), mean over all
(case, market, minute, tau). Also reports median_error and coverage_90.
"""
import json, math, os, subprocess, sys

KIT = os.path.expanduser("~/workspace/makefun/polymarket-faa-attribution-01-kit")
IN = os.path.join(KIT, "input")
TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]

def load_cases():
    cases = {}
    for k in range(6):
        with open(os.path.join(IN, f"p0{k}.in")) as f:
            cases[f"p0{k}"] = json.load(f)
    return cases

def truth_for(cases, cid):
    """Return {market_id: [240 actual prices]} for case cid, or None."""
    k = int(cid[2:])
    if k >= 5:
        return None
    nxt = cases[f"p0{k+1}"]
    cur = cases[cid]
    truth = {}
    for m in cur["markets"]:
        nm = next((x for x in nxt["markets"] if x["id"] == m["id"]), None)
        if nm is None:
            return None
        # continuity check: price at cur.at must equal nm's price 240 back
        assert nm["prices"][-241] == m["prices"][-1], f"continuity fail {cid} {m['id']}"
        truth[m["id"]] = nm["prices"][-240:]
    return truth

def score_outs(outdir, cases):
    total_loss, total_n = 0.0, 0
    med_err_sum, med_n = 0.0, 0
    cov_hit, cov_n = 0, 0
    per_case = {}
    for cid in [f"p0{k}" for k in range(5)]:
        truth = truth_for(cases, cid)
        path = os.path.join(outdir, f"{cid}.out")
        with open(path) as f:
            out = json.load(f)
        fmap = {fc["id"]: fc["q"] for fc in out["forecasts"]}
        closs, cn = 0.0, 0
        for mid, ys in truth.items():
            q = fmap[mid]
            assert len(q) == 240 and all(len(r) == 5 for r in q), f"shape {cid} {mid}"
            for h in range(240):
                y = ys[h]
                row = q[h]
                assert all(row[i] <= row[i+1] for i in range(4)), f"unordered {cid} {mid} h={h}"
                assert all(0.0 <= v <= 1.0 for v in row), f"out of range {cid} {mid} h={h}"
                for j, tau in enumerate(TAUS):
                    d = y - row[j]
                    closs += 100.0 * max(tau * d, (tau - 1.0) * d)
                    cn += 1
                med_err_sum += 100.0 * abs(y - row[2]); med_n += 1
                if row[0] <= y <= row[4]:
                    cov_hit += 1
                cov_n += 1
        per_case[cid] = closs / cn
        total_loss += closs; total_n += cn
    return {
        "loss": total_loss / total_n,
        "median_error": med_err_sum / med_n,
        "coverage_90": cov_hit / cov_n,
        "per_case": per_case,
    }

def run_entry(entry_dir, main_file, outdir, runner=("node",)):
    """Run the entry once per case file, like the grader (one case per run)."""
    os.makedirs(outdir, exist_ok=True)
    for k in range(6):
        cid = f"p0{k}"
        single_in = os.path.join(entry_dir, "dev", f"single-{cid}")
        os.makedirs(single_in, exist_ok=True)
        # copy just this case file in
        import shutil
        shutil.copy(os.path.join(IN, f"{cid}.in"), os.path.join(single_in, f"{cid}.in"))
        env = dict(os.environ, MAKEFUN_INPUT_DIR=single_in, MAKEFUN_OUTPUT_DIR=outdir)
        r = subprocess.run(list(runner) + [os.path.join(entry_dir, main_file)],
                           env=env, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            print(f"RUN FAILED {cid}: {r.stderr[:2000]}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    outdir = sys.argv[1]
    cases = load_cases()
    # sanity: continuity assertions run inside truth_for
    for cid in [f"p0{k}" for k in range(5)]:
        truth_for(cases, cid)
    print("truth alignment OK (continuity verified p00..p04)")
    res = score_outs(outdir, cases)
    print(f"loss={res['loss']:.6f} median_error={res['median_error']:.6f} "
          f"coverage_90={res['coverage_90']:.6f}")
    for cid, v in res["per_case"].items():
        print(f"  {cid}: {v:.6f}")

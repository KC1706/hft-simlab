"""P3.3 — stylized-fact validation: KS distances between two event-row datasets.

Compares a *generated* event stream against the *real* one on the microstructure
fingerprints of P1.3 (docs/JOURNAL.md Entry 4), quantified as KS distances — not
eyeballed. Both inputs are CSVs in the lob_export schema (core/.../bin/lob_export):
6 order features + 40-col top-10 book + mid/spread/imbalance/vwap. Real-vs-real gives
KS ~ 0 (the sanity floor); a realistic generator should keep every KS small.

Facts (event-indexed, so no time-grid alignment is required):
  spread          KS on per-event spread (ticks)
  ret             KS on standardized event-time mid log-returns (fat tails)
  size            KS on order/trade sizes
  dt              KS on inter-arrival times
  kurtosis        |excess kurtosis| gap of event returns (heavy tails)
  signacf_slope   gap in the power-law slope of the trade-sign ACF (long memory)
  imbalance       KS on top-10 volume imbalance

Run:  python scripts/p33_validate.py <real.csv> <gen.csv> [--out experiments/data/p33]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import ks_2samp

KS_CAP = 50_000  # subsample cap so ks_2samp stays fast on millions of rows


def load(path):
    df = pl.read_csv(path, infer_schema_length=None)
    bid1, ask1 = df["bid_p1"].to_numpy().astype(float), df["ask_p1"].to_numpy().astype(float)
    mid = (bid1 + ask1) / 2.0
    # Spread is inherently DISCRETE (integer ticks). Round before comparing: the z-score denorm
    # round-trip leaves real spreads at 1.0 +/- 1e-5, and KS against an exactly-1.0 generated
    # spread would report a spurious ~0.6 from that sub-tick float noise. Round -> compare at the
    # only resolution that means anything.
    spread = (df["spread"].to_numpy().astype(float) if "spread" in df.columns else ask1 - bid1)
    spread = np.round(spread)
    size = df["size"].to_numpy().astype(float)
    imb = (df["imbalance"].to_numpy().astype(float) if "imbalance" in df.columns
           else np.full(len(mid), np.nan))

    m = mid.copy()
    m[m <= 0] = np.nan
    ret = np.diff(np.log(m))
    ret = ret[np.isfinite(ret)]

    t = df["time"].to_numpy().astype(float)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt >= 0)]

    # Trade signs from execution events. lob_export emits raw event_type 4=EXECUTION;
    # a packed/remapped stream uses 2. Accept either so this works on both.
    et = df["event_type"].to_numpy()
    direction = df["direction"].to_numpy().astype(float)
    exec_mask = (et == 4) if (et == 4).any() else (et == 2)
    signs = direction[exec_mask] if exec_mask.any() else direction.astype(float)

    return {"spread": spread, "ret": ret, "size": size, "dt": dt,
            "imbalance": imb[np.isfinite(imb)], "signs": signs.astype(float)}


def excess_kurt(x):
    if len(x) < 10:
        return float("nan")
    z = (x - x.mean()) / (x.std() + 1e-12)
    return float((z**4).mean() - 3.0)


def sign_acf_slope(signs):
    """Power-law slope of the trade-sign ACF on log-log (long-memory order flow)."""
    s = signs - signs.mean()
    v = (s * s).sum()
    if v == 0 or len(s) < 100:
        return float("nan")
    top = min(3000, len(s) // 4)
    lags = np.unique(np.logspace(0, np.log10(max(top, 11)), 30).astype(int))
    lags = lags[(lags >= 1) & (lags < len(s))]
    ac = np.array([(s[: len(s) - l] * s[l:]).sum() / v for l in lags])
    mask = (ac > 0) & (lags >= 10)
    if mask.sum() < 5:
        return float("nan")
    slope, _ = np.polyfit(np.log(lags[mask]), np.log(ac[mask]), 1)
    return float(slope)


def ks(a, b):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    n = min(len(a), len(b))
    if n < 10:
        return None
    r = ks_2samp(a[:KS_CAP], b[:KS_CAP])
    return {"ks": float(r.statistic), "p": float(r.pvalue)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("real")
    ap.add_argument("gen")
    ap.add_argument("--out", default="experiments/data/p33")
    args = ap.parse_args()

    real, gen = load(args.real), load(args.gen)
    facts = {}
    for k in ("spread", "ret", "size", "dt", "imbalance"):
        facts[k] = ks(real[k], gen[k])
    kr, kg = excess_kurt(real["ret"]), excess_kurt(gen["ret"])
    facts["kurtosis"] = {"real": kr, "gen": kg, "abs_diff": abs(kr - kg)}
    sr, sg = sign_acf_slope(real["signs"]), sign_acf_slope(gen["signs"])
    facts["signacf_slope"] = {"real": sr, "gen": sg,
                              "abs_diff": (abs(sr - sg) if np.isfinite(sr) and np.isfinite(sg) else None)}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ks_report.json").write_text(json.dumps(facts, indent=2))

    print(f"P3.3 stylized-fact validation — real={args.real}  gen={args.gen}")
    print(f"{'fact':<14} | {'metric':<10} | {'real':>10} | {'gen':>10}")
    print("-" * 56)
    for k in ("spread", "ret", "size", "dt", "imbalance"):
        v = facts[k]
        if v is None:
            print(f"{k:<14} | {'KS':<10} | {'n/a':>10} | {'n/a':>10}")
        else:
            print(f"{k:<14} | {'KS=' + format(v['ks'], '.4f'):<10} | {'p=' + format(v['p'], '.2g'):>10} |")
    for k in ("kurtosis", "signacf_slope"):
        v = facts[k]
        d = "n/a" if v["abs_diff"] is None else f"{v['abs_diff']:.3f}"
        rr = "nan" if not np.isfinite(v["real"]) else f"{v['real']:.3f}"
        gg = "nan" if not np.isfinite(v["gen"]) else f"{v['gen']:.3f}"
        print(f"{k:<14} | {'Δ=' + d:<10} | {rr:>10} | {gg:>10}")
    print(f"\nsaved {out/'ks_report.json'}")
    print("interpretation: KS→0 and Δ→0 mean the generator reproduces that stylized fact;")
    print("large KS (e.g. >0.1) flags a fingerprint the model fails to match.")


if __name__ == "__main__":
    main()

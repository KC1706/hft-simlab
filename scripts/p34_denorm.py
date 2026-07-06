"""P3.4 step 1: invert p31's normalization and rebuild the lob_export-schema CSV.

This is the deterministic half of P3.4 — the exact inverse of scripts/p31_pack_trades.py.
The trained TRADES model (P3.2) generates order events in the *normalized* 46-column layout
it was trained on ([dt, event_type, size, price, direction, depth] + 40 LOB cols). To score
a generated stream with scripts/p33_validate.py — or to feed it back into the simulator — the
values must be turned back into prices/sizes/times. This module:

  1. de-normalizes the z-scored columns with the means/stds saved in experiments/data/p31/
     stats.json (the file p31 kept *precisely* so this inversion is possible),
  2. reconstructs wall-clock time from the inter-arrival column (time = cumsum(dt)),
  3. recomputes mid / spread / imbalance / vwap from the de-normalized top-10 book with the
     SAME formulas as core/src/bin/lob_export.rs, and
  4. writes a CSV byte-compatible with the lob_export schema, so p33_validate.py reads it
     with no special-casing.

The GPU sampling step (run on Kaggle once the P3.2 checkpoint exists) produces the input
array: each generated order paired with its conditioning top-10 book snapshot, in p31's
column order. This script is GPU-free and is verified locally by a round-trip self-test
(`--self-test`) that needs no checkpoint — see the manual test in docs/JOURNAL.md Entry 13.

Run:
  python scripts/p34_denorm.py <generated.npy> [--out /tmp/generated.csv]
  python scripts/p34_denorm.py --self-test          # round-trip val.npy, no model needed
"""
import argparse
import json
from pathlib import Path

import numpy as np

P31 = Path("experiments/data/p31")
LEVELS = 10


def load_stats(stats_path):
    s = json.loads(Path(stats_path).read_text())
    mean = np.asarray(s["mean"], dtype=np.float64)
    std = np.asarray(s["std"], dtype=np.float64)
    return s, mean, std


def denormalize(X, stats):
    """Invert the per-column z-score of p31 in place-safe fashion. Categorical columns
    (event_type, direction) were never scaled, so they pass through untouched."""
    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)
    out = X.astype(np.float64).copy()
    for c in stats["z_scored_cols"]:
        out[:, c] = out[:, c] * std[c] + mean[c]
    return out


def to_lob_export_rows(Xd, stats):
    """Map a de-normalized array (p31 layout) to lob_export columns + recomputed metrics."""
    order_feats = stats["order_features"]          # [dt, event_type, size, price, direction, depth]
    lob_cols = stats["lob_columns"]                # ask_p1,ask_s1,bid_p1,bid_s1, ... (40)
    n_ord = len(order_feats)
    idx = {name: i for i, name in enumerate(order_feats)}

    dt = np.clip(Xd[:, idx["dt"]], 0.0, None)      # inter-arrivals are non-negative
    time = np.cumsum(dt)                            # absolute origin is irrelevant: p33 uses diffs
    event_type = np.rint(Xd[:, idx["event_type"]]).astype(int)
    size = Xd[:, idx["size"]]
    price = Xd[:, idx["price"]]
    direction = np.rint(Xd[:, idx["direction"]]).astype(int)
    order_id = np.arange(1, len(Xd) + 1)

    lob = Xd[:, n_ord:n_ord + len(lob_cols)]       # the 40 book columns, p31 order
    col = {name: lob[:, j] for j, name in enumerate(lob_cols)}

    # Recompute the four convenience metrics exactly as lob_export.rs does.
    ask_p1, bid_p1 = col["ask_p1"], col["bid_p1"]
    mid = (bid_p1 + ask_p1) / 2.0
    spread = ask_p1 - bid_p1
    bid_vol = sum(col[f"bid_s{l}"] for l in range(1, LEVELS + 1))
    ask_vol = sum(col[f"ask_s{l}"] for l in range(1, LEVELS + 1))
    tot = bid_vol + ask_vol
    imbalance = np.where(tot > 0, bid_vol / np.where(tot > 0, tot, 1.0), 0.5)
    pv = sum(col[f"ask_p{l}"] * col[f"ask_s{l}"] + col[f"bid_p{l}"] * col[f"bid_s{l}"]
             for l in range(1, LEVELS + 1))
    vol = tot
    vwap = np.where(vol > 0, pv / np.where(vol > 0, vol, 1.0), mid)

    header = ["time", "event_type", "order_id", "size", "price", "direction"] + lob_cols \
        + ["mid", "spread", "imbalance", "vwap"]
    cols = [time, event_type, order_id, size, price, direction] \
        + [col[c] for c in lob_cols] + [mid, spread, imbalance, vwap]
    return header, cols


def write_csv(path, header, cols):
    arr = np.column_stack([np.asarray(c, dtype=np.float64) for c in cols])
    # event_type / order_id / direction are integers in the schema; format them as such.
    int_cols = {header.index("event_type"), header.index("order_id"), header.index("direction")}
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in arr:
            cells = [str(int(round(v))) if i in int_cols else repr(float(v))
                     for i, v in enumerate(row)]
            f.write(",".join(cells) + "\n")


def self_test(stats_path):
    """No checkpoint needed: prove the inverse is correct by round-tripping a real array.
    de-normalize(val.npy) -> re-normalize must return the original to float precision, and
    the de-normalized book must satisfy the schema's invariants (spread>=0, sizes>=0)."""
    stats, mean, std = load_stats(stats_path)
    X = np.load(P31 / "val.npy").astype(np.float64)
    Xd = denormalize(X, stats)

    # 1) invertibility: re-apply p31's z-score and compare to the stored normalized array.
    Xr = Xd.copy()
    for c in stats["z_scored_cols"]:
        Xr[:, c] = (Xr[:, c] - mean[c]) / std[c]
    max_err = float(np.abs(Xr - X).max())

    # 2) schema sanity on the de-normalized book.
    header, cols = to_lob_export_rows(Xd, stats)
    d = dict(zip(header, cols))
    spread, size = d["spread"], d["size"]
    et = set(np.unique(np.rint(d["event_type"]).astype(int)).tolist())
    print(f"[self-test] rows={len(X)}  features={X.shape[1]}")
    print(f"[self-test] round-trip max |renorm - original| = {max_err:.3e} "
          f"({'PASS' if max_err < 1e-3 else 'FAIL'} @ 1e-3)")
    print(f"[self-test] spread: min={spread.min():.3f} median={np.median(spread):.3f} "
          f"(ticks; expect >= 0)")
    print(f"[self-test] size:   min={size.min():.4f} median={np.median(size):.4f}")
    print(f"[self-test] event_type classes present: {sorted(et)} (expect subset of {{0,1,2}})")

    out = Path("/tmp/p34_selftest.csv")
    write_csv(out, header, cols)
    print(f"[self-test] wrote {out} — score the inverse against the real export with:")
    print(f"             python scripts/p33_validate.py /tmp/real.csv {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("generated", nargs="?", help="generated .npy in p31's normalized layout")
    ap.add_argument("--stats", default=str(P31 / "stats.json"))
    ap.add_argument("--out", default="/tmp/generated.csv")
    ap.add_argument("--self-test", action="store_true",
                    help="round-trip val.npy to verify the inverse (no checkpoint needed)")
    args = ap.parse_args()

    if args.self_test:
        self_test(args.stats)
        return
    if not args.generated:
        ap.error("provide a generated .npy, or use --self-test")

    stats, _, _ = load_stats(args.stats)
    X = np.load(args.generated).astype(np.float64)
    exp = len(stats["order_features"]) + len(stats["lob_columns"])
    if X.shape[1] != exp:
        raise SystemExit(f"expected {exp} columns (p31 layout), got {X.shape[1]} — "
                         "the array must be [order_features | lob] in stats.json order")
    Xd = denormalize(X, stats)
    header, cols = to_lob_export_rows(Xd, stats)
    write_csv(args.out, header, cols)
    print(f"de-normalized {len(X)} generated rows -> {args.out} (lob_export schema)")
    print(f"score it:  python scripts/p33_validate.py <real.csv> {args.out}")


if __name__ == "__main__":
    main()

"""P3.1 step 2: pack the lob_export CSV into the array TRADES trains on.

Reads the 50-column rows from `core/.../lob_export` (6 order features + 40 LOB + 4
metrics), engineers the TRADES order-feature vector, z-score normalizes, drops the
session-warm-up rows (book not yet 10 levels deep), splits train/val, and saves .npy
arrays plus a stats JSON (means/stds) so the same normalization can be inverted at
generation time.

Output layout per row: [order_features (6) | lob (40)] — matching refs/DeepMarket's
LOBDataset, which loads a single array and slices `orders` / `lob`. Exact parity with
TRADES' normalize_messages (their depth feature and inter-arrival convention) is a P3.2
task when we wire into their DataModule; this packer establishes the runnable pipeline.

Run from repo root:
  core/target/release/lob_export data/<day>.npz --out /tmp/rows.csv --max-rows 400000
  python3 scripts/p31_pack_trades.py /tmp/rows.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

CSV = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/trades_rows.csv")
OUT = Path("experiments/data/p31")
VAL_FRAC = 0.2
LEVELS = 10

# infer_schema_length=None scans the whole file: mid/vwap are integer-valued in early
# rows but fractional later, which would otherwise mis-infer them as Int64.
df = pl.read_csv(CSV, infer_schema_length=None)

# Drop warm-up rows where the top-10 book has not filled (any level price == 0).
level_price_cols = [c for c in df.columns if c.startswith(("ask_p", "bid_p"))]
mask = pl.fold(acc=pl.lit(True), function=lambda a, s: a & (s != 0),
               exprs=[pl.col(c) for c in level_price_cols])
df = df.filter(mask)

# Order-feature engineering: inter-arrival time, and depth = |price - mid| in ticks.
df = df.with_columns(
    (pl.col("time").diff().fill_null(0.0)).alias("dt"),
    (pl.col("price") - pl.col("mid")).abs().alias("depth"),
)

order_feats = ["dt", "event_type", "size", "price", "direction", "depth"]
lob_cols = [c for c in df.columns if c.startswith(("ask_p", "ask_s", "bid_p", "bid_s"))]

X = df.select(order_feats + lob_cols).to_numpy().astype(np.float64)
n, d = X.shape

# z-score the continuous columns; leave event_type/direction (categorical) untouched.
z_cols = [order_feats.index(c) for c in ("dt", "size", "price", "depth")]
z_cols += list(range(len(order_feats), d))  # all LOB columns
mean = X.mean(axis=0)
std = X.std(axis=0)
std[std == 0] = 1.0
for c in z_cols:
    X[:, c] = (X[:, c] - mean[c]) / std[c]

split = int(n * (1 - VAL_FRAC))
OUT.mkdir(parents=True, exist_ok=True)
np.save(OUT / "train.npy", X[:split].astype(np.float32))
np.save(OUT / "val.npy", X[split:].astype(np.float32))
stats = {
    "n_rows": n,
    "n_features": d,
    "order_features": order_feats,
    "lob_columns": lob_cols,
    "z_scored_cols": z_cols,
    "mean": mean.tolist(),
    "std": std.tolist(),
    "levels": LEVELS,
    "train_rows": split,
    "val_rows": n - split,
}
(OUT / "stats.json").write_text(json.dumps(stats, indent=2))
print(f"packed {n} rows x {d} cols -> {OUT}/train.npy ({split}) + val.npy ({n - split})")
print(f"order features: {order_feats}")

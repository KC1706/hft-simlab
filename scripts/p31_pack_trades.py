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

# TRADES uses a fixed 3-class type embedding (0=submission, 1=cancel/delete, 2=execution).
# Remap our raw {1=SUB, 2=CANCEL, 3=DELETE, 4=EXEC} -> {0, 1, 1, 2}, mirroring DeepMarket's
# normalize_messages (event_type-1; replace(2,1); replace(3,2)). Without this the raw codes
# 3 and 4 index past the 3-slot embedding -> CUDA device-side assert.
df = df.with_columns(
    pl.when(pl.col("event_type") == 1).then(0)
      .when(pl.col("event_type") == 4).then(2)
      .otherwise(1)  # cancellation (2) and deletion (3) merge into the 'remove' class
      .alias("event_type")
)

# --- Event-time spreading (L2->L3 timing reconstruction) ---------------------------------
# Our data is L2 snapshots: each exchange update at time T unpacks into MANY events (mean ~21,
# up to ~1266) all stamped with the same T, so raw inter-arrival dt is ~95% exact zeros -- a
# degenerate point mass that a continuous generator cannot reproduce (KS(dt) stuck ~0.54, see
# docs/JOURNAL.md). We reconstruct plausible per-event timing by distributing the k events of a
# snapshot uniformly across the gap [T, T_next) to the following snapshot. This is an explicit
# ASSUMPTION (uniform intra-snapshot spacing) -- the true sub-timestamp order timing is not in
# the L2 feed -- but it turns dt into a smooth positive distribution the model can learn.
def _spread_snapshot_times(t: np.ndarray) -> np.ndarray:
    t = t.astype(np.float64)
    n = len(t)
    out = t.copy()
    distinct = np.unique(t)
    med_gap = float(np.median(np.diff(distinct))) if len(distinct) > 1 else 1e-3
    i = 0
    while i < n:
        j = i
        while j < n and t[j] == t[i]:
            j += 1
        k = j - i
        T = t[i]
        T_next = t[j] if j < n else T + med_gap          # last run: fall back to median gap
        out[i:j] = T + (np.arange(k) / k) * (T_next - T)  # k events uniformly in [T, T_next)
        i = j
    return out

df = df.with_columns(pl.Series("time", _spread_snapshot_times(df["time"].to_numpy())))

# Order-feature engineering: inter-arrival time, and depth = |price - mid| in ticks.
df = df.with_columns(
    (pl.col("time").diff().fill_null(0.0)).alias("dt"),
    (pl.col("price") - pl.col("mid")).abs().alias("depth"),
)

order_feats = ["dt", "event_type", "size", "price", "direction", "depth"]
lob_cols = [c for c in df.columns if c.startswith(("ask_p", "ask_s", "bid_p", "bid_s"))]

X = df.select(order_feats + lob_cols).to_numpy().astype(np.float64)
n, d = X.shape

# Log-transform strictly-positive, heavy-tailed order features (size, dt) BEFORE z-scoring.
# Reason: the model generates on an unbounded support, so a plain z-scored `size` de-normalizes
# to negatives ~50% of the time (impossible orders — see docs/JOURNAL.md Entry 14). Training on
# log(size) means the generation inverse is exp(), which is ALWAYS positive: negative sizes
# become mathematically impossible. Same argument for inter-arrival dt (>= 0). A tiny floor
# guards log(0) (simultaneous events give dt=0). p34_denorm.py inverts with exp() on these cols.
LOG_FEATS = ["dt", "size"]
LOG_FLOOR = 1e-9
log_idx = [order_feats.index(c) for c in LOG_FEATS]
for c in log_idx:
    X[:, c] = np.log(np.maximum(X[:, c], LOG_FLOOR))

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
    "log_cols": LOG_FEATS,      # inverted with exp() in p34 (guarantees positive size/dt)
    "log_floor": LOG_FLOOR,
    "mean": mean.tolist(),
    "std": std.tolist(),
    "levels": LEVELS,
    "train_rows": split,
    "val_rows": n - split,
}
(OUT / "stats.json").write_text(json.dumps(stats, indent=2))
print(f"packed {n} rows x {d} cols -> {OUT}/train.npy ({split}) + val.npy ({n - split})")
print(f"order features: {order_feats}")

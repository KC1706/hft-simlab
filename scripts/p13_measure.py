"""P1.3 step 1: run `replay --measure`, convert the two CSVs -> parquet.

The Rust replay binary streams the day once and logs two CSVs (samples.csv on a
100 ms grid: top-of-book, top-10 depth profile per side, Cont-Kukanov-Stoikov OFI,
signed trade flow; trades.csv: the full tape). This script converts them to the
parquet the figure step consumes.

Produces:
  experiments/data/p13/samples.parquet   one row per 100 ms grid interval
  experiments/data/p13/trades.parquet    the full trade tape (ts, sign, px_tick, qty)
These feed experiments/figures/p13_stylized_facts.py (FIG-10) and double as
FIG-7's real-side input.

Run from repo root: .venv/bin/python scripts/p13_measure.py
"""
import subprocess
from pathlib import Path

import polars as pl

DATA = "data/btcusdt_20260501_0000_0656.npz"
BINARY = "core/target/release/replay"
GRID_MS = "100"
TMP = Path("/tmp/p13_measure")
OUT = Path("experiments/data/p13")

TMP.mkdir(parents=True, exist_ok=True)
subprocess.run([BINARY, DATA, "--measure", str(TMP), "--grid-ms", GRID_MS], check=True)

OUT.mkdir(parents=True, exist_ok=True)
samples = pl.read_csv(TMP / "samples.csv")
samples.write_parquet(OUT / "samples.parquet")
trades = pl.read_csv(TMP / "trades.csv")
trades.write_parquet(OUT / "trades.parquet")

for f in (TMP / "samples.csv", TMP / "trades.csv"):
    f.unlink(missing_ok=True)

print(f"samples: {samples.height} rows -> {OUT / 'samples.parquet'}")
print(f"trades:  {trades.height} rows -> {OUT / 'trades.parquet'}")
print(samples.head())

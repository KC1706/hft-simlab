"""P2.2 step 1: run fill_labeler, convert CSV -> parquet.

Produces:
  experiments/data/p22/fill_labels.parquet  ~300k rows of virtual-order outcomes
Run from repo root: .venv/bin/python scripts/p22_fill_label.py
"""
import subprocess
from pathlib import Path

import polars as pl

DATA = "data/btcusdt_20260501_0000_0656.npz"
BINARY = "core/target/release/fill_labeler"
TMP = "/tmp/fill_labels.csv"
OUT = Path("experiments/data/p22")

subprocess.run([BINARY, DATA, "--out", TMP], check=True)
OUT.mkdir(parents=True, exist_ok=True)

df = pl.read_csv(TMP)
df.write_parquet(OUT / "fill_labels.parquet")
Path(TMP).unlink(missing_ok=True)
print(f"fill_labels: {df.height} rows -> {OUT/'fill_labels.parquet'}")
print(df.head())

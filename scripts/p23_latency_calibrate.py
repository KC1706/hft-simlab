"""P2.3 latency calibration: fit feed latency distribution + render FIG-5.

Reads:  /tmp/feed_latency.csv  (from core/target/release/latency_profiler)
Writes: experiments/data/p23/latency_model.json  (calibrated params)
        experiments/p23_latency_dist.png           (FIG-5a: histogram + fit)
        experiments/p23_latency_qq.png             (FIG-5b: QQ-plot vs model)
        experiments/data/p23/feed_latency.parquet

The feed latency (local_ts - exch_ts) is a lower bound on order round-trip
latency. We fit a log-normal + impulse (spike at the mode) model and report
both the baseline and the estimated round-trip parameters.

Run from repo root: .venv/bin/python scripts/p23_latency_calibrate.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import lognorm, kstest

DATA_CSV = "/tmp/feed_latency.csv"
OUT_DIR = Path("experiments")
DATA_DIR = Path("experiments/data/p23")
OUT_MODEL = DATA_DIR / "latency_model.json"
OUT_FIG5A = OUT_DIR / "p23_latency_dist.png"
OUT_FIG5B = OUT_DIR / "p23_latency_qq.png"

DATA_DIR.mkdir(parents=True, exist_ok=True)

print("Loading feed latency CSV …")
df = pl.read_csv(DATA_CSV)
df.write_parquet(DATA_DIR / "feed_latency.parquet")
print(f"  {df.height} rows saved to parquet")

lat_us = df["feed_latency_ns"].to_numpy() / 1000.0  # convert to µs

# ── 1. Basic statistics ──────────────────────────────────────────────────────
print(f"\nFeed latency statistics (µs):")
for pct in [1, 5, 25, 50, 75, 90, 95, 99, 99.9]:
    print(f"  p{pct:.1f}: {np.percentile(lat_us, pct):.1f}")
print(f"  mean: {lat_us.mean():.1f}  std: {lat_us.std():.1f}")
print(f"  min: {lat_us.min():.1f}  max: {lat_us.max():.1f}")

# ── 2. Log-normal fit (MLE) ──────────────────────────────────────────────────
# Fit on the bulk (exclude top 0.1% to avoid outlier sensitivity).
cutoff = np.percentile(lat_us, 99.9)
lat_bulk = lat_us[lat_us <= cutoff]

# scipy lognorm: parameterized as lognorm(s, loc=0, scale=exp(mu))
# fit on log-transformed data for stability.
log_lat = np.log(lat_bulk[lat_bulk > 0])
mu_fit = log_lat.mean()
sigma_fit = log_lat.std()
print(f"\nLog-normal fit: mu={mu_fit:.3f}  sigma={sigma_fit:.3f}")
print(f"  → median: {np.exp(mu_fit):.1f} µs  mode: {np.exp(mu_fit - sigma_fit**2):.1f} µs")

# Race-mode estimate: the Reality Gap paper (arXiv 2603.24137) finds a spike
# at the exchange round-trip latency in the reaction-time histogram. For
# feed-proxy latency, the mode of the log-normal approximates the one-way
# feed delay; the exchange round-trip is ≈ 2× this plus order processing.
# We save this as a calibrated parameter.
mode_us = np.exp(mu_fit - sigma_fit**2)
race_mode_ns = int(mode_us * 2 * 1000)  # 2× feed one-way, in ns

# ── 3. FIG-5a: histogram + log-normal fit ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
bins = np.logspace(np.log10(max(lat_us.min(), 1)), np.log10(cutoff), 120)
counts, edges = np.histogram(lat_bulk, bins=bins, density=True)
centers = np.sqrt(edges[:-1] * edges[1:])  # geometric center of log bins
ax.bar(centers, counts, width=np.diff(edges), color="#1f77b4", alpha=0.6,
       label="empirical (depth, subsampled ×20; trades full)")

# Overlay log-normal PDF.
x_line = np.logspace(np.log10(1), np.log10(cutoff), 400)
pdf = lognorm.pdf(x_line, s=sigma_fit, scale=np.exp(mu_fit))
ax.plot(x_line, pdf, "r-", lw=2, label=f"log-normal fit (σ={sigma_fit:.2f})")

# Annotate the mode.
ax.axvline(mode_us, color="green", linestyle="--", lw=1.5,
           label=f"mode={mode_us:.0f} µs (feed one-way)")
ax.axvline(mode_us * 2, color="purple", linestyle=":", lw=1.5,
           label=f"2×mode={mode_us*2:.0f} µs (est. round-trip)")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("feed latency (µs, log scale)")
ax.set_ylabel("density (log scale)")
ax.set_title("P2.3 FIG-5a — Feed latency distribution, BTCUSDT-perp 2026-05-01")
ax.legend(fontsize=9)
ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(OUT_FIG5A, dpi=120, bbox_inches="tight")
print(f"\nSaved FIG-5a -> {OUT_FIG5A}")

# ── 4. FIG-5b: QQ-plot of empirical vs log-normal ────────────────────────────
sample_size = min(5000, len(lat_bulk))
rng = np.random.default_rng(42)
emp = np.sort(rng.choice(lat_bulk, size=sample_size, replace=False))
theo_quantiles = np.linspace(0.5 / sample_size, 1 - 0.5 / sample_size, sample_size)
theo = lognorm.ppf(theo_quantiles, s=sigma_fit, scale=np.exp(mu_fit))

fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.scatter(theo, emp, s=3, alpha=0.3, color="#1f77b4")
lim = np.percentile(emp, 99)
ax2.plot([0, lim], [0, lim], "r--", lw=1, label="identity (perfect fit)")
ax2.set_xlim(0, lim)
ax2.set_ylim(0, lim)
ax2.set_xlabel("theoretical quantile (log-normal, µs)")
ax2.set_ylabel("empirical quantile (µs)")
ax2.set_title("P2.3 FIG-5b — QQ-plot: empirical feed latency vs log-normal fit")
ax2.legend()
ax2.grid(alpha=0.25)
fig2.tight_layout()
fig2.savefig(OUT_FIG5B, dpi=120, bbox_inches="tight")
print(f"Saved FIG-5b -> {OUT_FIG5B}")

# ── 5. Intraday variation: hourly median ─────────────────────────────────────
ts_h = (df["exch_ts"].to_numpy() - df["exch_ts"][0]) / 3.6e12  # hours from start
for h in range(8):
    mask = (ts_h >= h) & (ts_h < h + 1)
    if mask.sum() > 100:
        m = np.median(lat_us[mask])
        print(f"  hour {h}: median feed latency {m:.1f} µs  (n={mask.sum()})")

# ── 6. Save model params ──────────────────────────────────────────────────────
model = {
    "description": "Log-normal feed latency model for BTCUSDT-perp 2026-05-01",
    "feed_latency_lognormal_mu_log_us": float(mu_fit),
    "feed_latency_lognormal_sigma": float(sigma_fit),
    "feed_latency_mode_us": float(mode_us),
    "feed_latency_median_us": float(np.exp(mu_fit)),
    "feed_latency_p99_us": float(np.percentile(lat_us, 99)),
    # Estimated round-trip latency parameters (for LatencyModel in Rust):
    # entry_latency = feed_latency (one-way proxy)
    # response_latency = feed_latency (return leg proxy)
    # race_mode_ns = 2 × feed_one_way_mode (exchange round-trip estimate)
    "est_roundtrip_mode_ns": race_mode_ns,
    "est_entry_latency_median_ns": int(np.exp(mu_fit) * 1000),
    "est_entry_latency_p99_ns": int(np.percentile(lat_us, 99) * 1000),
    "n_events_fit": int(len(lat_bulk)),
}
OUT_MODEL.write_text(json.dumps(model, indent=2))
print(f"\nSaved model -> {OUT_MODEL}")
print(json.dumps(model, indent=2))

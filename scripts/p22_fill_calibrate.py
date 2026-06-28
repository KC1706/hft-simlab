"""P2.2 step 2: fit calibrated fill-probability model and render FIG-4.

Reads:  experiments/data/p22/fill_labels.parquet
Writes: experiments/data/p22/fill_model.json   (model params for the simulator)
        experiments/p22_fill_curves.png         (FIG-4a: empirical curves + fit)
        experiments/p22_reliability.png         (FIG-4b: reliability diagram, held-out)

The "model" is an isotonic regression fit per (side, horizon, criterion):
  P(fill | queue_frac) — monotone non-increasing in queue_frac.
Saved as a sorted quantile table so the Rust simulator can binary-search it.

Run from repo root: .venv/bin/python scripts/p22_fill_calibrate.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

DATA_DIR = Path("experiments/data/p22")
OUT_DIR = Path("experiments")
OUT_MODEL = DATA_DIR / "fill_model.json"
OUT_FIG4A = OUT_DIR / "p22_fill_curves.png"
OUT_FIG4B = OUT_DIR / "p22_reliability.png"

df = pl.read_parquet(DATA_DIR / "fill_labels.parquet")
print(f"Loaded {df.height} rows")

# Columns: ts_placed, side(0=bid/1=ask), price_tick, init_qty, queue_frac,
#          spread_ticks, fill_trade_1s/10s/60s, fill_any_1s/10s/60s

HORIZONS = ["1s", "10s", "60s"]
CRITERIA = ["trade", "any"]
SIDES = {0: "bid", 1: "ask"}

# ── 1. Build bins and empirical fill curves ──────────────────────────────────
FRAC_BINS = np.array([0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0])
bin_centers = (FRAC_BINS[:-1] + FRAC_BINS[1:]) / 2

def empirical_curves(df_side, criterion):
    """Return dict {horizon: (bin_means, bin_rates, bin_counts)}."""
    results = {}
    for hor in HORIZONS:
        col = f"fill_{criterion}_{hor}"
        frac = df_side["queue_frac"].to_numpy()
        y = df_side[col].to_numpy().astype(float)
        bin_idx = np.digitize(frac, FRAC_BINS) - 1
        bin_idx = np.clip(bin_idx, 0, len(FRAC_BINS) - 2)
        means, rates, counts = [], [], []
        for b in range(len(FRAC_BINS) - 1):
            mask = bin_idx == b
            if mask.sum() == 0:
                means.append(bin_centers[b])
                rates.append(np.nan)
                counts.append(0)
            else:
                means.append(frac[mask].mean())
                rates.append(y[mask].mean())
                counts.append(mask.sum())
        results[hor] = (np.array(means), np.array(rates), np.array(counts))
    return results

# ── 2. Fit isotonic regression per (side, horizon, criterion) ────────────────
model_params = {}
iso_fits = {}  # key=(side_str, crit, hor) -> (x_pts, y_pts)

for side_int, side_str in SIDES.items():
    df_s = df.filter(pl.col("side") == side_int)
    frac_all = df_s["queue_frac"].to_numpy()

    for crit in CRITERIA:
        for hor in HORIZONS:
            col = f"fill_{crit}_{hor}"
            y_all = df_s[col].to_numpy().astype(float)

            # Isotonic: P(fill) should be non-increasing in queue_frac.
            iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
            iso.fit(frac_all, y_all)

            # Sample at 100 points for storage / plotting.
            x_pts = np.linspace(0.0, 1.0, 101)
            y_pts = iso.predict(x_pts)

            key = f"{side_str}_{crit}_{hor}"
            model_params[key] = {
                "x": x_pts.tolist(),
                "y": y_pts.tolist(),
            }
            iso_fits[(side_str, crit, hor)] = (x_pts, y_pts)

OUT_MODEL.write_text(json.dumps(model_params, indent=2))
print(f"Saved model -> {OUT_MODEL}")

# ── 3. FIG-4a: empirical fill curves (any-decrease criterion, both sides) ────
COLORS = {"1s": "#e41a1c", "10s": "#377eb8", "60s": "#4daf4a"}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("P2.2 FIG-4a — Fill probability vs normalized queue position (any-decrease)")

for ax, (side_int, side_str) in zip(axes, SIDES.items()):
    df_s = df.filter(pl.col("side") == side_int)
    curves = empirical_curves(df_s, "any")
    for hor in HORIZONS:
        means, rates, counts = curves[hor]
        ok = ~np.isnan(rates)
        ax.errorbar(means[ok], rates[ok], fmt="o", ms=5, color=COLORS[hor],
                    label=f"empirical τ={hor}")
        x_fit, y_fit = iso_fits[(side_str, "any", hor)]
        ax.plot(x_fit, y_fit, "-", color=COLORS[hor], lw=1.5,
                label=f"isotonic τ={hor}")
    ax.set_xlabel("normalized queue fraction (0=front, 1=back)")
    ax.set_ylabel("P(fill)")
    ax.set_title(f"{side_str.capitalize()} passive orders")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(OUT_FIG4A, dpi=120, bbox_inches="tight")
print(f"Saved FIG-4a -> {OUT_FIG4A}")

# ── 4. FIG-4b: reliability diagram on held-out half ─────────────────────────
# Split: first half of day calibration, second half evaluation.
ts = df["ts_placed"].to_numpy()
midpoint = (ts.min() + ts.max()) // 2
df_cal = df.filter(pl.col("ts_placed") < midpoint)
df_eval = df.filter(pl.col("ts_placed") >= midpoint)

# Refit on calibration half only.
iso_cal = {}
for side_int, side_str in SIDES.items():
    df_c = df_cal.filter(pl.col("side") == side_int)
    df_e = df_eval.filter(pl.col("side") == side_int)
    for crit in ["any"]:
        for hor in HORIZONS:
            col = f"fill_{crit}_{hor}"
            iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
            iso.fit(df_c["queue_frac"].to_numpy(),
                    df_c[col].to_numpy().astype(float))
            iso_cal[(side_str, crit, hor)] = iso

# Reliability diagram: bin by predicted probability, compute realized rate.
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("P2.2 FIG-4b — Out-of-sample reliability (second half of day)")
N_DECILES = 10

for ax2, (side_int, side_str) in zip(axes2, SIDES.items()):
    df_e = df_eval.filter(pl.col("side") == side_int)
    frac_e = df_e["queue_frac"].to_numpy()
    eces = {}
    for hor in HORIZONS:
        iso = iso_cal[(side_str, "any", hor)]
        pred = iso.predict(frac_e)
        real = df_e[f"fill_any_{hor}"].to_numpy().astype(float)
        # Bin by pred into deciles.
        quantile_edges = np.percentile(pred, np.linspace(0, 100, N_DECILES + 1))
        bin_idx = np.digitize(pred, quantile_edges) - 1
        bin_idx = np.clip(bin_idx, 0, N_DECILES - 1)
        bin_pred, bin_real, bin_cnt = [], [], []
        for b in range(N_DECILES):
            mask = bin_idx == b
            if mask.sum() > 0:
                bin_pred.append(pred[mask].mean())
                bin_real.append(real[mask].mean())
                bin_cnt.append(mask.sum())
        bin_pred = np.array(bin_pred)
        bin_real = np.array(bin_real)
        bin_cnt = np.array(bin_cnt)
        ece = (np.abs(bin_pred - bin_real) * bin_cnt / bin_cnt.sum()).sum()
        eces[hor] = ece
        ax2.scatter(bin_pred, bin_real, s=np.sqrt(bin_cnt) * 3,
                    color=COLORS[hor], label=f"τ={hor} ECE={ece:.3f}", zorder=3)

    ax2.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect calibration")
    ax2.set_xlabel("predicted P(fill)")
    ax2.set_ylabel("realized fill rate")
    ax2.set_title(f"{side_str.capitalize()} — out-of-sample (point area ∝ bin count)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)

fig2.tight_layout()
fig2.savefig(OUT_FIG4B, dpi=120, bbox_inches="tight")
print(f"Saved FIG-4b -> {OUT_FIG4B}")

# ── 5. Summary stats ─────────────────────────────────────────────────────────
print("\n── Fill rate summary (any-decrease criterion, all orders) ──")
for side_int, side_str in SIDES.items():
    df_s = df.filter(pl.col("side") == side_int)
    n = df_s.height
    for hor in HORIZONS:
        rate = df_s[f"fill_any_{hor}"].mean()
        rate_trade = df_s[f"fill_trade_{hor}"].mean()
        print(f"  {side_str} τ={hor:3s}: any={rate:.3f}  trade-only={rate_trade:.3f}  n={n}")

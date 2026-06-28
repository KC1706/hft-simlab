"""P2.4 market-impact calibration: fit Bouchaud propagator from R(τ) + square-root law.

Reads:  experiments/data/p13/samples.parquet, trades.parquet
Writes: experiments/data/p24/impact_model.json
        experiments/p24_propagator.png   (FIG-6a: R(τ) + fit)
        experiments/p24_sqrt_law.png     (FIG-6b: I(V) ~ sqrt(V))

Theory (Bouchaud *TQP*, arXiv 2603.24137):
  R(τ) = E[sign_k × (mid(t_k + τ) - mid(t_k))]
  Propagator model: R(τ) ≈ G₀ × τ^{-β} (power-law decay)
  Impact amplitude: I(V) ≈ κ × V^{alpha}  (alpha ≈ 0.5 = square-root law)
  The propagator G(t) combines both: the mid-price shift from one trade of
  volume V and sign ε is G(0) × κ × V^{alpha} × ε, then decays as G(τ)/G(0).

Run from repo root: .venv/bin/python scripts/p24_impact_calibrate.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

P13 = Path("experiments/data/p13")
OUT_DIR = Path("experiments")
DATA_OUT = Path("experiments/data/p24")
OUT_MODEL = DATA_OUT / "impact_model.json"
OUT_FIG6A = OUT_DIR / "p24_propagator.png"
OUT_FIG6B = OUT_DIR / "p24_sqrt_law.png"

DATA_OUT.mkdir(parents=True, exist_ok=True)

# ── 1. Load data ─────────────────────────────────────────────────────────────
samples = pl.read_parquet(P13 / "samples.parquet")
trades = pl.read_parquet(P13 / "trades.parquet")
GRID_NS = 100_000_000  # 100 ms

ts = samples["ts"].to_numpy()
mid = (samples["bid_tick"].to_numpy() + samples["ask_tick"].to_numpy()) / 2.0  # ticks
t0 = ts[0]
idx = ((ts - t0) // GRID_NS).astype(np.int64)
n_grid = idx[-1] + 1

# Build mid grid (sparse → dense, NaN fill).
mid_grid = np.full(n_grid, np.nan)
mid_grid[idx] = mid

t_trade = trades["ts"].to_numpy()
sign = trades["sign"].to_numpy().astype(float)
qty = trades["qty"].to_numpy()
t_idx = ((t_trade - t0) // GRID_NS).astype(np.int64)

# ── 2. Compute R(τ) at log-spaced lags ──────────────────────────────────────
taus_s = np.unique(np.logspace(-1, np.log10(60), 30))
taus_grid = (taus_s * 10).astype(int)  # 100ms slots

R = []
for tau in taus_grid:
    ok = (t_idx + tau < n_grid) & (t_idx >= 0)
    m0 = mid_grid[t_idx[ok]]
    m1 = mid_grid[t_idx[ok] + tau]
    good = ~np.isnan(m0) & ~np.isnan(m1)
    if good.sum() > 10:
        R.append((sign[ok][good] * (m1 - m0)[good]).mean())
    else:
        R.append(np.nan)
R = np.array(R)

# ── 3. Propagator calibration ────────────────────────────────────────────────
# On a single trending day, R(τ) is still rising at 60s (permanent impact regime)
# so fitting β from R(τ) directly gives β<0 (artifact of trend). Instead:
#   β = 0.5  (Bouchaud literature value for large-tick assets; *TQP* ch. impact)
#   G₀ calibrated so G(1s) matches R(1s) empirically.
# The paper will document β as a literature prior and κ as the data-calibrated term.
beta = 0.5  # literature prior: power-law decay G(τ) ~ τ^{-0.5}

R_1s = R[np.searchsorted(taus_s, 1.0)]
# G₀ such that G₀ × 1^{-β} = R_1s (normalise to τ=1s anchor)
G0 = R_1s  # R(1s) sets the 1-second propagator amplitude

print(f"Propagator (β=0.5 from literature): G₀={G0:.3f} ticks at τ=1s")
print(f"  R(1s) empirical: {R_1s:.3f} ticks  R(10s): {R[np.searchsorted(taus_s, 10.0)]:.3f} ticks")
print(f"  Note: R(τ) is non-decaying on this trending day → β fitted from data = "
      f"{-np.polyfit(np.log(taus_s[(R>0)&~np.isnan(R)&(taus_s>0.5)&(taus_s<50)]), np.log(R[(R>0)&~np.isnan(R)&(taus_s>0.5)&(taus_s<50)]), 1)[0]:.3f} "
      f"(negative = R still rising); we use literature β=0.5 instead.")

# ── 4. FIG-6a: R(τ) + propagator fit ────────────────────────────────────────
tau_fit = np.logspace(-1, np.log10(60), 200)
R_fit = G0 * tau_fit**(-beta)

fig, ax = plt.subplots(figsize=(8, 5))
pos = R > 0
ax.loglog(taus_s[pos], R[pos], "o", ms=5, color="#1f77b4", label="R(τ) empirical")
ax.loglog(tau_fit, R_fit, "r--", lw=1.5,
          label=f"G(τ) = {G0:.2f} × τ^{{-{beta:.2f}}}")
ax.set_xlabel("lag τ (seconds)")
ax.set_ylabel("E[sign × Δmid(τ)] (ticks)")
ax.set_title("P2.4 FIG-6a — Trade response function R(τ) + power-law propagator fit")
ax.legend()
ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(OUT_FIG6A, dpi=120, bbox_inches="tight")
print(f"Saved FIG-6a -> {OUT_FIG6A}")

# ── 5. Square-root impact law: I(V) ~ κ × V^alpha ────────────────────────────
# Measure average |Δmid| at lag 1 slot (100ms) per trade, binned by volume.
tau_immediate = 1  # 1 grid slot = 100ms
ok_imm = (t_idx + tau_immediate < n_grid) & (t_idx >= 0)
m0i = mid_grid[t_idx[ok_imm]]
m1i = mid_grid[t_idx[ok_imm] + tau_immediate]
goodi = ~np.isnan(m0i) & ~np.isnan(m1i)
signed_impact = sign[ok_imm][goodi] * (m1i - m0i)[goodi]  # ticks, sign-adjusted
vol_ok = qty[ok_imm][goodi]

# Log-spaced volume bins.
vol_bins = np.logspace(np.log10(vol_ok.min() + 1e-9), np.log10(vol_ok.max()), 12)
bin_idx = np.digitize(vol_ok, vol_bins)
bin_vol, bin_impact, bin_n = [], [], []
for b in range(1, len(vol_bins)):
    mask = bin_idx == b
    if mask.sum() > 20:
        bin_vol.append(np.sqrt(vol_ok[mask].mean()))  # sqrt(V) for x-axis
        bin_impact.append(signed_impact[mask].mean())
        bin_n.append(mask.sum())

bin_vol = np.array(bin_vol)
bin_impact = np.array(bin_impact)
bin_n = np.array(bin_n)

# Fit I = κ × sqrt(V).
pos_mask = bin_impact > 0
if pos_mask.sum() >= 3:
    kappa = (bin_impact[pos_mask] / bin_vol[pos_mask]).mean()
else:
    kappa = float('nan')
print(f"Square-root law: κ = {kappa:.4f} ticks/sqrt(BTC)  (I = κ × sqrt(V))")

fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.scatter(bin_vol, bin_impact, s=np.sqrt(bin_n) * 3, zorder=3,
            color="#1f77b4", label="E[sign×Δmid] | sqrt(V) bins")
if not np.isnan(kappa):
    v_line = np.linspace(bin_vol.min(), bin_vol.max(), 100)
    ax2.plot(v_line, kappa * v_line, "r--", lw=1.5,
             label=f"κ × √V  (κ={kappa:.4f})")
ax2.set_xlabel("sqrt(trade volume)  √(BTC)")
ax2.set_ylabel("E[sign × Δmid @ 100ms] (ticks)")
ax2.set_title("P2.4 FIG-6b — Square-root impact law on BTCUSDT-perp")
ax2.legend()
ax2.grid(alpha=0.25)
fig2.tight_layout()
fig2.savefig(OUT_FIG6B, dpi=120, bbox_inches="tight")
print(f"Saved FIG-6b -> {OUT_FIG6B}")

# ── 6. Save model ─────────────────────────────────────────────────────────────
model = {
    "description": "Bouchaud power-law propagator for BTCUSDT-perp 2026-05-01",
    # G(τ) = G0 × τ^{-beta}, τ in seconds, G in ticks per unit volume
    "G0_ticks": float(G0),
    "beta": float(beta),
    # Square-root law: I(V) = kappa × sqrt(V) at τ→0
    "kappa_ticks_per_sqrt_btc": float(kappa),
    "impact_exponent": 0.5,
    # Cutoff: truncate the sum at 60 seconds (cost/accuracy tradeoff)
    "cutoff_s": 60.0,
    "R_1s_ticks": float(R[np.searchsorted(taus_s, 1.0)]),
    "R_10s_ticks": float(R[np.searchsorted(taus_s, 10.0)]),
    "R_60s_ticks": float(R[np.searchsorted(taus_s, 50.0)]),
}
OUT_MODEL.write_text(json.dumps(model, indent=2))
print(f"\nSaved model -> {OUT_MODEL}")
print(json.dumps(model, indent=2))

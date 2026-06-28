"""P1.3 step 2: stylized-facts panel + OFI regression from the logged parquet.

Renders FIG-10 (see paper/FIGURES.md): six panels of the dataset's microstructure
fingerprint, plus the OFI->return regression printed and saved as JSON. Pure
rendering — no recomputation of raw data; rerunning is always safe.
Run from repo root: .venv/bin/python experiments/figures/p13_stylized_facts.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

DATA_DIR = Path("experiments/data/p13")
OUT_PNG = Path("experiments/p13_stylized_facts.png")
OUT_STATS = DATA_DIR / "stats.json"
GRID_NS = 100_000_000  # 100 ms

samples = pl.read_parquet(DATA_DIR / "samples.parquet")
trades = pl.read_parquet(DATA_DIR / "trades.parquet")

ts = samples["ts"].to_numpy()
mid = (samples["bid_tick"].to_numpy() + samples["ask_tick"].to_numpy()) / 2.0  # in ticks
spread = samples["spread_ticks"].to_numpy()
ofi = samples["ofi"].to_numpy()

# Build a contiguous 100ms grid index (a few rows at session start are skipped).
idx = ((ts - ts[0]) // GRID_NS).astype(np.int64)
n_grid = idx[-1] + 1
mid_grid = np.full(n_grid, np.nan)
mid_grid[idx] = mid

# 1-second series for returns/OFI regression (every 10th grid slot).
mid_1s = mid_grid[::10]
valid = ~np.isnan(mid_1s)
r_1s = np.diff(np.log(mid_1s))  # log returns in tick space == relative moves
r_1s = r_1s[~np.isnan(r_1s)]

ofi_grid = np.zeros(n_grid)
ofi_grid[idx] = ofi
# Sample row at slot j covers (j-1, j]; the return mid_1s[i+1]-mid_1s[i] covers
# slots 10i+1 .. 10i+10 — so OFI groups must start at slot 1, not 0 (a one-slot
# misalignment here drops the regression R^2 from ~0.5 to ~0.01).
dmid_1s = np.diff(mid_1s)  # in ticks
ofi_1s = np.add.reduceat(ofi_grid, np.arange(1, n_grid, 10))[: len(dmid_1s)]
ok = ~np.isnan(dmid_1s)
X, Y = ofi_1s[ok], dmid_1s[ok]
beta = (X * Y).sum() / (X * X).sum()
r2 = 1 - ((Y - beta * X) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()

def acf(x, lags):
    x = x - x.mean()
    v = (x * x).sum()
    return np.array([(x[: len(x) - l] * x[l:]).sum() / v for l in lags])

fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
fig.suptitle("BTCUSDT-perp 2026-05-01 00:00–06:56 — stylized facts (P1.3 / FIG-10)")

# (a) spread distribution (time-weighted, 100ms sampling)
ax = axes[0, 0]
smax = 12
counts = np.bincount(np.clip(spread, 0, smax), minlength=smax + 1)[1:]
ax.bar(range(1, smax + 1), counts / counts.sum(), color="#1f77b4")
ax.set_yscale("log")
ax.set_xlabel("spread (ticks)")
ax.set_ylabel("fraction of time")
ax.set_title(f"(a) spread distribution  P(1 tick)={counts[0]/counts.sum():.2f}")

# (b) average depth profile, top 10 levels
ax = axes[0, 1]
bq = np.array([samples[f"bq{i}"].mean() for i in range(1, 11)])
aq = np.array([samples[f"aq{i}"].mean() for i in range(1, 11)])
lv = np.arange(1, 11)
ax.bar(lv - 0.2, bq, width=0.4, label="bid", color="#2ca02c")
ax.bar(lv + 0.2, aq, width=0.4, label="ask", color="#d62728")
ax.set_xlabel("level (1 = best)")
ax.set_ylabel("mean resting qty (BTC)")
ax.set_title("(b) average depth profile")
ax.legend()

# (c) 1s return distribution vs gaussian (fat tails)
ax = axes[0, 2]
sd = r_1s.std()
z = r_1s / sd
kurt = ((z**4).mean()) - 3.0
bins = np.linspace(-12, 12, 121)
hist, edges = np.histogram(z, bins=bins, density=True)
centers = (edges[:-1] + edges[1:]) / 2
ax.semilogy(centers, hist, "o", ms=3, label="1s returns (std.)")
g = np.exp(-(centers**2) / 2) / np.sqrt(2 * np.pi)
ax.semilogy(centers, g, "--", color="grey", label="N(0,1)")
ax.set_ylim(1e-6, 1)
ax.set_xlabel("standardized return")
ax.set_title(f"(c) fat tails  excess kurtosis={kurt:.1f}")
ax.legend()

# (d) ACF of returns and |returns| (vol clustering)
ax = axes[1, 0]
lags_s = np.arange(1, 301)
ax.plot(lags_s, acf(r_1s, lags_s), label="returns", lw=1)
ax.plot(lags_s, acf(np.abs(r_1s), lags_s), label="|returns|", lw=1)
ax.axhline(0, color="grey", lw=0.5)
ax.set_xlabel("lag (s)")
ax.set_ylabel("autocorrelation")
ax.set_title("(d) no return memory, slow |r| decay")
ax.legend()

# (e) ACF of trade signs, log-log (long memory of order flow)
ax = axes[1, 1]
signs = trades["sign"].to_numpy().astype(float)
lags_t = np.unique(np.logspace(0, 4, 40).astype(int))
acf_t = acf(signs, lags_t)
pos = acf_t > 0
ax.loglog(lags_t[pos], acf_t[pos], "o-", ms=3)
# power-law fit over 10..3000 trades
m = (lags_t >= 10) & (lags_t <= 3000) & pos
gamma, logc = np.polyfit(np.log(lags_t[m]), np.log(acf_t[m]), 1)
ax.loglog(lags_t[m], np.exp(logc) * lags_t[m] ** gamma, "--", color="grey",
          label=f"slope ≈ {gamma:.2f}")
ax.set_xlabel("lag (trades)")
ax.set_ylabel("sign autocorrelation")
ax.set_title("(e) long-memory order flow")
ax.legend()

# (f) trade signature: response R(tau) in ticks
ax = axes[1, 2]
t_idx = ((trades["ts"].to_numpy() - ts[0]) // GRID_NS).astype(np.int64)
t_sign = trades["sign"].to_numpy().astype(float)
taus_s = np.unique(np.logspace(-1, np.log10(60), 25))
taus = (taus_s * 10).astype(int)  # in 100ms slots
R = []
for tau in taus:
    ok2 = (t_idx + tau < n_grid) & (t_idx >= 0)
    m0, m1 = mid_grid[t_idx[ok2]], mid_grid[t_idx[ok2] + tau]
    good = ~np.isnan(m0) & ~np.isnan(m1)
    R.append((t_sign[ok2][good] * (m1 - m0)[good]).mean())
R = np.array(R)
ax.semilogx(taus / 10.0, R, "o-", ms=3)
ax.set_xlabel("lag after trade (s)")
ax.set_ylabel("E[sign × Δmid] (ticks)")
ax.set_title("(f) response function R(τ)")

for a in axes.flat:
    a.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=120, bbox_inches="tight")

stats = {
    "p_spread_1tick": float(counts[0] / counts.sum()),
    "excess_kurtosis_1s": float(kurt),
    "return_acf_lag1": float(acf(r_1s, [1])[0]),
    "abs_return_acf_lag60": float(acf(np.abs(r_1s), [60])[0]),
    "sign_acf_powerlaw_gamma": float(gamma),
    "ofi_beta_ticks_per_unit": float(beta),
    "ofi_r2_1s": float(r2),
    "response_1s_ticks": float(R[np.searchsorted(taus_s, 1.0)]),
    "n_trades": int(trades.height),
    "n_1s_returns": int(len(r_1s)),
}
OUT_STATS.write_text(json.dumps(stats, indent=2))
print(json.dumps(stats, indent=2))
print(f"saved {OUT_PNG}")

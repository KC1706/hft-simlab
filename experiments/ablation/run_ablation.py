"""P4 step 1 — the sim-to-real ablation harness (in-sample, single-day).

Sweeps (strategy x realism-config x seed) by driving the Rust `backtest --csv` binary, collects a
tidy parquet, and reports the realism WATERFALL: how PnL shifts as each component (fill -> latency
-> impact) is switched on, per strategy, with seed dispersion. See docs/PHASE4_ABLATION_DESIGN.md.

Strategies here are parametric fixed-offset MMs distinguished by half-spread (ticks) — a first,
data-free ablation subject; the OFI-momentum taker and the +generative config are follow-ups.

Ground truth (the true sim-to-real GAP) needs a HELD-OUT second day (we have one); until then this
runs in-sample to build/validate the harness and quantify the realism effects. When a 2nd day
exists, point --real-day at it and the gap columns populate.

  python experiments/ablation/run_ablation.py --npz data/btcusdt_20260501_0000_0656.npz
"""
import argparse
import io
import subprocess
from pathlib import Path

import numpy as np
import polars as pl

BIN = Path("core/target/release/backtest")
CONFIG_ORDER = ["naive", "+fill", "+fill+latency", "+fill+latency+impact"]


def one_run(npz, strategy, half_spread, seed):
    out = subprocess.run(
        [str(BIN), npz, "--csv", "--strategy", strategy,
         "--half-spread", str(half_spread), "--seed", str(seed)],
        capture_output=True, text=True, check=True,
    ).stdout
    return pl.read_csv(io.StringIO(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="the day to backtest")
    ap.add_argument("--half-spreads", default="20,40,60,80", help="MM strategy variants (ticks)")
    ap.add_argument("--seeds", type=int, default=8, help="seeds per cell (PnL dispersion)")
    ap.add_argument("--out", default="experiments/ablation/results.parquet")
    args = ap.parse_args()

    if not BIN.exists():
        raise SystemExit(f"{BIN} not found — build it: (cd core && cargo build --release --bin backtest)")

    # subjects: MM at several half-spreads + the OFI taker (half_spread nominal; taker ignores it).
    mm_hs = [float(x) for x in args.half_spreads.split(",")]
    subjects = [("mm", hs) for hs in mm_hs] + [("ofi", 60.0)]
    frames = []
    for strat, hs in subjects:
        for seed in range(1, args.seeds + 1):
            frames.append(one_run(args.npz, strat, hs, seed))
    df = pl.concat(frames)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.out)
    print(f"[ablation] ran {len(subjects)} subjects x {args.seeds} seeds x "
          f"{len(CONFIG_ORDER)} configs = {len(df)} rows -> {args.out}\n")

    # --- realism waterfall: mean PnL +/- std per (subject, config), across seeds ---
    agg = (df.group_by(["strategy", "half_spread", "config"])
             .agg(pl.col("pnl").mean().alias("pnl_mean"),
                  pl.col("pnl").std().alias("pnl_std"),
                  pl.col("fills").mean().alias("fills_mean")))
    print(f"{'subject':>13} | {'config':>22} | {'PnL mean':>10} | {'PnL std':>8} | {'Δ vs prev':>9} | {'fills':>7}")
    print("-" * 90)
    for strat, hs in subjects:
        label = f"ofi-taker" if strat == "ofi" else f"mm(½={hs:.0f})"
        prev = None
        for cfg in CONFIG_ORDER:
            row = agg.filter((pl.col("strategy") == strat) & (pl.col("half_spread") == hs)
                             & (pl.col("config") == cfg))
            if row.is_empty():
                continue
            m = row["pnl_mean"][0]; s = row["pnl_std"][0] or 0.0; f = row["fills_mean"][0]
            delta = "" if prev is None else f"{m - prev:+.3f}"
            print(f"{label:>13} | {cfg:>22} | {m:>10.4f} | {s:>8.4f} | {delta:>9} | {f:>7.0f}")
            prev = m
        print()

    # --- the cross-strategy story: marginal effect of each component, PER strategy type ---
    print("marginal realism effect (mean ΔPnL when each component switches on), by strategy type:")
    for strat in ("mm", "ofi"):
        sub = df.filter(pl.col("strategy") == strat)
        means = {cfg: sub.filter(pl.col("config") == cfg)["pnl"].mean() for cfg in CONFIG_ORDER}
        parts = [f"{b.split('+')[-1]}: {means[b] - means[a]:+.3f}"
                 for a, b in zip(CONFIG_ORDER, CONFIG_ORDER[1:])]
        print(f"  {strat:>4}:  " + "   ".join(parts) + "  USDT")
    print("  -> the fill/queue model matters for the MM but is ~0 for the taker (takers always fill);")
    print("     impact affects both. WHICH realism component matters depends on the strategy.")
    print("\n(NOTE: in-sample only. The true sim-to-real GAP needs a held-out 2nd day — see the design doc.)")


if __name__ == "__main__":
    main()

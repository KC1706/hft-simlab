"""P1.2 parity test: our Rust book vs hftbacktest's reconstruction, level for level.

PLAN.md P1 manual test: pick 3 random timestamps; the top-5 levels of our book
must EXACTLY match hftbacktest's HashMapMarketDepth at the same local timestamps.
"Exactly" = integer price ticks equal, quantities equal to float round-trip.

Both sides define a snapshot at T identically: the book after applying every
LOCAL-flagged event with local_ts <= T. Ours comes from `core` (replay binary,
--snapshots); theirs from elapsing a no-op hftbacktest backtest to T and walking
the depth accessors. Run from repo root:  .venv/bin/python scripts/p12_parity.py
"""
import json
import random
import subprocess
import sys
import time

import numpy as np
from numba import njit

from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest

DATA = "data/btcusdt_20260501_0000_0656.npz"
REPLAY = "core/target/release/replay"
TICK, LOT = 0.1, 0.001
DEPTH_N = 5
SEED = 20260501

# ---- 1. our side: full-pass for the ts range, then snapshots at 3 random ts ----
t0 = time.perf_counter()
subprocess.run([REPLAY, DATA, "--json", "/tmp/p12_ours_full.json"], check=True,
               capture_output=True)
ours_full_sec = time.perf_counter() - t0
full = json.load(open("/tmp/p12_ours_full.json"))
first_ts, last_ts = full["first_local_ts"], full["last_local_ts"]
data_start = full["data_start_ts"]  # hftbacktest's elapse() origin

rng = random.Random(SEED)
ts_list = sorted(rng.randrange(first_ts + 600_000_000_000, last_ts - 60_000_000_000)
                 for _ in range(3))
print(f"chosen local timestamps (seed {SEED}):")
for t in ts_list:
    print(f"  {t}  (+{(t - first_ts) / 60e9:.1f} min into session)")

subprocess.run(
    [REPLAY, DATA, "--snapshots", ",".join(map(str, ts_list)),
     "--depth", str(DEPTH_N), "--json", "/tmp/p12_ours.json"],
    check=True, capture_output=True)
ours = json.load(open("/tmp/p12_ours.json"))["snapshots"]

# ---- 2. their side: elapse a no-op backtest to each ts, walk the depth ----
I64_UNSET = 9223372036854775807  # fresh backtest's current_timestamp (i64::MAX)

@njit
def snap_at(hbt, ts_arr, data_start, bid_ticks, bid_qty, ask_ticks, ask_qty):
    asset_no = 0
    for k in range(len(ts_arr)):
        cur = hbt.current_timestamp
        base = data_start if cur == I64_UNSET else cur
        dur = ts_arr[k] - base
        if dur > 0:
            hbt.elapse(dur)
        depth = hbt.depth(asset_no)
        # top-N bids: walk down from the best pointer over live levels
        n, t = 0, depth.best_bid_tick
        while n < DEPTH_N and t > depth.best_bid_tick - 100_000:
            q = depth.bid_qty_at_tick(t)
            if q > 0:
                bid_ticks[k, n] = t
                bid_qty[k, n] = q
                n += 1
            t -= 1
        n, t = 0, depth.best_ask_tick
        while n < DEPTH_N and t < depth.best_ask_tick + 100_000:
            q = depth.ask_qty_at_tick(t)
            if q > 0:
                ask_ticks[k, n] = t
                ask_qty[k, n] = q
                n += 1
            t += 1
    return True

asset = (
    BacktestAsset()
    .data([DATA])
    .linear_asset(1.0)
    .constant_order_latency(10_000_000, 10_000_000)
    .risk_adverse_queue_model()
    .no_partial_fill_exchange()
    .trading_value_fee_model(0.0002, 0.0005)
    .tick_size(TICK)
    .lot_size(LOT)
)
t0 = time.perf_counter()
hbt = HashMapMarketDepthBacktest([asset])
theirs_load_sec = time.perf_counter() - t0

ts_arr = np.array(ts_list, dtype=np.int64)
bid_ticks = np.full((3, DEPTH_N), -1, dtype=np.int64)
ask_ticks = np.full((3, DEPTH_N), -1, dtype=np.int64)
bid_qty = np.zeros((3, DEPTH_N))
ask_qty = np.zeros((3, DEPTH_N))
t0 = time.perf_counter()
snap_at(hbt, ts_arr, data_start, bid_ticks, bid_qty, ask_ticks, ask_qty)
theirs_elapse_sec = time.perf_counter() - t0
hbt.close()

# ---- 3. compare ----
def fail(msg):
    print("PARITY FAIL:", msg)
    sys.exit(1)

for k, t in enumerate(ts_list):
    snap = ours[k]
    assert snap["ts"] == t
    for side, their_ticks, their_qty in (
        ("bids", bid_ticks, bid_qty), ("asks", ask_ticks, ask_qty)):
        our_levels = snap[side]
        if len(our_levels) != DEPTH_N or (their_ticks[k] < 0).any():
            fail(f"ts {t}: {side} has fewer than {DEPTH_N} levels")
        for i in range(DEPTH_N):
            o_tick, _, o_qty = our_levels[i]
            if o_tick != their_ticks[k, i]:
                fail(f"ts {t} {side}[{i}]: tick {o_tick} != {their_ticks[k, i]}")
            if abs(o_qty - their_qty[k, i]) > 1e-12:
                fail(f"ts {t} {side}[{i}]: qty {o_qty} != {their_qty[k, i]}")
    bb, ba = snap["bids"][0], snap["asks"][0]
    print(f"  ts {t}: top5 MATCH  (best bid {bb[1]} x {bb[2]}, best ask {ba[1]} x {ba[2]})")

print("\nPARITY OK: all top-5 levels identical at all 3 timestamps")
print(f"\ntiming: ours full pass {ours_full_sec:.2f}s | "
      f"hftbacktest load {theirs_load_sec:.2f}s + elapse-to-end {theirs_elapse_sec:.2f}s")

"""P1.2 benchmark: our streaming replay vs hftbacktest processing the same day.

Method notes (honesty box):
- Ours streams + decompresses + applies LOCAL book events in one pass.
- hftbacktest decompresses at construction ("load"), then `elapse` runs BOTH
  processors (local + exchange) with latency queues and order machinery (idle).
  More work per event than ours — this is a sanity-scale comparison, not a race.
- numba JIT compile is excluded by a 1 ns warm-up elapse before timing.
Run from repo root: .venv/bin/python scripts/p12_bench.py
"""
import subprocess
import time

from numba import njit

from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest

DATA = "data/btcusdt_20260501_0000_0656.npz"
REPLAY = "core/target/release/replay"
N_EVENTS = 26_663_697
SESSION_NS = 26_000_000_000_000  # > 416.4 min, runs to end of data


@njit(cache=True)
def run_to_end(hbt, duration):
    hbt.elapse(1)  # warm-up: triggers JIT before the timed call in caller
    return 0


@njit(cache=True)
def elapse_all(hbt, duration):
    return hbt.elapse(duration)


# ours
t0 = time.perf_counter()
subprocess.run([REPLAY, DATA], check=True, capture_output=True)
ours = time.perf_counter() - t0

# theirs
asset = (BacktestAsset().data([DATA]).linear_asset(1.0)
         .constant_order_latency(10_000_000, 10_000_000)
         .risk_adverse_queue_model().no_partial_fill_exchange()
         .trading_value_fee_model(0.0002, 0.0005).tick_size(0.1).lot_size(0.001))
t0 = time.perf_counter()
hbt = HashMapMarketDepthBacktest([asset])
load = time.perf_counter() - t0
run_to_end(hbt, 1)  # JIT warm-up
t0 = time.perf_counter()
elapse_all(hbt, SESSION_NS)
elapse = time.perf_counter() - t0
hbt.close()

print(f"ours  (stream+decompress+book):       {ours:6.2f}s  {N_EVENTS/ours/1e6:5.2f}M ev/s")
print(f"hftbacktest load (decompress to RAM): {load:6.2f}s")
print(f"hftbacktest elapse to end (no orders):{elapse:6.2f}s  {N_EVENTS/elapse/1e6:5.2f}M ev/s")
print(f"hftbacktest end-to-end:               {load+elapse:6.2f}s  {N_EVENTS/(load+elapse)/1e6:5.2f}M ev/s")

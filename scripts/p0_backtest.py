"""P0.3: First end-to-end backtest — naive baseline market maker.

Strategy: quote one level each side at mid +/- half_spread, skewed by inventory.
This is deliberately the NAIVE BASELINE configuration of the whole project:
constant 10ms latency + risk-adverse queue model. Phase 2 replaces both with
calibrated models; Phase 4 measures how much the answers change.

Loop pattern follows refs/hftbacktest/examples/High-Frequency Grid Trading.ipynb:
order id = price-in-ticks, so requoting the same price is a no-op instead of an
order-id collision, and stale quotes are cancelled by set-difference.
"""
import numpy as np
from numba import njit, uint64
from numba.typed import Dict

from hftbacktest import (
    BacktestAsset, HashMapMarketDepthBacktest, Recorder, BUY, SELL, GTX, LIMIT,
)
from hftbacktest.stats import LinearAssetRecord

DATA = "data/btcusdt_20260501_0000_0656.npz"


@njit
def market_maker(hbt, recorder):
    asset_no = 0
    tick = hbt.depth(asset_no).tick_size
    half_spread = 60.0 * tick          # quote distance from mid
    skew_ticks = 10.0                  # quote shift per unit of max inventory
    order_qty = 0.001                  # BTC
    max_position = 0.005               # BTC, inventory cap

    while hbt.elapse(100_000_000) == 0:        # act every 100ms
        hbt.clear_inactive_orders(asset_no)
        depth = hbt.depth(asset_no)
        position = hbt.position(asset_no)
        orders = hbt.orders(asset_no)

        mid = (depth.best_bid + depth.best_ask) / 2.0
        # Inventory skew: long inventory pushes both quotes down (eager to sell,
        # reluctant to buy) and vice versa — the simplest Avellaneda-Stoikov idea.
        skew = (position / max_position) * skew_ticks * tick

        bid_px = np.floor((mid - half_spread - skew) / tick) * tick
        ask_px = np.ceil((mid + half_spread - skew) / tick) * tick

        new_bids = Dict.empty(np.uint64, np.float64)
        if position < max_position and np.isfinite(bid_px):
            new_bids[uint64(round(bid_px / tick))] = bid_px
        new_asks = Dict.empty(np.uint64, np.float64)
        if position > -max_position and np.isfinite(ask_px):
            new_asks[uint64(round(ask_px / tick))] = ask_px

        order_values = orders.values()
        while order_values.has_next():
            order = order_values.get()
            if order.cancellable:
                if (order.side == BUY and order.order_id not in new_bids) or (
                    order.side == SELL and order.order_id not in new_asks
                ):
                    hbt.cancel(asset_no, order.order_id, False)

        for oid, px in new_bids.items():
            if oid not in orders:
                hbt.submit_buy_order(asset_no, oid, px, order_qty, GTX, LIMIT, False)
        for oid, px in new_asks.items():
            if oid not in orders:
                hbt.submit_sell_order(asset_no, oid, px, order_qty, GTX, LIMIT, False)

        recorder.record(hbt)
    return True


asset = (
    BacktestAsset()
    .data([DATA])
    .linear_asset(1.0)
    .constant_latency(10_000_000, 10_000_000)   # 10ms each way — naive, P2.3 target
    .risk_adverse_queue_model()                  # worst-case queue — P2.2 target
    .no_partial_fill_exchange()
    .trading_value_fee_model(0.0002, 0.0005)     # maker 2bps, taker 5bps
    .tick_size(0.1)
    .lot_size(0.001)
)

hbt = HashMapMarketDepthBacktest([asset])
rec = Recorder(1, 1_000_000)
market_maker(hbt, rec.recorder)
hbt.close()

stats = LinearAssetRecord(rec.get(0)).stats(book_size=10_000)
print(stats.summary())

import matplotlib
matplotlib.use("Agg")
# stats.plot() closes its figure and RETURNS it (refs/.../stats/stats.py:245-247),
# so we must save the returned Figure object, not pyplot's "current figure".
fig = stats.plot()
fig.savefig("experiments/p0_baseline_equity.png", dpi=120, bbox_inches="tight")
print("saved experiments/p0_baseline_equity.png")

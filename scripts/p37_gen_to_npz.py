"""P4 +generative bridge: convert a generated market (p34 denorm CSV) -> the .npz Event stream
the Rust backtest replays. This lets a strategy be backtested against the P3 generative market,
i.e. the `+generative` ablation cell (docs/PHASE4_ABLATION_DESIGN.md).

The generated stream is a sequence of [event_type, size, price(tick), direction] + the top-10 book
(ticks) after each event. We replay it as the exact Event format core/src/npz.rs expects:
  dtype [('ev','<u8'),('exch_ts','<i8'),('local_ts','<i8'),('px','<f8'),('qty','<f8'),
         ('order_id','<u8'),('ival','<i8'),('fval','<f8')], DEFLATE-zipped member data.npy.
Per row we (a) DIFF the book vs the previous snapshot and emit a DEPTH_EVENT (absolute new qty)
for each changed level — that reconstructs the book the strategy quotes into — and (b) for an
execution, emit a TRADE_EVENT (so naive/taker fills trigger). Prices are ticks*0.1; qty in BTC.

  python scripts/p37_gen_to_npz.py <generated.csv> --out /tmp/gen.npz
"""
import argparse
import csv

import numpy as np

TICK_SIZE = 0.1
LEVELS = 10

# event kinds + attribute bits (core/src/events.rs)
DEPTH_EVENT, TRADE_EVENT = 1, 2
BUY_EVENT, SELL_EVENT = 1 << 29, 1 << 28
EXCH_EVENT, LOCAL_EVENT = 1 << 31, 1 << 30
BOTH = EXCH_EVENT | LOCAL_EVENT

EVENT_DTYPE = np.dtype([
    ("ev", "<u8"), ("exch_ts", "<i8"), ("local_ts", "<i8"),
    ("px", "<f8"), ("qty", "<f8"), ("order_id", "<u8"), ("ival", "<i8"), ("fval", "<f8"),
])


def book_of(row):
    """{'ask': {tick: qty}, 'bid': {tick: qty}} from a lob_export-schema CSV row (ticks)."""
    b = {"ask": {}, "bid": {}}
    for l in range(1, LEVELS + 1):
        ap, asz = float(row[f"ask_p{l}"]), float(row[f"ask_s{l}"])
        bp, bsz = float(row[f"bid_p{l}"]), float(row[f"bid_s{l}"])
        if ap > 0 and asz > 0:
            b["ask"][round(ap)] = asz
        if bp > 0 and bsz > 0:
            b["bid"][round(bp)] = bsz
    return b


def convert(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    events = []
    prev = {"ask": {}, "bid": {}}
    for r in rows:
        t_ns = int(round(float(r["time"]) * 1e9))
        cur = book_of(r)
        # (a) emit DEPTH events for every level whose qty changed (0 = level cleared)
        for side, flag in (("bid", BUY_EVENT), ("ask", SELL_EVENT)):
            ticks = set(cur[side]) | set(prev[side])
            for tk in ticks:
                new_q = cur[side].get(tk, 0.0)
                if new_q != prev[side].get(tk, 0.0):
                    events.append((DEPTH_EVENT | flag | BOTH, t_ns, t_ns,
                                   tk * TICK_SIZE, new_q, 0, 0, 0.0))
        # (b) executions -> TRADE_EVENT (lob_export: buy trade carries BUY_EVENT, dir -1)
        if round(float(r["event_type"])) == 2:
            is_buy = float(r["direction"]) < 0
            events.append((TRADE_EVENT | (BUY_EVENT if is_buy else SELL_EVENT) | BOTH,
                           t_ns, t_ns, round(float(r["price"])) * TICK_SIZE, abs(float(r["size"])),
                           0, 0, 0.0))
        prev = cur

    arr = np.array(events, dtype=EVENT_DTYPE)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("generated", help="generated market in lob_export CSV schema (from p34_denorm)")
    ap.add_argument("--out", default="/tmp/gen.npz")
    args = ap.parse_args()
    arr = convert(args.generated)
    # np.savez_compressed writes a DEFLATE member 'data.npy' — exactly what npz.rs decodes.
    np.savez_compressed(args.out, data=arr)
    # np.savez appends .npz if absent
    out = args.out if args.out.endswith(".npz") else args.out + ".npz"
    print(f"[p37] {len(arr)} events -> {out}  ({arr.nbytes/1e6:.1f} MB uncompressed)")
    kinds = arr["ev"] & 0xFF
    print(f"[p37]   depth={int((kinds==DEPTH_EVENT).sum())}  trade={int((kinds==TRADE_EVENT).sum())}  "
          f"span={ (arr['local_ts'].max()-arr['local_ts'].min())/1e9:.2f}s")


if __name__ == "__main__":
    main()

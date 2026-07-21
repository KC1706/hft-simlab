"""P3.4 step 3, Piece 1+2 — a top-10 L2 book + order APPLY for autoregressive generation.

The autoregressive coupling (docs/P34_AUTOREGRESSIVE_DESIGN.md) rolls the book forward with the
model's own orders. This module is the book-update half: a lightweight Python L2 book that mirrors
core/src/book.rs semantics and an APPLY that reproduces lob_export.rs's event->book mapping.

Event semantics (from core/src/bin/lob_export.rs, our L2 synthesis):
  * SUBMISSION   : a level's qty INCREASED  -> qty[side, price] += size
  * CANCELLATION : a level's qty DECREASED  -> qty[side, price] -= size  (drop level at <=0)
  * DELETION     : same as cancellation, level emptied
  * EXECUTION    : a trade PRINT -> the book is NOT modified (the depth change arrives as a
                   separate DEPTH event); execution rows are informational for the flow.
Prices are integer TICKS throughout (tick size 0.1), matching the export. Snapshot layout is the
interleaved 40-col LOBSTER order: [ask_p1, ask_s1, bid_p1, bid_s1, ask_p2, ...].

No torch/GPU deps — importable on Kaggle (by p36) and runnable locally:
  python experiments/kaggle/autoreg_book.py --validate /tmp/rows_log.csv   # replay real transitions
"""
import argparse
import csv

import numpy as np

LEVELS = 10

# raw LOBSTER codes as emitted by lob_export.rs (event_type column in the CSV)
RAW_SUBMISSION, RAW_CANCELLATION, RAW_DELETION, RAW_EXECUTION = 1, 2, 3, 4
# the model, after p31's remap, emits 3 classes: 0=submission, 1=remove(cancel/delete), 2=execution
MODEL_SUBMIT, MODEL_REMOVE, MODEL_EXECUTE = 0, 1, 2


def kind_from_raw(code):
    if code == RAW_SUBMISSION:
        return "submit"
    if code in (RAW_CANCELLATION, RAW_DELETION):
        return "remove"
    if code == RAW_EXECUTION:
        return "execute"
    return None


def kind_from_model(cls):
    return {MODEL_SUBMIT: "submit", MODEL_REMOVE: "remove", MODEL_EXECUTE: "execute"}.get(int(round(cls)))


class L2Book:
    """Top-N L2 book keyed by integer price tick. bids/asks are {tick: qty} with qty > 0."""

    def __init__(self, levels=LEVELS):
        self.levels = levels
        self.bids = {}   # tick -> qty
        self.asks = {}

    # ---- construction / readout -------------------------------------------------------------
    @classmethod
    def from_snapshot(cls, snap, levels=LEVELS):
        """Build from a 40-col interleaved snapshot [ask_p,ask_s,bid_p,bid_s]*levels (ticks)."""
        b = cls(levels)
        snap = np.asarray(snap, dtype=np.float64)
        for l in range(levels):
            ap, asz, bp, bsz = snap[4 * l:4 * l + 4]
            if ap > 0 and asz > 0:
                b.asks[int(round(ap))] = float(asz)
            if bp > 0 and bsz > 0:
                b.bids[int(round(bp))] = float(bsz)
        return b

    def snapshot(self, levels=None):
        """Emit the interleaved 40-col top-N snapshot (ticks), zero-padded, best-first."""
        levels = levels or self.levels
        asks = sorted(self.asks.items())                 # ascending: best ask = lowest
        bids = sorted(self.bids.items(), reverse=True)    # descending: best bid = highest
        out = np.zeros(4 * levels, dtype=np.float64)
        for l in range(levels):
            if l < len(asks):
                out[4 * l + 0], out[4 * l + 1] = asks[l]
            if l < len(bids):
                out[4 * l + 2], out[4 * l + 3] = bids[l]
        return out

    def best_bid_tick(self):
        return max(self.bids) if self.bids else None

    def best_ask_tick(self):
        return min(self.asks) if self.asks else None

    def mid_tick(self):
        bb, ba = self.best_bid_tick(), self.best_ask_tick()
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    # ---- the APPLY --------------------------------------------------------------------------
    def apply(self, kind, size, price_tick, side):
        """Apply one synthesized order. side: +1 = bid, -1 = ask. Returns True if the book changed.

        Marketable-price guard: a submission that would cross (bid >= best ask, or ask <= best bid)
        is clipped to the touch, so the book never crosses (mirrors a real venue rejecting/repricing).
        """
        price_tick = int(round(price_tick))
        book = self.bids if side == 1 else self.asks
        if kind == "execute":
            return False                                  # trades don't move the L2 book (see lob_export.rs)
        if kind == "submit":
            if side == 1 and self.best_ask_tick() is not None:
                price_tick = min(price_tick, self.best_ask_tick() - 1)
            if side == -1 and self.best_bid_tick() is not None:
                price_tick = max(price_tick, self.best_bid_tick() + 1)
            book[price_tick] = book.get(price_tick, 0.0) + size
            return True
        if kind == "remove":
            if price_tick in book:
                book[price_tick] -= size
                if book[price_tick] <= 1e-12:
                    del book[price_tick]
                return True
            return False                                  # removing from an unseen level: no-op
        raise ValueError(f"unknown kind {kind!r}")


def reconstruct_price_tick(depth, direction, book):
    """Generation-time price: the model emits a small relative `depth` (|price-mid| in ticks) and a
    side, which is far more learnable than the absolute tick. price = mid - direction*depth
    (bid below mid, ask above). Falls back to the best touch if the book is one-sided."""
    mid = book.mid_tick()
    if mid is None:
        bb, ba = book.best_bid_tick(), book.best_ask_tick()
        mid = bb if bb is not None else (ba if ba is not None else 0.0)
    return int(round(mid - direction * depth))


# ---- validation: replay REAL transitions and check APPLY reproduces the next real snapshot ------
def validate(csv_path, max_rows=200_000, levels=LEVELS):
    with open(csv_path) as f:
        rd = csv.reader(f)
        header = next(rd)
        idx = {name: i for i, name in enumerate(header)}
        lob_names = [f"{s}{l}" for l in range(1, levels + 1) for s in ("ask_p", "ask_s", "bid_p", "bid_s")]
        lob_idx = [idx[n] for n in lob_names]
        rows = []
        for r in rd:
            rows.append(r)
            if len(rows) > max_rows:
                break

    def snap_of(r):
        return np.array([float(r[i]) for i in lob_idx], dtype=np.float64)

    exact = touched = counted = 0
    per_kind = {"submit": [0, 0], "remove": [0, 0], "execute": [0, 0]}  # [reproduced, total]
    for i in range(len(rows) - 1):
        prev, cur = rows[i], rows[i + 1]
        kind = kind_from_raw(int(float(cur[idx["event_type"]])))
        if kind is None:
            continue
        size = float(cur[idx["size"]])
        price_tick = float(cur[idx["price"]])
        direction = int(float(cur[idx["direction"]]))
        book = L2Book.from_snapshot(snap_of(prev), levels)
        book.apply(kind, size, price_tick, direction)
        got, want = book.snapshot(levels), snap_of(cur)
        counted += 1
        per_kind[kind][1] += 1
        if np.array_equal(got, want):
            exact += 1
            per_kind[kind][0] += 1
        # top-of-book match (robust to hidden-level promotion deeper in the book)
        if np.allclose([got[0], got[1], got[2], got[3]], [want[0], want[1], want[2], want[3]], atol=1e-9):
            touched += 1

    print(f"[validate] transitions checked: {counted}")
    print(f"[validate] EXACT full-top{levels} match:   {100*exact/counted:.1f}%")
    print(f"[validate] top-of-book (L1) match:       {100*touched/counted:.1f}%  "
          f"(full-book misses are hidden-level promotions beyond the {levels}-level horizon)")
    for k, (rep, tot) in per_kind.items():
        if tot:
            print(f"[validate]   {k:8s}: {tot:7d} events, exact {100*rep/tot:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", metavar="EXPORT_CSV", help="replay real transitions from a lob_export CSV")
    ap.add_argument("--max-rows", type=int, default=200_000)
    args = ap.parse_args()
    if args.validate:
        validate(args.validate, args.max_rows)
    else:
        ap.error("nothing to do — pass --validate <export.csv>")


if __name__ == "__main__":
    main()

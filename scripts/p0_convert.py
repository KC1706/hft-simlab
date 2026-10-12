"""P0.2: Convert Tardis CSV slices -> hftbacktest .npz event format.

Key facts encoded here (see docs/JOURNAL.md Entry 1):
- Trades file MUST precede the depth file: a trade both prints AND removes book volume;
  hftbacktest processes the trade first so simulated queue positions aren't decremented twice.
- buffer_size is a preallocated numpy array of 64-byte events; sized to the slice (~26M rows)
  instead of the 100M default, which would not fit in 8 GB RAM.
- Tardis Binance Futures rows carry the exchange 'E' (send) timestamp, not 'T' (match) time,
  so measured feed latency is slightly understated (converter docstring, tardis.py:67).

Usage (Day A, the default, reproduces the original slice):
    .venv/bin/python scripts/p0_convert.py

Day B (held-out second day, Entry 19) — pass the sliced CSVs + output + buffer:
    .venv/bin/python scripts/p0_convert.py \
        --trades data/raw_dayB/trades_sliceB.csv \
        --book   data/raw_dayB/book_sliceB.csv \
        --out    data/btcusdt_20260601_dayB.npz \
        --buffer 27000000
"""
import argparse

from hftbacktest.data.utils import tardis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", default="data/raw/trades_slice.csv",
                    help="trades CSV — passed FIRST (queue double-count prevention)")
    ap.add_argument("--book", default="data/raw/book_slice.csv",
                    help="incremental L2 depth CSV — passed second")
    ap.add_argument("--out", default="data/btcusdt_20260501_0000_0656.npz")
    ap.add_argument("--buffer", type=int, default=27_000_000,
                    help="preallocated 64-byte event slots; keep ~= row count to fit 8 GB RAM")
    ap.add_argument("--ss-buffer", type=int, default=2_000_000)
    args = ap.parse_args()

    data = tardis.convert(
        input_files=[args.trades, args.book],  # trades FIRST, depth second
        output_filename=args.out,
        buffer_size=args.buffer,
        ss_buffer_size=args.ss_buffer,
    )
    print("events:", len(data))
    print("first event ts:", data[0]["exch_ts"], "last:", data[-1]["exch_ts"])


if __name__ == "__main__":
    main()

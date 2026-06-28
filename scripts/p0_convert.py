"""P0.2: Convert Tardis CSV slices -> hftbacktest .npz event format.

Key facts encoded here (see docs/JOURNAL.md Entry 1):
- Trades file MUST precede the depth file: a trade both prints AND removes book volume;
  hftbacktest processes the trade first so simulated queue positions aren't decremented twice.
- buffer_size is a preallocated numpy array of 64-byte events; sized to the slice (~26M rows)
  instead of the 100M default, which would not fit in 8 GB RAM.
- Tardis Binance Futures rows carry the exchange 'E' (send) timestamp, not 'T' (match) time,
  so measured feed latency is slightly understated (converter docstring, tardis.py:67).
"""
from hftbacktest.data.utils import tardis

data = tardis.convert(
    input_files=[
        "data/raw/trades_slice.csv",     # trades FIRST (queue double-count prevention)
        "data/raw/book_slice.csv",       # incremental L2 depth second
    ],
    output_filename="data/btcusdt_20260501_0000_0656.npz",
    buffer_size=27_000_000,
    ss_buffer_size=2_000_000,
)
print("events:", len(data))
print("first event ts:", data[0]["exch_ts"], "last:", data[-1]["exch_ts"])

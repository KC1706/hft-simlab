#!/usr/bin/env bash
# Self-recording market-data collector (PLAN P0.2) — accumulates OUR OWN dataset from live Binance
# USD-M futures websockets (free, $0). Records trade + bookTicker + depth@0ms, gzip, one file per
# UTC day (auto-rotates). This is how we acquire the 2nd+ days the Phase-4 sim-to-real gap needs
# (docs/PHASE4_ABLATION_DESIGN.md). Leave it running; each full UTC day yields one recording.
#
# The binary is built from refs/hftbacktest/collector (read-only reference; we only RUN it).
#   Usage:  scripts/record_market.sh [symbol ...]        # default: btcusdt
#   Stop:   Ctrl-C  (partial day is still a valid, if short, recording)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

BIN="$ROOT/refs/hftbacktest/target/release/collector"
[ -x "$BIN" ] || BIN="$ROOT/refs/hftbacktest/collector/target/release/collector"
if [ ! -x "$BIN" ]; then
  echo "collector not built. Run: (cd refs/hftbacktest/collector && cargo build --release)" >&2
  exit 1
fi

OUT="$ROOT/data/raw/recorded"
mkdir -p "$OUT"
SYMBOLS="${*:-btcusdt}"
PREFIX="$OUT/binancefuturesum"

echo "recording [$SYMBOLS] from binancefuturesum -> ${PREFIX}_<UTCdate>.gz"
echo "(streams: trade + bookTicker + depth@0ms; Ctrl-C to stop)"
exec "$BIN" "$PREFIX" binancefuturesum $SYMBOLS

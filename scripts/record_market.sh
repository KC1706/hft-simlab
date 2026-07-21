#!/usr/bin/env bash
# Self-recording market-data collector (PLAN P0.2) — accumulates OUR OWN dataset from live Binance
# USD-M futures websockets (free, $0). Records trade + bookTicker + depth@0ms, gzip, one file per
# UTC day (auto-rotates). This is how we acquire the 2nd+ days the Phase-4 sim-to-real gap needs
# (docs/PHASE4_ABLATION_DESIGN.md). Leave it running; each full UTC day yields one recording.
#
# The binary is built from refs/hftbacktest/collector (read-only reference; we only RUN it).
#   Usage:  scripts/record_market.sh [symbol ...]                  # default: btcusdt, Binance USD-M
#           EXCHANGE=bybit scripts/record_market.sh btcusdt        # if Binance is geo-blocked for you
#   Stop:   Ctrl-C  (partial day is still a valid, if short, recording)
#
# NOTE: must run on YOUR machine, NOT the Claude sandbox (which allowlists only a few domains and
# blocks all crypto exchanges). If Binance returns nothing, you are likely in a Binance-blocked
# region (e.g. US) — set EXCHANGE=bybit or EXCHANGE=hyperliquid (both supported by the collector).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXCHANGE="${EXCHANGE:-binancefuturesum}"   # binancefuturesum | binancefutures | bybit | hyperliquid

BIN="$ROOT/refs/hftbacktest/target/release/collector"
[ -x "$BIN" ] || BIN="$ROOT/refs/hftbacktest/collector/target/release/collector"
if [ ! -x "$BIN" ]; then
  echo "collector not built. Run: (cd refs/hftbacktest/collector && cargo build --release)" >&2
  exit 1
fi

OUT="$ROOT/data/raw/recorded"
mkdir -p "$OUT"
SYMBOLS="${*:-btcusdt}"
PREFIX="$OUT/$EXCHANGE"

echo "recording [$SYMBOLS] from $EXCHANGE -> ${PREFIX}_<UTCdate>.gz"
echo "(streams: trade + bookTicker + depth; Ctrl-C to stop)"
exec "$BIN" "$PREFIX" "$EXCHANGE" $SYMBOLS

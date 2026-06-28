//! A minimal L2 (market-by-price) order book.
//!
//! The book mirrors what an L2 feed publishes: one aggregate quantity per price
//! level per side. Updates are *absolute* ("level 50000.1 now holds 3.2"), not
//! deltas — sending qty 0 deletes the level. This is Binance-futures/Tardis
//! semantics and the dominant convention for crypto MBP feeds.
//!
//! Crossed/locked handling replicates hftbacktest's `HashMapMarketDepth`
//! (refs/hftbacktest/hftbacktest/src/depth/hashmapmarketdepth.rs:85-194): when an
//! update makes best_bid >= best_ask, the *opposite* best pointer skips past the
//! now-stale levels, but their map entries are retained — the feed is trusted to
//! refresh them. We must match this exactly for the P1.2 level-for-level
//! comparison against hftbacktest's reconstruction. We additionally count these
//! occurrences (`locked_updates` / `crossed_updates`), which hftbacktest doesn't.

use std::collections::BTreeMap;

/// Book side.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Side {
    Bid,
    Ask,
}

/// One price level as (price, aggregate quantity).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Level {
    pub px: f64,
    pub qty: f64,
}

/// Top-of-book state derived from the current best pointers.
///
/// Because crossing updates resolve immediately (pointer skip), `Locked` and
/// `Crossed` can only be observed if resolution is bypassed; they are kept in
/// the enum because the P1.2 harness asserts the book is *never* in them after
/// an update. Transient locks/crosses show up in the counters instead.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BookState {
    Empty,
    BidOnly,
    AskOnly,
    Normal,
    Locked,
    Crossed,
}

/// Minimal L2 order book over integer price ticks.
pub struct L2Book {
    tick_size: f64,
    lot_size: f64,
    bids: BTreeMap<i64, f64>,
    asks: BTreeMap<i64, f64>,
    best_bid: Option<i64>,
    best_ask: Option<i64>,
    /// Timestamp (ns) of the last applied update.
    pub last_ts: i64,
    /// Number of depth updates applied.
    pub depth_updates: u64,
    /// Updates that arrived locking the book (incoming level == opposite best).
    pub locked_updates: u64,
    /// Updates that arrived crossing the book (incoming level beyond opposite best).
    pub crossed_updates: u64,
}

impl L2Book {
    pub fn new(tick_size: f64, lot_size: f64) -> Self {
        assert!(tick_size > 0.0 && lot_size > 0.0);
        Self {
            tick_size,
            lot_size,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            best_bid: None,
            best_ask: None,
            last_ts: 0,
            depth_updates: 0,
            locked_updates: 0,
            crossed_updates: 0,
        }
    }

    /// Price -> integer tick. `round`, not `trunc`: f64 cannot represent most
    /// decimal prices exactly (50000.1 / 0.1 = 500000.99999...), so truncation
    /// would put the level one tick off.
    #[inline(always)]
    pub fn px_to_tick(&self, px: f64) -> i64 {
        (px / self.tick_size).round() as i64
    }

    #[inline(always)]
    pub fn tick_to_px(&self, tick: i64) -> f64 {
        tick as f64 * self.tick_size
    }

    /// Apply one incremental L2 update: set the aggregate quantity at a price
    /// level. `qty` rounding to zero lots deletes the level.
    pub fn set_level(&mut self, side: Side, px: f64, qty: f64, ts: i64) {
        let tick = self.px_to_tick(px);
        let qty_lots = (qty / self.lot_size).round() as i64;
        self.last_ts = ts;
        self.depth_updates += 1;

        match side {
            Side::Bid => {
                if qty_lots == 0 {
                    self.bids.remove(&tick);
                    if self.best_bid == Some(tick) {
                        // Best bid deleted: next best is the highest remaining bid below.
                        self.best_bid = self.bids.range(..tick).next_back().map(|(&t, _)| t);
                    }
                } else {
                    self.bids.insert(tick, qty);
                    if self.best_bid.map_or(true, |b| tick > b) {
                        self.best_bid = Some(tick);
                        if let Some(a) = self.best_ask {
                            if tick >= a {
                                if tick == a {
                                    self.locked_updates += 1;
                                } else {
                                    self.crossed_updates += 1;
                                }
                                // Stale-ask skip: pointer moves strictly above the
                                // crossing bid; entries in (old ask ..= tick] remain
                                // until the feed refreshes them (hftbacktest:122-128).
                                self.best_ask =
                                    self.asks.range(tick + 1..).next().map(|(&t, _)| t);
                            }
                        }
                    }
                }
            }
            Side::Ask => {
                if qty_lots == 0 {
                    self.asks.remove(&tick);
                    if self.best_ask == Some(tick) {
                        self.best_ask = self.asks.range(tick + 1..).next().map(|(&t, _)| t);
                    }
                } else {
                    self.asks.insert(tick, qty);
                    if self.best_ask.map_or(true, |a| tick < a) {
                        self.best_ask = Some(tick);
                        if let Some(b) = self.best_bid {
                            if tick <= b {
                                if tick == b {
                                    self.locked_updates += 1;
                                } else {
                                    self.crossed_updates += 1;
                                }
                                // Mirror image: best bid drops strictly below the
                                // crossing ask (hftbacktest:177-183).
                                self.best_bid =
                                    self.bids.range(..tick).next_back().map(|(&t, _)| t);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Clear one side up to and including `upto_px` (toward the book interior),
    /// as emitted before a snapshot. `None` clears the whole side.
    ///
    /// Next-best search here is a clean range scan. hftbacktest's equivalent
    /// (hashmapmarketdepth.rs:208-209) scans from `clear_upto - 1` exclusive,
    /// skipping a possible level at exactly `clear_upto - 1`; harmless there
    /// because a full snapshot always follows a clear, but we don't copy it.
    pub fn clear_side(&mut self, side: Side, upto_px: Option<f64>) {
        match (side, upto_px) {
            (Side::Bid, None) => {
                self.bids.clear();
                self.best_bid = None;
            }
            (Side::Bid, Some(px)) => {
                let upto = self.px_to_tick(px);
                self.bids.retain(|&t, _| t < upto);
                self.best_bid = self.bids.range(..upto).next_back().map(|(&t, _)| t);
            }
            (Side::Ask, None) => {
                self.asks.clear();
                self.best_ask = None;
            }
            (Side::Ask, Some(px)) => {
                let upto = self.px_to_tick(px);
                self.asks.retain(|&t, _| t > upto);
                self.best_ask = self.asks.range(upto + 1..).next().map(|(&t, _)| t);
            }
        }
    }

    pub fn clear_all(&mut self) {
        self.clear_side(Side::Bid, None);
        self.clear_side(Side::Ask, None);
    }

    // ---- inspection ----

    pub fn best_bid_tick(&self) -> Option<i64> {
        self.best_bid
    }

    pub fn best_ask_tick(&self) -> Option<i64> {
        self.best_ask
    }

    pub fn best_bid(&self) -> Option<Level> {
        self.best_bid.map(|t| Level {
            px: self.tick_to_px(t),
            qty: *self.bids.get(&t).unwrap_or(&0.0),
        })
    }

    pub fn best_ask(&self) -> Option<Level> {
        self.best_ask.map(|t| Level {
            px: self.tick_to_px(t),
            qty: *self.asks.get(&t).unwrap_or(&0.0),
        })
    }

    /// Mid price, defined only for a two-sided book.
    pub fn mid(&self) -> Option<f64> {
        match (self.best_bid, self.best_ask) {
            (Some(b), Some(a)) => Some((self.tick_to_px(b) + self.tick_to_px(a)) / 2.0),
            _ => None,
        }
    }

    /// Spread in ticks (>= 1 whenever the book is two-sided, by construction).
    pub fn spread_ticks(&self) -> Option<i64> {
        match (self.best_bid, self.best_ask) {
            (Some(b), Some(a)) => Some(a - b),
            _ => None,
        }
    }

    pub fn state(&self) -> BookState {
        match (self.best_bid, self.best_ask) {
            (None, None) => BookState::Empty,
            (Some(_), None) => BookState::BidOnly,
            (None, Some(_)) => BookState::AskOnly,
            (Some(b), Some(a)) if b > a => BookState::Crossed,
            (Some(b), Some(a)) if b == a => BookState::Locked,
            _ => BookState::Normal,
        }
    }

    /// Aggregate quantity at a tick (0 if absent). Reads the raw map, so stale
    /// retained levels are visible here — intentionally, for parity checks.
    pub fn qty_at_tick(&self, side: Side, tick: i64) -> f64 {
        let map = match side {
            Side::Bid => &self.bids,
            Side::Ask => &self.asks,
        };
        *map.get(&tick).unwrap_or(&0.0)
    }

    /// Top `n` levels from the best pointer toward the interior. Stale entries
    /// beyond the pointer (crossing leftovers) are excluded — this is "the book
    /// as the venue would display it".
    pub fn top_n(&self, side: Side, n: usize) -> Vec<Level> {
        match side {
            Side::Bid => match self.best_bid {
                None => Vec::new(),
                Some(b) => self
                    .bids
                    .range(..=b)
                    .rev()
                    .take(n)
                    .map(|(&t, &q)| Level { px: self.tick_to_px(t), qty: q })
                    .collect(),
            },
            Side::Ask => match self.best_ask {
                None => Vec::new(),
                Some(a) => self
                    .asks
                    .range(a..)
                    .take(n)
                    .map(|(&t, &q)| Level { px: self.tick_to_px(t), qty: q })
                    .collect(),
            },
        }
    }

    /// Number of price levels currently stored (including stale retained ones).
    pub fn n_levels(&self, side: Side) -> usize {
        match side {
            Side::Bid => self.bids.len(),
            Side::Ask => self.asks.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn book() -> L2Book {
        // BTCUSDT-perp-like: tick 0.1, lot 0.001
        L2Book::new(0.1, 0.001)
    }

    /// Display prices are reconstructed as `tick * tick_size`, an inexact f64
    /// product (500000 * 0.1 != 50000.0 exactly) — the very reason the book
    /// keys on integer ticks. Tests therefore assert ticks with `==` and
    /// display prices only approximately.
    fn assert_px(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-6,
            "px {actual} != {expected}"
        );
    }

    #[test]
    fn empty_book() {
        let b = book();
        assert_eq!(b.state(), BookState::Empty);
        assert_eq!(b.best_bid(), None);
        assert_eq!(b.best_ask(), None);
        assert_eq!(b.mid(), None);
        assert!(b.top_n(Side::Bid, 5).is_empty());
    }

    #[test]
    fn tick_rounding_is_exact_for_decimal_prices() {
        let b = book();
        // 50000.1 / 0.1 = 500000.99999... in f64; round() must land on 500001.
        assert_eq!(b.px_to_tick(50_000.1), 500_001);
        assert_eq!(b.px_to_tick(50_000.0), 500_000);
        assert_eq!(b.px_to_tick(0.1), 1);
    }

    #[test]
    fn basic_two_sided_book() {
        let mut b = book();
        b.set_level(Side::Bid, 50_000.0, 1.0, 1);
        b.set_level(Side::Bid, 49_999.9, 2.0, 2);
        b.set_level(Side::Ask, 50_000.2, 1.5, 3);
        b.set_level(Side::Ask, 50_000.3, 3.0, 4);

        assert_eq!(b.state(), BookState::Normal);
        assert_eq!(b.best_bid_tick(), Some(500_000));
        assert_eq!(b.best_ask_tick(), Some(500_002));
        assert_px(b.best_bid().unwrap().px, 50_000.0);
        assert_px(b.best_ask().unwrap().px, 50_000.2);
        assert_eq!(b.spread_ticks(), Some(2));
        assert_px(b.mid().unwrap(), 50_000.1);
    }

    #[test]
    fn updates_are_absolute_not_deltas() {
        let mut b = book();
        b.set_level(Side::Bid, 50_000.0, 1.0, 1);
        b.set_level(Side::Bid, 50_000.0, 2.5, 2);
        assert_eq!(b.best_bid().unwrap().qty, 2.5); // replaced, not 3.5
    }

    #[test]
    fn zero_qty_deletes_and_promotes_next_best() {
        let mut b = book();
        b.set_level(Side::Bid, 50_000.0, 1.0, 1);
        b.set_level(Side::Bid, 49_999.8, 2.0, 2);
        b.set_level(Side::Bid, 50_000.0, 0.0, 3);
        assert_eq!(b.best_bid_tick(), Some(499_998));

        // Deleting a non-best level must not move the best pointer.
        b.set_level(Side::Bid, 49_999.0, 5.0, 4);
        b.set_level(Side::Bid, 49_999.0, 0.0, 5);
        assert_eq!(b.best_bid_tick(), Some(499_998));

        // Sub-lot dust (< lot_size/2) rounds to zero lots -> also a delete.
        b.set_level(Side::Bid, 49_999.8, 0.0004, 6);
        assert_eq!(b.best_bid(), None);
        assert_eq!(b.state(), BookState::Empty);
    }

    #[test]
    fn crossing_bid_skips_pointer_but_retains_stale_asks() {
        let mut b = book();
        b.set_level(Side::Bid, 50_000.0, 1.0, 1);
        b.set_level(Side::Ask, 50_000.2, 1.0, 2);
        b.set_level(Side::Ask, 50_000.5, 2.0, 3);

        // Bid jumps over the best ask.
        b.set_level(Side::Bid, 50_000.3, 0.7, 4);

        assert_eq!(b.crossed_updates, 1);
        assert_eq!(b.state(), BookState::Normal); // resolved immediately
        assert_eq!(b.best_bid_tick(), Some(500_003));
        assert_eq!(b.best_ask_tick(), Some(500_005)); // skipped past 50000.2

        // hftbacktest parity: the stale ask entry is retained in the map...
        assert_eq!(b.qty_at_tick(Side::Ask, b.px_to_tick(50_000.2)), 1.0);
        // ...but excluded from the displayed ladder.
        let asks = b.top_n(Side::Ask, 5);
        assert_px(asks[0].px, 50_000.5);
        assert_eq!(asks.len(), 1);
    }

    #[test]
    fn locking_update_is_counted_and_resolved() {
        let mut b = book();
        b.set_level(Side::Bid, 50_000.0, 1.0, 1);
        b.set_level(Side::Ask, 50_000.2, 1.0, 2);
        // Ask lands exactly on the best bid -> locked, bid pointer must drop.
        b.set_level(Side::Ask, 50_000.0, 1.0, 3);
        assert_eq!(b.locked_updates, 1);
        assert_eq!(b.best_ask_tick(), Some(500_000));
        assert_eq!(b.best_bid(), None); // no bid left below
        assert_eq!(b.state(), BookState::AskOnly);
    }

    #[test]
    fn crossing_ask_drops_bid_below_crossing_price() {
        let mut b = book();
        b.set_level(Side::Bid, 50_000.0, 1.0, 1);
        b.set_level(Side::Bid, 49_999.9, 1.0, 2);
        b.set_level(Side::Bid, 49_999.5, 1.0, 3);
        b.set_level(Side::Ask, 50_000.5, 1.0, 4);

        // Ask crosses deep: below best AND second-best bid.
        b.set_level(Side::Ask, 49_999.8, 0.4, 5);

        assert_eq!(b.crossed_updates, 1);
        // New best bid is the highest bid strictly below the crossing ask,
        // not strictly below the old best bid (hftbacktest:179-182 parity).
        assert_eq!(b.best_bid_tick(), Some(499_995));
        assert_eq!(b.best_ask_tick(), Some(499_998));
        // Stale bids retained in the map, hidden from the ladder.
        assert_eq!(b.qty_at_tick(Side::Bid, b.px_to_tick(50_000.0)), 1.0);
        assert_px(b.top_n(Side::Bid, 5)[0].px, 49_999.5);
    }

    #[test]
    fn top_n_orders_outward_from_best() {
        let mut b = book();
        for (i, px) in [50_000.0, 49_999.9, 49_999.7, 49_999.6].iter().enumerate() {
            b.set_level(Side::Bid, *px, (i + 1) as f64, i as i64);
        }
        let bids = b.top_n(Side::Bid, 3);
        let ticks: Vec<i64> = bids.iter().map(|l| b.px_to_tick(l.px)).collect();
        assert_eq!(ticks, vec![500_000, 499_999, 499_997]); // descending, truncated
    }

    #[test]
    fn clear_full_and_upto() {
        let mut b = book();
        b.set_level(Side::Bid, 50_000.0, 1.0, 1);
        b.set_level(Side::Bid, 49_999.9, 1.0, 2);
        b.set_level(Side::Bid, 49_999.8, 1.0, 3);
        b.set_level(Side::Ask, 50_000.2, 1.0, 4);

        // Clear bids down to and including 49999.9.
        b.clear_side(Side::Bid, Some(49_999.9));
        assert_eq!(b.best_bid_tick(), Some(499_998));
        assert_eq!(b.n_levels(Side::Bid), 1);

        b.clear_all();
        assert_eq!(b.state(), BookState::Empty);
        assert_eq!(b.n_levels(Side::Ask), 0);
    }

    #[test]
    fn snapshot_rebuild_after_clear() {
        let mut b = book();
        b.set_level(Side::Bid, 50_000.0, 1.0, 1);
        b.set_level(Side::Ask, 50_000.2, 1.0, 2);
        // Day-boundary pattern in our npz: clear then snapshot rows as plain sets.
        b.clear_all();
        b.set_level(Side::Bid, 51_000.0, 2.0, 3);
        b.set_level(Side::Ask, 51_000.4, 2.0, 4);
        assert_eq!(b.best_bid_tick(), Some(510_000));
        assert_eq!(b.best_ask_tick(), Some(510_004));
        assert_eq!(b.spread_ticks(), Some(4));
    }

    /// Pseudo-random soak: after every update the book must be uncrossed and the
    /// best pointers must sit on live levels. Catches pointer-maintenance bugs
    /// that handpicked cases miss.
    #[test]
    fn invariants_hold_under_random_updates() {
        let mut b = book();
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut rng = move || {
            // xorshift64* — tiny deterministic PRNG, no dependencies
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
            state
        };
        for i in 0..100_000 {
            let r = rng();
            let side = if r & 1 == 0 { Side::Bid } else { Side::Ask };
            let tick = 500_000 + ((r >> 8) % 41) as i64 - 20; // ±20 ticks around 50k
            let qty = if (r >> 16) % 4 == 0 { 0.0 } else { ((r >> 24) % 100) as f64 * 0.001 };
            b.set_level(side, tick as f64 * 0.1, qty, i);

            if let (Some(bb), Some(ba)) = (b.best_bid_tick(), b.best_ask_tick()) {
                assert!(bb < ba, "book crossed/locked after update {i}: bid {bb} ask {ba}");
            }
            if let Some(bb) = b.best_bid_tick() {
                assert!(b.qty_at_tick(Side::Bid, bb) > 0.0, "best bid points at empty level");
            }
            if let Some(ba) = b.best_ask_tick() {
                assert!(b.qty_at_tick(Side::Ask, ba) > 0.0, "best ask points at empty level");
            }
        }
        // The soak must actually have exercised the interesting paths.
        assert!(b.crossed_updates + b.locked_updates > 0);
    }
}

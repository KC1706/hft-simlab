//! P2.2 fill labeler: generate virtual-order fill outcomes from L2 tape.
//!
//! Every second, places virtual passive orders at the current best bid and
//! best ask at six queue fractions (0.05..1.0 × level quantity). Tracks each
//! order forward until it fills, its level disappears, or the 60-second
//! horizon expires. Emits a CSV used by scripts/p22_fill_calibrate.py to fit
//! the calibrated queue model.
//!
//! Fill criteria (two flavors — both logged):
//!   trade-only : queue_ahead shrinks only on TRADE events at the order's level
//!   any-decrease: queue_ahead also shrinks when depth updates show level qty < queue_ahead
//!                 (optimistic bound — treats all level decreases as from ahead of you)
//!
//! Output columns:
//!   ts_placed, side (0=bid/1=ask), price_tick, init_qty, queue_frac,
//!   spread_ticks, fill_trade_1s, fill_trade_10s, fill_trade_60s,
//!   fill_any_1s, fill_any_10s, fill_any_60s
//!
//! Usage: fill_labeler <file.npz> [--out fill_labels.csv]

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write as IoWrite};
use std::time::Instant;

use lob_core::events::{
    BUY_EVENT, DEPTH_BBO_EVENT, DEPTH_CLEAR_EVENT, DEPTH_EVENT, DEPTH_SNAPSHOT_EVENT,
    LOCAL_EVENT, TRADE_EVENT,
};
use lob_core::{L2Book, NpzEventReader, Side};

const TICK_SIZE: f64 = 0.1;
const LOT_SIZE: f64 = 0.001;
const PLACEMENT_INTERVAL_NS: i64 = 1_000_000_000; // 1s between placement batches
const QUEUE_FRACS: &[f64] = &[0.05, 0.2, 0.4, 0.6, 0.8, 1.0];
// Three horizons in nanoseconds: 1s, 10s, 60s.
const HORIZONS_NS: [i64; 3] = [1_000_000_000, 10_000_000_000, 60_000_000_000];
const MAX_HORIZON: i64 = 60_000_000_000;

struct VirtualOrder {
    ts_placed: i64,
    side: Side,
    price_tick: i64,
    init_qty: f64,
    queue_frac: f64,
    queue_ahead_trade: f64,
    queue_ahead_any: f64,
    spread_ticks: i64,
    /// fill_trade[h] = Some(true/false) once horizon h expires or order is done.
    fill_trade: [Option<bool>; 3],
    fill_any: [Option<bool>; 3],
    done: bool,
}

impl VirtualOrder {
    fn new(ts: i64, side: Side, price_tick: i64, init_qty: f64,
           queue_frac: f64, spread_ticks: i64) -> Self {
        let qa = queue_frac * init_qty;
        Self {
            ts_placed: ts, side, price_tick, init_qty, queue_frac,
            queue_ahead_trade: qa, queue_ahead_any: qa, spread_ticks,
            fill_trade: [None; 3], fill_any: [None; 3], done: false,
        }
    }

    /// Apply a trade at this order's level.
    fn on_trade(&mut self, qty: f64) {
        self.queue_ahead_trade -= qty;
        self.queue_ahead_any -= qty;
    }

    /// Apply a depth update at this order's level (optimistic: min with new_qty).
    /// Returns true if the level was deleted (new_qty = 0 → abandon).
    fn on_depth(&mut self, new_qty: f64) -> bool {
        if new_qty <= 0.0 {
            return true; // level deleted
        }
        self.queue_ahead_any = self.queue_ahead_any.min(new_qty);
        false
    }

    /// Snapshot fill outcomes at current timestamp. Returns true if fully done.
    fn tick(&mut self, now_ts: i64) -> bool {
        let elapsed = now_ts - self.ts_placed;
        for h in 0..3 {
            let deadline = HORIZONS_NS[h];
            if elapsed >= deadline {
                if self.fill_trade[h].is_none() {
                    self.fill_trade[h] = Some(self.queue_ahead_trade <= 0.0);
                }
                if self.fill_any[h].is_none() {
                    self.fill_any[h] = Some(self.queue_ahead_any <= 0.0);
                }
            }
        }
        // If the max horizon expired, mark everything done.
        if elapsed >= MAX_HORIZON {
            for h in 0..3 {
                if self.fill_trade[h].is_none() {
                    self.fill_trade[h] = Some(self.queue_ahead_trade <= 0.0);
                }
                if self.fill_any[h].is_none() {
                    self.fill_any[h] = Some(self.queue_ahead_any <= 0.0);
                }
            }
            self.done = true;
        }
        // If all three horizons for both criteria are set, done.
        if self.fill_trade.iter().all(|x| x.is_some())
            && self.fill_any.iter().all(|x| x.is_some())
        {
            self.done = true;
        }
        self.done
    }

    /// Level disappeared or tape ended — record current queue state for all open horizons.
    /// If the queue was already depleted (filled) before abandonment, future horizons are
    /// true (the fill happened); if not, they are false (order would have been cancelled).
    fn abandon(&mut self) {
        let t = self.queue_ahead_trade <= 0.0;
        let a = self.queue_ahead_any <= 0.0;
        for h in 0..3 {
            if self.fill_trade[h].is_none() {
                self.fill_trade[h] = Some(t);
            }
            if self.fill_any[h].is_none() {
                self.fill_any[h] = Some(a);
            }
        }
        self.done = true;
    }

    fn write_row(&self, out: &mut impl IoWrite) {
        let s = if self.side == Side::Bid { 0 } else { 1 };
        let ft = self.fill_trade.map(|v| v.unwrap_or(false) as u8);
        let fa = self.fill_any.map(|v| v.unwrap_or(false) as u8);
        writeln!(
            out,
            "{},{},{},{:.4},{:.4},{},{},{},{},{},{},{}",
            self.ts_placed, s, self.price_tick,
            self.init_qty, self.queue_frac, self.spread_ticks,
            ft[0], ft[1], ft[2], fa[0], fa[1], fa[2],
        ).unwrap();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: fill_labeler <file.npz> [--out fill_labels.csv]");
        std::process::exit(1);
    }
    let path = &args[1];
    let mut out_path = "fill_labels.csv".to_string();
    let mut i = 2;
    while i < args.len() {
        if args[i] == "--out" && i + 1 < args.len() {
            out_path = args[i + 1].clone();
            i += 2;
        } else {
            i += 1;
        }
    }

    let out_file = File::create(&out_path).expect("create output CSV");
    let mut out = BufWriter::new(out_file);
    writeln!(out, "ts_placed,side,price_tick,init_qty,queue_frac,spread_ticks,\
                   fill_trade_1s,fill_trade_10s,fill_trade_60s,\
                   fill_any_1s,fill_any_10s,fill_any_60s").unwrap();

    let reader = NpzEventReader::open(path).expect("open npz");
    let mut book = L2Book::new(TICK_SIZE, LOT_SIZE);

    // Active virtual orders, indexed by encoded key = price_tick * 2 + side_bit.
    // Value: Vec of indices into `orders`.
    let mut by_level: HashMap<i64, Vec<usize>> = HashMap::new();
    let mut orders: Vec<VirtualOrder> = Vec::new();
    let mut completed: Vec<usize> = Vec::new();

    let mut next_placement: i64 = i64::MIN;
    let mut n_placed: u64 = 0;
    let mut n_written: u64 = 0;
    let t0 = Instant::now();

    let level_key = |side: Side, tick: i64| tick * 2 + if side == Side::Bid { 0 } else { 1 };

    for ev in reader {
        let ev = ev.expect("read event");
        if ev.ev & LOCAL_EVENT == 0 {
            continue;
        }
        let ts = ev.local_ts;

        // ── 1. Advance horizon check for all active orders ──────────────────
        // We do this before applying the event so that the horizon snapshot
        // reflects the state at exactly the deadline, not after more events.
        // (Minor: in practice all active orders are checked each event.)
        for &idx in &completed {
            // will drain below
            let _ = idx;
        }
        completed.clear();
        for (idx, ord) in orders.iter_mut().enumerate() {
            if !ord.done && ord.tick(ts) {
                completed.push(idx);
            }
        }
        // Write and remove completed orders (in reverse index order).
        completed.sort_unstable_by(|a, b| b.cmp(a)); // reverse
        for &idx in &completed {
            orders[idx].write_row(&mut out);
            n_written += 1;
            // Remove from by_level index.
            let k = level_key(orders[idx].side, orders[idx].price_tick);
            if let Some(v) = by_level.get_mut(&k) {
                v.retain(|&i| i != idx);
            }
            // Swap-remove: swap with last element, update its index in by_level.
            let last = orders.len() - 1;
            if idx != last {
                let last_k = level_key(orders[last].side, orders[last].price_tick);
                if let Some(v) = by_level.get_mut(&last_k) {
                    for x in v.iter_mut() {
                        if *x == last { *x = idx; }
                    }
                }
                orders.swap(idx, last);
            }
            orders.pop();
        }
        completed.clear();

        // ── 2. Apply event to book ───────────────────────────────────────────
        let is_buy = ev.ev & BUY_EVENT != 0;
        let side = if is_buy { Side::Bid } else { Side::Ask };
        let is_depth = ev.ev & DEPTH_EVENT != 0;
        let is_snap = ev.ev & DEPTH_SNAPSHOT_EVENT != 0;
        let is_bbo = ev.ev & DEPTH_BBO_EVENT != 0;
        let is_clear = ev.ev & DEPTH_CLEAR_EVENT != 0;
        let is_trade = ev.ev & TRADE_EVENT != 0;

        if is_trade {
            // Trades do not touch the book, but do advance queue position.
            let trade_side = side; // aggressor side: BUY means a buy hit the ask
            // The resting orders being filled are on the *opposite* side.
            let resting_side = if trade_side == Side::Bid { Side::Ask } else { Side::Bid };
            let price_tick = book.px_to_tick(ev.px);
            let k = level_key(resting_side, price_tick);
            if let Some(indices) = by_level.get(&k) {
                for &idx in indices.iter() {
                    orders[idx].on_trade(ev.qty);
                }
            }
        } else if is_depth || is_snap || is_bbo {
            let prev_qty = book.qty_at_tick(side, book.px_to_tick(ev.px));
            book.set_level(side, ev.px, ev.qty, ts);
            let price_tick = book.px_to_tick(ev.px);
            let new_qty = ev.qty;
            // Only notify virtual orders if qty changed.
            if (new_qty - prev_qty).abs() > 1e-9 {
                let k = level_key(side, price_tick);
                if let Some(indices) = by_level.get(&k) {
                    let mut to_abandon = Vec::new();
                    for &idx in indices.iter() {
                        if orders[idx].on_depth(new_qty) {
                            to_abandon.push(idx);
                        }
                    }
                    for idx in to_abandon {
                        orders[idx].abandon();
                    }
                }
            }
        } else if is_clear {
            // CLEAR: all active orders on the cleared side are abandoned.
            let clear_side = side;
            let keys_to_abandon: Vec<i64> = by_level.keys()
                .filter(|&&k| {
                    let ord_side = if k % 2 == 0 { Side::Bid } else { Side::Ask };
                    ord_side == clear_side
                })
                .copied()
                .collect();
            for k in keys_to_abandon {
                if let Some(indices) = by_level.get(&k) {
                    for &idx in indices.iter() {
                        orders[idx].abandon();
                    }
                }
            }
            book.clear_side(clear_side, if ev.px > 0.0 { Some(ev.px) } else { None });
        }

        // ── 3. Abandon orders whose level is no longer the best ─────────────
        // A passive bid at P is abandoned if best_bid has moved below P.
        // A passive ask at P is abandoned if best_ask has moved above P.
        // We only check orders placed at the best level (which they all are).
        if let (Some(bb), Some(ba)) = (book.best_bid_tick(), book.best_ask_tick()) {
            // Collect keys that correspond to stale levels.
            let stale: Vec<i64> = by_level.keys().copied().filter(|&k| {
                let tick = k / 2;
                let s = if k % 2 == 0 { Side::Bid } else { Side::Ask };
                match s {
                    Side::Bid => tick < bb, // placed at a bid level now below best bid
                    Side::Ask => tick > ba, // placed at an ask level now above best ask
                }
            }).collect();
            for k in stale {
                if let Some(indices) = by_level.get(&k) {
                    for &idx in indices.iter() {
                        orders[idx].abandon();
                    }
                }
            }
        }

        // ── 4. Flush abandoned orders ────────────────────────────────────────
        let mut i2 = 0;
        while i2 < orders.len() {
            if orders[i2].done {
                orders[i2].write_row(&mut out);
                n_written += 1;
                let k = level_key(orders[i2].side, orders[i2].price_tick);
                if let Some(v) = by_level.get_mut(&k) {
                    v.retain(|&x| x != i2);
                }
                let last = orders.len() - 1;
                if i2 != last {
                    let last_k = level_key(orders[last].side, orders[last].price_tick);
                    if let Some(v) = by_level.get_mut(&last_k) {
                        for x in v.iter_mut() { if *x == last { *x = i2; } }
                    }
                    orders.swap(i2, last);
                }
                orders.pop();
                // don't increment i2 — recheck same index
            } else {
                i2 += 1;
            }
        }

        // ── 5. Place new virtual orders ──────────────────────────────────────
        if next_placement == i64::MIN {
            next_placement = ts;
        }
        if ts >= next_placement {
            next_placement += PLACEMENT_INTERVAL_NS;
            // Only place if book is two-sided.
            if let (Some(bb_tick), Some(ba_tick)) = (book.best_bid_tick(), book.best_ask_tick()) {
                let spread = (ba_tick - bb_tick) as i64;
                let bb_qty = book.qty_at_tick(Side::Bid, bb_tick);
                let ba_qty = book.qty_at_tick(Side::Ask, ba_tick);
                // Only place if the level has non-trivial depth.
                if bb_qty > LOT_SIZE && ba_qty > LOT_SIZE {
                    for &frac in QUEUE_FRACS {
                        let bid_idx = orders.len();
                        orders.push(VirtualOrder::new(
                            ts, Side::Bid, bb_tick, bb_qty, frac, spread,
                        ));
                        by_level.entry(level_key(Side::Bid, bb_tick))
                            .or_default().push(bid_idx);

                        let ask_idx = orders.len();
                        orders.push(VirtualOrder::new(
                            ts, Side::Ask, ba_tick, ba_qty, frac, spread,
                        ));
                        by_level.entry(level_key(Side::Ask, ba_tick))
                            .or_default().push(ask_idx);
                        n_placed += 2;
                    }
                }
            }
        }
    }

    // Flush remaining active orders (horizons not yet expired — write with
    // whatever fill state we have at end of tape).
    for ord in &mut orders {
        ord.abandon();
        ord.write_row(&mut out);
        n_written += 1;
    }

    let elapsed = t0.elapsed();
    eprintln!(
        "fill_labeler: placed {n_placed} virtual orders, wrote {n_written} rows in {elapsed:.2?}"
    );
    eprintln!("output: {out_path}");
}

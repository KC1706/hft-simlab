//! P3.1 — Export our L2 crypto data into the TRADES/LOBSTER training representation.
//!
//! TRADES (arXiv 2502.07071, code in refs/DeepMarket) is a conditional diffusion model
//! trained on **L3** LOBSTER message streams: each row is one order event
//! `(time, event_type, order_id, size, price, direction)` conditioned on the top-10 book.
//! Our crypto feed is **L2** (aggregate per-level quantities + trades, no order IDs), so we
//! *synthesize* an L3-like message stream from the L2 deltas:
//!
//!   - a level quantity INCREASE  → SUBMISSION   (1)  size = +Δ
//!   - a level quantity DECREASE  → CANCELLATION (2)  size = −Δ   (DELETION (3) if level emptied)
//!   - a trade print              → EXECUTION    (4)  size = traded qty
//!
//! Each emitted row carries the order features followed by the 40-column top-10 book
//! snapshot (LOBSTER order: askP1,askS1,bidP1,bidS1,…) taken *after* the event, plus four
//! convenience metrics (mid, spread, imbalance, vwap) — the 50-column TRADES-LOB layout.
//!
//! Approximations (documented in docs/JOURNAL.md Entry 11): without order IDs we cannot
//! attribute a decrease to a specific order, and the trade-vs-cancel split is inferred
//! (a decrease co-located with a just-printed trade is real execution, but we tag it from
//! the event kind only) — so the synthesized cancellation rate is an upper bound. This is
//! acceptable for *learning the event distribution*; it is not a claim of true L3 ground
//! truth. The Python packer (scripts/p31_pack_trades.py) normalizes and slices into the
//! .npy the TRADES DataModule expects.
//!
//! Usage: lob_export <file.npz> --out rows.csv [--max-rows N] [--levels N]

use std::env;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufWriter, Write};

use lob_core::book::{L2Book, Side};
use lob_core::events::{
    BUY_EVENT, DEPTH_BBO_EVENT, DEPTH_CLEAR_EVENT, DEPTH_EVENT, DEPTH_SNAPSHOT_EVENT, LOCAL_EVENT,
    SELL_EVENT, TRADE_EVENT,
};
use lob_core::NpzEventReader;

const TICK_SIZE: f64 = 0.1;
const LOT_SIZE: f64 = 0.001;
const LEVELS: usize = 10;

// LOBSTER event-type codes (constants.py: OrderEvent).
const SUBMISSION: u8 = 1;
const CANCELLATION: u8 = 2;
const DELETION: u8 = 3;
const EXECUTION: u8 = 4;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: lob_export <file.npz> --out rows.csv [--max-rows N] [--levels N]");
        std::process::exit(2);
    }
    let path = &args[1];
    let mut out_path = String::from("trades_rows.csv");
    let mut max_rows: u64 = 500_000;
    let mut levels = LEVELS;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => { out_path = args[i + 1].clone(); i += 2; }
            "--max-rows" => { max_rows = args[i + 1].parse().expect("bad max-rows"); i += 2; }
            "--levels" => { levels = args[i + 1].parse().expect("bad levels"); i += 2; }
            a => { eprintln!("unknown arg: {a}"); std::process::exit(2); }
        }
    }

    let reader = NpzEventReader::open(path).expect("open npz");
    let mut book = L2Book::new(TICK_SIZE, LOT_SIZE);
    let mut out = BufWriter::new(File::create(&out_path).expect("create out"));

    // header: order features (6) + 40 LOB cols + 4 metrics
    let mut header = String::from("time,event_type,order_id,size,price,direction");
    for l in 1..=levels {
        let _ = write!(header, ",ask_p{l},ask_s{l},bid_p{l},bid_s{l}");
    }
    header.push_str(",mid,spread,imbalance,vwap");
    writeln!(out, "{header}").unwrap();

    let mut order_id: u64 = 0;
    let mut rows: u64 = 0;
    let mut session_start = i64::MIN;

    for ev in reader {
        let ev = ev.expect("read event");
        if ev.ev & LOCAL_EVENT == 0 {
            continue;
        }
        if session_start == i64::MIN {
            session_start = ev.local_ts;
        }
        let now = ev.local_ts;
        let kind = ev.kind();

        // Determine the synthesized order event (if any) BEFORE mutating the book.
        let synth: Option<(u8, f64, i64, i32)> = match kind {
            DEPTH_EVENT | DEPTH_SNAPSHOT_EVENT | DEPTH_BBO_EVENT => {
                let side = if ev.ev & BUY_EVENT != 0 { Side::Bid } else { Side::Ask };
                let tick = book.px_to_tick(ev.px);
                let old_qty = book.qty_at_tick(side, tick);
                let delta = ev.qty - old_qty;
                let dir = if side == Side::Bid { 1 } else { -1 };
                if delta > 0.0 {
                    Some((SUBMISSION, delta, tick, dir))
                } else if delta < 0.0 {
                    let et = if ev.qty <= 0.0 { DELETION } else { CANCELLATION };
                    Some((et, -delta, tick, dir))
                } else {
                    None
                }
            }
            TRADE_EVENT => {
                // Aggressor: BUY trade lifts the ask (executes a sell limit) → direction -1.
                let dir = if ev.ev & BUY_EVENT != 0 { -1 } else { 1 };
                Some((EXECUTION, ev.qty, book.px_to_tick(ev.px), dir))
            }
            _ => None,
        };

        // Apply the event to the book (mirrors bin/replay.rs).
        match kind {
            DEPTH_EVENT | DEPTH_SNAPSHOT_EVENT | DEPTH_BBO_EVENT => {
                let side = if ev.ev & BUY_EVENT != 0 { Side::Bid } else { Side::Ask };
                book.set_level(side, ev.px, ev.qty, now);
            }
            DEPTH_CLEAR_EVENT => {
                let has_buy = ev.ev & BUY_EVENT != 0;
                let has_sell = ev.ev & SELL_EVENT != 0;
                match (has_buy, has_sell) {
                    (true, false) => book.clear_side(Side::Bid, Some(ev.px)),
                    (false, true) => book.clear_side(Side::Ask, Some(ev.px)),
                    _ => book.clear_all(),
                }
            }
            _ => {}
        }

        // Emit a row only for a synthesized event with a two-sided book.
        let Some((event_type, size, price_tick, dir)) = synth else { continue };
        let (Some(_bb), Some(_ba)) = (book.best_bid(), book.best_ask()) else { continue };

        order_id += 1;
        let t = (now - session_start) as f64 / 1e9; // seconds since session start

        let mut row = String::with_capacity(512);
        let _ = write!(row, "{t:.9},{event_type},{order_id},{size},{price_tick},{dir}");

        let bids = book.top_n(Side::Bid, levels);
        let asks = book.top_n(Side::Ask, levels);
        let (mut bid_vol, mut ask_vol, mut pv, mut vol) = (0.0, 0.0, 0.0, 0.0);
        for l in 0..levels {
            let ap = asks.get(l).map_or(0, |x| book.px_to_tick(x.px));
            let asz = asks.get(l).map_or(0.0, |x| x.qty);
            let bp = bids.get(l).map_or(0, |x| book.px_to_tick(x.px));
            let bsz = bids.get(l).map_or(0.0, |x| x.qty);
            let _ = write!(row, ",{ap},{asz},{bp},{bsz}");
            ask_vol += asz; bid_vol += bsz;
            pv += ap as f64 * asz + bp as f64 * bsz;
            vol += asz + bsz;
        }
        let bb_t = book.best_bid_tick().unwrap();
        let ba_t = book.best_ask_tick().unwrap();
        let mid = (bb_t + ba_t) as f64 / 2.0;
        let spread = (ba_t - bb_t) as f64;
        let imb = if bid_vol + ask_vol > 0.0 { bid_vol / (bid_vol + ask_vol) } else { 0.5 };
        let vwap = if vol > 0.0 { pv / vol } else { mid };
        let _ = write!(row, ",{mid},{spread},{imb},{vwap}");
        writeln!(out, "{row}").unwrap();

        rows += 1;
        if rows >= max_rows {
            break;
        }
    }

    out.flush().unwrap();
    println!("lob_export: wrote {rows} rows ({levels} levels) -> {out_path}");
}

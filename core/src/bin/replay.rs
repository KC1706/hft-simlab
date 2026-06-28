//! P1.2 replay harness: stream an hftbacktest `.npz` day through our `L2Book`.
//!
//! Replays the LOCAL view of the feed (what our collector saw, at local_ts),
//! mirroring hftbacktest's local processor dispatch (refs/.../proc/local.rs:290+):
//! depth/snapshot/BBO events set levels, clear events clear up to a price,
//! trades do NOT touch the book (their depth deltas arrive separately — applying
//! both would double-count).
//!
//! Usage:
//!   replay <file.npz> [--snapshots ts1,ts2,...] [--depth N] [--json out.json]
//!                     [--measure out_dir] [--grid-ms N]
//!
//! Prints a consistency + throughput report; with --snapshots, captures the
//! top-N ladder at each local timestamp (state after all LOCAL events with
//! local_ts <= ts) for the parity test against hftbacktest's reconstruction.
//!
//! --measure (P1.3) logs two CSVs for the microstructure measurement scripts:
//!   samples.csv  one row per grid interval (default 100 ms): top-of-book,
//!                top-10 level quantities per side, per-interval OFI
//!                (Cont-Kukanov-Stoikov), signed trade volume and counts.
//!   trades.csv   the full trade tape: ts, aggressor sign, price tick, qty.

use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufWriter, Write as IoWrite};
use std::time::Instant;

use lob_core::events::{
    BUY_EVENT, DEPTH_BBO_EVENT, DEPTH_CLEAR_EVENT, DEPTH_EVENT, DEPTH_SNAPSHOT_EVENT,
    LOCAL_EVENT, SELL_EVENT, TRADE_EVENT,
};
use lob_core::{L2Book, Level, NpzEventReader, Side};

// BTCUSDT USDT-margined perp on Binance Futures (same as Phase 0 config).
const TICK_SIZE: f64 = 0.1;
const LOT_SIZE: f64 = 0.001;

#[derive(Default)]
struct Counts {
    depth: u64,
    snapshot: u64,
    bbo: u64,
    clear: u64,
    trade: u64,
    other: u64,
    local_applied: u64,
}

#[derive(Default)]
struct Violations {
    local_ts_order: u64,
    exch_ts_order: u64,
    neg_qty: u64,
    bad_px: u64,
    crossed_state: u64,
}

struct Snapshot {
    ts: i64,
    bids: Vec<Level>,
    asks: Vec<Level>,
}

const PROFILE_LEVELS: usize = 10;

/// P1.3 measurement state: grid sampling + per-interval flow accumulators.
struct Measure {
    samples: BufWriter<File>,
    trades: BufWriter<File>,
    grid_ns: i64,
    next_grid: i64, // i64::MIN until the first LOCAL event sets the grid origin
    // previous top-of-book observation (bb_tick, bb_qty, ba_tick, ba_qty) for OFI
    prev_top: Option<(i64, f64, i64, f64)>,
    ofi: f64,
    buy_vol: f64,
    sell_vol: f64,
    n_buys: u32,
    n_sells: u32,
    rows_written: u64,
    skipped_one_sided: u64,
}

impl Measure {
    fn new(dir: &str, grid_ms: i64) -> Self {
        std::fs::create_dir_all(dir).expect("create measure dir");
        let mut samples =
            BufWriter::new(File::create(format!("{dir}/samples.csv")).expect("samples.csv"));
        let mut header = String::from("ts,bid_tick,bid_qty,ask_tick,ask_qty,spread_ticks,ofi,\
             buy_vol,sell_vol,n_buys,n_sells");
        for i in 1..=PROFILE_LEVELS {
            let _ = write!(header, ",bq{i}");
        }
        for i in 1..=PROFILE_LEVELS {
            let _ = write!(header, ",aq{i}");
        }
        writeln!(samples, "{header}").unwrap();
        let mut trades =
            BufWriter::new(File::create(format!("{dir}/trades.csv")).expect("trades.csv"));
        writeln!(trades, "ts,sign,px_tick,qty").unwrap();
        Self {
            samples,
            trades,
            grid_ns: grid_ms * 1_000_000,
            next_grid: i64::MIN,
            prev_top: None,
            ofi: 0.0,
            buy_vol: 0.0,
            sell_vol: 0.0,
            n_buys: 0,
            n_sells: 0,
            rows_written: 0,
            skipped_one_sided: 0,
        }
    }

    /// Emit rows for every grid boundary strictly before this event's local_ts.
    /// (A sample at G = book state after all LOCAL events with local_ts <= G.)
    fn maybe_emit(&mut self, book: &L2Book, ev_local_ts: i64) {
        if self.next_grid == i64::MIN {
            // Grid origin: first boundary at/after the first LOCAL event.
            self.next_grid = ev_local_ts.div_euclid(self.grid_ns) * self.grid_ns;
            if self.next_grid < ev_local_ts {
                self.next_grid += self.grid_ns;
            }
            return;
        }
        while ev_local_ts > self.next_grid {
            self.emit(book, self.next_grid);
            self.ofi = 0.0;
            self.buy_vol = 0.0;
            self.sell_vol = 0.0;
            self.n_buys = 0;
            self.n_sells = 0;
            self.next_grid += self.grid_ns;
        }
    }

    fn emit(&mut self, book: &L2Book, ts: i64) {
        let (Some(bb), Some(ba)) = (book.best_bid(), book.best_ask()) else {
            self.skipped_one_sided += 1;
            return;
        };
        let mut row = String::with_capacity(360);
        let _ = write!(
            row,
            "{ts},{},{},{},{},{},{},{},{},{},{}",
            book.px_to_tick(bb.px),
            bb.qty,
            book.px_to_tick(ba.px),
            ba.qty,
            book.spread_ticks().unwrap(),
            self.ofi,
            self.buy_vol,
            self.sell_vol,
            self.n_buys,
            self.n_sells
        );
        for side in [Side::Bid, Side::Ask] {
            let levels = book.top_n(side, PROFILE_LEVELS);
            for i in 0..PROFILE_LEVELS {
                let q = levels.get(i).map_or(0.0, |l| l.qty);
                let _ = write!(row, ",{q}");
            }
        }
        writeln!(self.samples, "{row}").unwrap();
        self.rows_written += 1;
    }

    /// Cont-Kukanov-Stoikov order-flow imbalance contribution of one
    /// top-of-book transition (arXiv 1011.6402 eq. for e_n).
    fn update_ofi(&mut self, book: &L2Book) {
        let (Some(bb), Some(ba)) = (book.best_bid(), book.best_ask()) else {
            return;
        };
        let cur = (
            book.px_to_tick(bb.px),
            bb.qty,
            book.px_to_tick(ba.px),
            ba.qty,
        );
        if let Some((pb0, qb0, pa0, qa0)) = self.prev_top {
            let (pb, qb, pa, qa) = cur;
            if (pb, qb, pa, qa) != (pb0, qb0, pa0, qa0) {
                let mut e = 0.0;
                if pb >= pb0 {
                    e += qb;
                }
                if pb <= pb0 {
                    e -= qb0;
                }
                if pa <= pa0 {
                    e -= qa;
                }
                if pa >= pa0 {
                    e += qa0;
                }
                self.ofi += e;
            }
        }
        self.prev_top = Some(cur);
    }

    fn record_trade(&mut self, book: &L2Book, ts: i64, is_buy: bool, px: f64, qty: f64) {
        if is_buy {
            self.buy_vol += qty;
            self.n_buys += 1;
        } else {
            self.sell_vol += qty;
            self.n_sells += 1;
        }
        writeln!(
            self.trades,
            "{ts},{},{},{qty}",
            if is_buy { 1 } else { -1 },
            book.px_to_tick(px)
        )
        .unwrap();
    }
}

fn capture(book: &L2Book, ts: i64, depth_n: usize) -> Snapshot {
    Snapshot {
        ts,
        bids: book.top_n(Side::Bid, depth_n),
        asks: book.top_n(Side::Ask, depth_n),
    }
}

fn levels_json(book: &L2Book, levels: &[Level]) -> String {
    let mut s = String::from("[");
    for (i, l) in levels.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        // Emit integer ticks (exact) alongside px/qty for float-free comparison.
        let _ = write!(s, "[{},{},{}]", book.px_to_tick(l.px), l.px, l.qty);
    }
    s.push(']');
    s
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: replay <file.npz> [--snapshots ts1,ts2,...] [--depth N] [--json out.json]");
        std::process::exit(2);
    }
    let path = &args[1];
    let mut snapshot_ts: Vec<i64> = Vec::new();
    let mut depth_n: usize = 5;
    let mut json_out: Option<String> = None;
    let mut measure_dir: Option<String> = None;
    let mut grid_ms: i64 = 100;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--measure" => {
                measure_dir = Some(args[i + 1].clone());
                i += 2;
            }
            "--grid-ms" => {
                grid_ms = args[i + 1].parse().expect("bad grid-ms");
                i += 2;
            }
            "--snapshots" => {
                snapshot_ts = args[i + 1]
                    .split(',')
                    .map(|s| s.trim().parse().expect("bad snapshot ts"))
                    .collect();
                i += 2;
            }
            "--depth" => {
                depth_n = args[i + 1].parse().expect("bad depth");
                i += 2;
            }
            "--json" => {
                json_out = Some(args[i + 1].clone());
                i += 2;
            }
            a => {
                eprintln!("unknown arg: {a}");
                std::process::exit(2);
            }
        }
    }
    snapshot_ts.sort_unstable();

    let reader = NpzEventReader::open(path).expect("open npz");
    let total_rows = reader.rows();
    eprintln!("streaming {total_rows} events from {path} ...");

    let mut book = L2Book::new(TICK_SIZE, LOT_SIZE);
    let mut counts = Counts::default();
    let mut viol = Violations::default();
    let mut snapshots: Vec<Snapshot> = Vec::new();
    let mut next_snap = 0usize;
    let mut measure = measure_dir.as_deref().map(|d| Measure::new(d, grid_ms));

    let mut first_local_ts = i64::MAX;
    let mut last_local_ts = i64::MIN;
    let mut prev_local_ts = i64::MIN;
    let mut prev_exch_ts = i64::MIN;
    // Earliest timestamp in the file = hftbacktest's elapse() origin.
    let mut data_start_ts = i64::MAX;

    let t0 = Instant::now();
    for ev in reader {
        let ev = ev.expect("read event");
        let is_local = ev.ev & LOCAL_EVENT != 0;
        let kind = ev.kind();
        data_start_ts = data_start_ts.min(ev.exch_ts).min(ev.local_ts);

        if is_local {
            // Snapshot boundary: state after all LOCAL events with local_ts <= T.
            while next_snap < snapshot_ts.len() && ev.local_ts > snapshot_ts[next_snap] {
                snapshots.push(capture(&book, snapshot_ts[next_snap], depth_n));
                next_snap += 1;
            }
            if let Some(m) = measure.as_mut() {
                m.maybe_emit(&book, ev.local_ts);
            }
            if ev.local_ts < prev_local_ts {
                viol.local_ts_order += 1;
            }
            prev_local_ts = ev.local_ts;
            first_local_ts = first_local_ts.min(ev.local_ts);
            last_local_ts = last_local_ts.max(ev.local_ts);
        }
        if ev.is_exch() {
            if ev.exch_ts < prev_exch_ts {
                viol.exch_ts_order += 1;
            }
            prev_exch_ts = ev.exch_ts;
        }

        match kind {
            DEPTH_EVENT | DEPTH_SNAPSHOT_EVENT | DEPTH_BBO_EVENT => {
                match kind {
                    DEPTH_EVENT => counts.depth += 1,
                    DEPTH_SNAPSHOT_EVENT => counts.snapshot += 1,
                    _ => counts.bbo += 1,
                }
                if ev.qty < 0.0 {
                    viol.neg_qty += 1;
                }
                if !(ev.px > 0.0) {
                    viol.bad_px += 1;
                }
                if is_local {
                    let side = if ev.ev & BUY_EVENT != 0 { Side::Bid } else { Side::Ask };
                    book.set_level(side, ev.px, ev.qty, ev.local_ts);
                    counts.local_applied += 1;
                    if book.spread_ticks().is_some_and(|s| s <= 0) {
                        viol.crossed_state += 1;
                    }
                    if let Some(m) = measure.as_mut() {
                        m.update_ofi(&book);
                    }
                }
            }
            DEPTH_CLEAR_EVENT => {
                counts.clear += 1;
                if is_local {
                    let has_buy = ev.ev & BUY_EVENT != 0;
                    let has_sell = ev.ev & SELL_EVENT != 0;
                    match (has_buy, has_sell) {
                        (true, false) => book.clear_side(Side::Bid, Some(ev.px)),
                        (false, true) => book.clear_side(Side::Ask, Some(ev.px)),
                        _ => book.clear_all(),
                    }
                    counts.local_applied += 1;
                }
            }
            TRADE_EVENT => {
                counts.trade += 1;
                if is_local {
                    if let Some(m) = measure.as_mut() {
                        m.record_trade(&book, ev.local_ts, ev.ev & BUY_EVENT != 0, ev.px, ev.qty);
                    }
                }
            }
            _ => counts.other += 1,
        }
    }
    // Snapshot timestamps at/after the last event: capture final state.
    while next_snap < snapshot_ts.len() {
        snapshots.push(capture(&book, snapshot_ts[next_snap], depth_n));
        next_snap += 1;
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let eps = total_rows as f64 / elapsed;
    if let Some(m) = measure.as_mut() {
        m.samples.flush().unwrap();
        m.trades.flush().unwrap();
        println!(
            "measure: {} sample rows ({}ms grid), {} skipped one-sided",
            m.rows_written, grid_ms, m.skipped_one_sided
        );
    }

    println!("== replay report: {path} ==");
    println!("events: {total_rows}  elapsed: {elapsed:.2}s  throughput: {:.2}M events/s", eps / 1e6);
    println!(
        "local_ts span: {first_local_ts} .. {last_local_ts} ({:.1} min)",
        (last_local_ts - first_local_ts) as f64 / 60e9
    );
    println!(
        "counts: depth={} snapshot={} bbo={} clear={} trade={} other={} | applied to book: {}",
        counts.depth, counts.snapshot, counts.bbo, counts.clear, counts.trade, counts.other,
        counts.local_applied
    );
    println!(
        "violations: local_ts_order={} exch_ts_order={} neg_qty={} bad_px={} crossed_state={}",
        viol.local_ts_order, viol.exch_ts_order, viol.neg_qty, viol.bad_px, viol.crossed_state
    );
    println!(
        "book telemetry: locked_updates={} crossed_updates={} | final state {:?}, {} bid / {} ask levels",
        book.locked_updates, book.crossed_updates, book.state(),
        book.n_levels(Side::Bid), book.n_levels(Side::Ask)
    );
    if let (Some(bb), Some(ba)) = (book.best_bid(), book.best_ask()) {
        println!(
            "final top of book: bid {:.1} x {:.3} | ask {:.1} x {:.3} | spread {} tick(s)",
            bb.px, bb.qty, ba.px, ba.qty, book.spread_ticks().unwrap()
        );
    }
    for s in &snapshots {
        println!("-- snapshot @ {} --", s.ts);
        for l in s.asks.iter().rev() {
            println!("        | {:>10.1} | {:<10.3} ASK", l.px, l.qty);
        }
        for l in &s.bids {
            println!("BID {:>10.3} | {:>10.1} |", l.qty, l.px);
        }
    }

    if let Some(out) = json_out {
        let mut j = String::from("{");
        let _ = write!(
            j,
            "\"file\":\"{path}\",\"rows\":{total_rows},\"data_start_ts\":{data_start_ts},\
             \"first_local_ts\":{first_local_ts},\"last_local_ts\":{last_local_ts},\
             \"elapsed_sec\":{elapsed},\"events_per_sec\":{eps},\
             \"violations\":{{\"local_ts_order\":{},\"exch_ts_order\":{},\"neg_qty\":{},\
             \"bad_px\":{},\"crossed_state\":{}}},\
             \"book\":{{\"locked_updates\":{},\"crossed_updates\":{}}},\"snapshots\":[",
            viol.local_ts_order, viol.exch_ts_order, viol.neg_qty, viol.bad_px,
            viol.crossed_state, book.locked_updates, book.crossed_updates
        );
        for (i, s) in snapshots.iter().enumerate() {
            if i > 0 {
                j.push(',');
            }
            let _ = write!(
                j,
                "{{\"ts\":{},\"bids\":{},\"asks\":{}}}",
                s.ts,
                levels_json(&book, &s.bids),
                levels_json(&book, &s.asks)
            );
        }
        j.push_str("]}");
        std::fs::write(&out, j).expect("write json");
        eprintln!("wrote {out}");
    }
}

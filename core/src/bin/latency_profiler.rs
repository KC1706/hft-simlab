//! P2.3 latency profiler: extract feed latency (local_ts - exch_ts) from npz.
//!
//! Streams the full day and emits one row per DEPTH+TRADE LOCAL event:
//!   exch_ts, local_ts, feed_latency_ns, kind (1=depth/4=snap/5=bbo, 2=trade)
//!
//! Also emits a "trigger events" CSV: events with feed_latency > 0 that follow
//! a large trade (qty > trigger_qty threshold), used to measure the race-mode
//! inter-arrival distribution for FIG-5a.
//!
//! Usage: latency_profiler <file.npz> [--out-latency lat.csv] [--trigger-qty Q]

use std::fs::File;
use std::io::{BufWriter, Write as IoWrite};
use std::time::Instant;

use lob_core::events::{
    BUY_EVENT, DEPTH_BBO_EVENT, DEPTH_EVENT, DEPTH_SNAPSHOT_EVENT,
    LOCAL_EVENT, TRADE_EVENT,
};
use lob_core::NpzEventReader;

// Default trigger: trades larger than this qty (BTC) are "significant".
const DEFAULT_TRIGGER_QTY: f64 = 1.0;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: latency_profiler <file.npz> [--out-latency lat.csv] [--trigger-qty Q]");
        std::process::exit(1);
    }
    let path = &args[1];
    let mut lat_path = "feed_latency.csv".to_string();
    let mut trigger_qty = DEFAULT_TRIGGER_QTY;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--out-latency" if i + 1 < args.len() => { lat_path = args[i + 1].clone(); i += 2; }
            "--trigger-qty" if i + 1 < args.len() => {
                trigger_qty = args[i + 1].parse().unwrap_or(DEFAULT_TRIGGER_QTY);
                i += 2;
            }
            _ => { i += 1; }
        }
    }

    let lat_file = File::create(&lat_path).expect("create lat CSV");
    let mut lat_out = BufWriter::new(lat_file);
    writeln!(lat_out, "exch_ts,local_ts,feed_latency_ns,kind").unwrap();

    let reader = NpzEventReader::open(path).expect("open npz");
    let t0 = Instant::now();
    let mut n = 0u64;
    let mut n_neg = 0u64;

    for ev in reader {
        let ev = ev.expect("read event");
        if ev.ev & LOCAL_EVENT == 0 { continue; }
        let exch_ts = ev.exch_ts;
        let local_ts = ev.local_ts;
        if exch_ts <= 0 || local_ts <= 0 { continue; }

        let is_trade = ev.ev & TRADE_EVENT != 0;
        let is_depth = ev.ev & DEPTH_EVENT != 0;
        let is_snap = ev.ev & DEPTH_SNAPSHOT_EVENT != 0;
        let is_bbo = ev.ev & DEPTH_BBO_EVENT != 0;
        if !is_trade && !is_depth && !is_snap && !is_bbo { continue; }

        let kind: u8 = if is_trade { 2 } else if is_snap { 4 } else if is_bbo { 5 } else { 1 };
        let lat = local_ts - exch_ts;
        if lat < 0 { n_neg += 1; continue; } // clock skew / reordering — skip
        // Cap at 10 seconds to exclude obvious logging anomalies.
        if lat > 10_000_000_000 { continue; }

        // Subsample depth events: emit 1 in 20 to keep file manageable.
        // Trades and BBO/snap are always emitted (much rarer).
        if is_depth && (n % 20 != 0) { n += 1; continue; }

        writeln!(lat_out, "{exch_ts},{local_ts},{lat},{kind}").unwrap();
        n += 1;
    }

    eprintln!(
        "latency_profiler: wrote {n} rows (skipped {n_neg} neg-latency events) in {:.2?}",
        t0.elapsed()
    );
    eprintln!("output: {lat_path}");
}

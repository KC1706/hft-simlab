//! Feeds a handcrafted update sequence through the book and prints the ladder
//! after each step — watch best bid/ask, deletions, and a crossing update
//! resolve in slow motion.
//!
//! Run: cargo run --example ladder_demo

use lob_core::{L2Book, Side};

fn print_ladder(b: &L2Book, label: &str) {
    println!("\n=== {label} ===");
    let asks = b.top_n(Side::Ask, 5);
    let bids = b.top_n(Side::Bid, 5);
    for l in asks.iter().rev() {
        println!("          | {:>10.1} | {:<8.3}  ASK", l.px, l.qty);
    }
    match (b.spread_ticks(), b.mid()) {
        (Some(s), Some(m)) => println!("  --- spread {s} tick(s), mid {m:.2} ---"),
        _ => println!("  --- one-sided or empty ---"),
    }
    for l in &bids {
        println!("BID  {:>8.3} | {:>10.1} |", l.qty, l.px);
    }
    println!(
        "state: {:?} | updates: {} | locked: {} | crossed: {}",
        b.state(),
        b.depth_updates,
        b.locked_updates,
        b.crossed_updates
    );
}

fn main() {
    let mut b = L2Book::new(0.1, 0.001);

    // 1. Build a small two-sided book.
    b.set_level(Side::Bid, 50_000.0, 1.200, 1);
    b.set_level(Side::Bid, 49_999.9, 0.800, 2);
    b.set_level(Side::Bid, 49_999.7, 2.500, 3);
    b.set_level(Side::Ask, 50_000.2, 0.900, 4);
    b.set_level(Side::Ask, 50_000.4, 1.100, 5);
    b.set_level(Side::Ask, 50_000.8, 3.000, 6);
    print_ladder(&b, "1. initial two-sided book");

    // 2. Absolute update: best ask requoted smaller (NOT a delta).
    b.set_level(Side::Ask, 50_000.2, 0.150, 7);
    print_ladder(&b, "2. best ask requoted 0.9 -> 0.15 (absolute semantics)");

    // 3. Delete the best bid: next level promotes.
    b.set_level(Side::Bid, 50_000.0, 0.0, 8);
    print_ladder(&b, "3. best bid cancelled -> 49999.9 promotes");

    // 4. A bid arrives ABOVE the best ask: crossing update. The stale ask at
    //    50000.2 is skipped (kept internally until the feed refreshes it).
    b.set_level(Side::Bid, 50_000.3, 0.500, 9);
    print_ladder(&b, "4. bid 50000.3 crosses ask 50000.2 -> pointer skip");

    // 5. The feed catches up: the stale ask level is reported gone.
    b.set_level(Side::Ask, 50_000.2, 0.0, 10);
    print_ladder(&b, "5. feed refresh deletes the stale ask");
}

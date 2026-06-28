//! Integration test: stream the committed fixture npz (written by numpy, the
//! same writer that produced the real data) through the reader and the book.
//! Covers zip/npy container parsing end to end, not just the book logic.

use lob_core::events::{BUY_EVENT, DEPTH_CLEAR_EVENT, LOCAL_EVENT, TRADE_EVENT};
use lob_core::{L2Book, NpzEventReader, Side};

#[test]
fn fixture_replays_to_expected_book() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/tiny.npz");
    let reader = NpzEventReader::open(path).expect("open fixture");
    assert_eq!(reader.rows(), 10);

    let mut book = L2Book::new(0.1, 0.001);
    let mut trades = 0;
    for ev in reader {
        let ev = ev.expect("decode");
        if ev.ev & LOCAL_EVENT == 0 {
            continue;
        }
        match ev.kind() {
            TRADE_EVENT => trades += 1,
            DEPTH_CLEAR_EVENT => {
                let side = if ev.ev & BUY_EVENT != 0 { Side::Bid } else { Side::Ask };
                book.clear_side(side, Some(ev.px));
            }
            _ => {
                let side = if ev.ev & BUY_EVENT != 0 { Side::Bid } else { Side::Ask };
                book.set_level(side, ev.px, ev.qty, ev.local_ts);
            }
        }
    }

    assert_eq!(trades, 1);
    assert_eq!(book.crossed_updates, 1); // the 100.3 bid over the 100.2 ask

    // Final book: bids 100.3(0.5), 100.0(0.6), 99.9(2.0); asks 100.4(3.0).
    assert_eq!(book.best_bid_tick(), Some(1003));
    assert_eq!(book.best_ask_tick(), Some(1004));
    let bids = book.top_n(Side::Bid, 5);
    let ticks: Vec<i64> = bids.iter().map(|l| book.px_to_tick(l.px)).collect();
    assert_eq!(ticks, vec![1003, 1000, 999]);
    assert_eq!(bids[1].qty, 0.6); // trade's depth delta applied once, not twice
    let asks = book.top_n(Side::Ask, 5);
    assert_eq!(asks.len(), 1);
    assert_eq!(book.px_to_tick(asks[0].px), 1004);
}

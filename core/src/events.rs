//! Feed-event representation, binary-compatible with hftbacktest's `.npz` data
//! so the P1.2 replay harness can stream the exact files Phase 0 produced.
//!
//! Flag constants and the `Event` layout are copied (with simplification) from
//! refs/hftbacktest/hftbacktest/src/types.rs:150-331. The encoding packs the
//! event *kind* in the low byte and side/validity *attribute bits* in the high
//! bits of a single u64.

/// Event kind (low byte of `ev`): aggregate depth change at one price level.
pub const DEPTH_EVENT: u64 = 1;
/// Event kind: a trade print.
pub const TRADE_EVENT: u64 = 2;
/// Event kind: clear the book (or one side of it) up to a price.
pub const DEPTH_CLEAR_EVENT: u64 = 3;
/// Event kind: one level of a full book snapshot.
pub const DEPTH_SNAPSHOT_EVENT: u64 = 4;
/// Event kind: best-bid/offer update (some venues publish a faster BBO feed).
pub const DEPTH_BBO_EVENT: u64 = 5;

/// Attribute bit: bid-side (depth) or buyer-initiated (trade).
pub const BUY_EVENT: u64 = 1 << 29;
/// Attribute bit: ask-side (depth) or seller-initiated (trade).
pub const SELL_EVENT: u64 = 1 << 28;
/// Attribute bit: event is valid for the exchange-side processor (exch_ts).
pub const EXCH_EVENT: u64 = 1 << 31;
/// Attribute bit: event is valid for the local-side processor (local_ts).
pub const LOCAL_EVENT: u64 = 1 << 30;

/// One feed event. Field order, widths, and the 64-byte alignment match
/// hftbacktest's `Event` (types.rs:312-331) so a memory-mapped `.npz` row can be
/// reinterpreted as this struct in P1.2.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Event {
    /// Packed kind + attribute flags.
    pub ev: u64,
    /// Timestamp at the exchange (ns).
    pub exch_ts: i64,
    /// Timestamp when our collector received it (ns).
    pub local_ts: i64,
    /// Price.
    pub px: f64,
    /// Quantity (aggregate at level for depth; trade size for trades).
    pub qty: f64,
    /// Order ID (L3 feeds only; zero in our L2 data).
    pub order_id: u64,
    /// Reserved.
    pub ival: i64,
    /// Reserved.
    pub fval: f64,
}

impl Event {
    /// Event kind (low byte).
    #[inline(always)]
    pub fn kind(&self) -> u64 {
        self.ev & 0xff
    }

    #[inline(always)]
    pub fn is_buy(&self) -> bool {
        self.ev & BUY_EVENT != 0
    }

    #[inline(always)]
    pub fn is_sell(&self) -> bool {
        self.ev & SELL_EVENT != 0
    }

    #[inline(always)]
    pub fn is_local(&self) -> bool {
        self.ev & LOCAL_EVENT != 0
    }

    #[inline(always)]
    pub fn is_exch(&self) -> bool {
        self.ev & EXCH_EVENT != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_matches_hftbacktest_npz_row() {
        // hftbacktest aligns Event to 64 bytes so rows sit on cache lines; the
        // npz dtype is exactly 64 bytes wide. If this breaks, P1.2's mmap cast breaks.
        assert_eq!(std::mem::size_of::<Event>(), 64);
        assert_eq!(std::mem::align_of::<Event>(), 64);
    }

    #[test]
    fn flag_packing_round_trips() {
        let ev = Event {
            ev: DEPTH_EVENT | BUY_EVENT | LOCAL_EVENT | EXCH_EVENT,
            exch_ts: 1,
            local_ts: 2,
            px: 50_000.1,
            qty: 0.5,
            order_id: 0,
            ival: 0,
            fval: 0.0,
        };
        assert_eq!(ev.kind(), DEPTH_EVENT);
        assert!(ev.is_buy() && !ev.is_sell());
        assert!(ev.is_local() && ev.is_exch());
    }
}

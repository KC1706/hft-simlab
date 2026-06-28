//! Calibrated fill-probability model (P2.5) — the Rust side of P2.2's isotonic fit.
//!
//! P2.2 fitted, from virtual-order labeling on real BTCUSDT L2 data, twelve isotonic
//! curves `P(fill within horizon | queue_frac)` for {bid, ask} × {trade-only,
//! any-decrease} × {1 s, 10 s, 60 s}. `scripts/p25_gen_fill_tables.py` bakes them into
//! `fill_tables.rs`; this module interpolates them into the calibrated replacement for
//! hftbacktest's *uncalibrated* RiskAdverse / Prob queue models (docs/JOURNAL.md
//! Entries 5–6). The sim (P2.5) turns a returned probability into a fill draw.
//!
//! `queue_frac` is the fraction of the resting level volume *ahead* of your order at
//! placement: 0.0 = front of the queue, 1.0 = back. Fill probability is monotone
//! non-increasing in it (isotonic constraint — queue priority has only positive value).

use crate::book::Side;
use crate::fill_tables as t;

/// Which queue-advancement rule the curve was labeled under (docs/JOURNAL.md Entry 6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FillCriterion {
    /// Queue advances only on trade prints — pessimistic; matches `RiskAdverseQueueModel`.
    Trade,
    /// Queue advances on any level decrease (cancels included) — optimistic bound.
    Any,
}

/// Fill horizon the curve answers for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Horizon {
    S1,
    S10,
    S60,
}

/// Lookup over the twelve calibrated fill curves. Zero-sized: the tables are `const`.
#[derive(Clone, Copy, Debug, Default)]
pub struct CalibratedFillModel;

impl CalibratedFillModel {
    pub fn new() -> Self {
        Self
    }

    /// `P(fill within horizon | queue_frac)`, linearly interpolated on the 101-point
    /// grid. `queue_frac` is clamped to `[0, 1]`.
    pub fn p_fill(
        &self,
        side: Side,
        crit: FillCriterion,
        horizon: Horizon,
        queue_frac: f64,
    ) -> f64 {
        interp(Self::curve(side, crit, horizon), queue_frac)
    }

    fn curve(side: Side, crit: FillCriterion, horizon: Horizon) -> &'static [f64; t::N] {
        use FillCriterion::{Any, Trade};
        use Horizon::{S1, S10, S60};
        match (side, crit, horizon) {
            (Side::Bid, Trade, S1) => &t::BID_TRADE_1S,
            (Side::Bid, Trade, S10) => &t::BID_TRADE_10S,
            (Side::Bid, Trade, S60) => &t::BID_TRADE_60S,
            (Side::Bid, Any, S1) => &t::BID_ANY_1S,
            (Side::Bid, Any, S10) => &t::BID_ANY_10S,
            (Side::Bid, Any, S60) => &t::BID_ANY_60S,
            (Side::Ask, Trade, S1) => &t::ASK_TRADE_1S,
            (Side::Ask, Trade, S10) => &t::ASK_TRADE_10S,
            (Side::Ask, Trade, S60) => &t::ASK_TRADE_60S,
            (Side::Ask, Any, S1) => &t::ASK_ANY_1S,
            (Side::Ask, Any, S10) => &t::ASK_ANY_10S,
            (Side::Ask, Any, S60) => &t::ASK_ANY_60S,
        }
    }
}

/// Linear interpolation of a 101-point curve sampled on `queue_frac = i * STEP`.
fn interp(curve: &[f64; t::N], q: f64) -> f64 {
    let q = q.clamp(0.0, 1.0);
    let pos = q / t::STEP; // 0.0 ..= 100.0
    let i = pos.floor() as usize;
    if i >= t::N - 1 {
        return curve[t::N - 1];
    }
    let frac = pos - i as f64;
    curve[i] + (curve[i + 1] - curve[i]) * frac
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [(Side, FillCriterion, Horizon); 12] = [
        (Side::Bid, FillCriterion::Trade, Horizon::S1),
        (Side::Bid, FillCriterion::Trade, Horizon::S10),
        (Side::Bid, FillCriterion::Trade, Horizon::S60),
        (Side::Bid, FillCriterion::Any, Horizon::S1),
        (Side::Bid, FillCriterion::Any, Horizon::S10),
        (Side::Bid, FillCriterion::Any, Horizon::S60),
        (Side::Ask, FillCriterion::Trade, Horizon::S1),
        (Side::Ask, FillCriterion::Trade, Horizon::S10),
        (Side::Ask, FillCriterion::Trade, Horizon::S60),
        (Side::Ask, FillCriterion::Any, Horizon::S1),
        (Side::Ask, FillCriterion::Any, Horizon::S10),
        (Side::Ask, FillCriterion::Any, Horizon::S60),
    ];

    #[test]
    fn probabilities_are_in_unit_interval() {
        let m = CalibratedFillModel::new();
        for &(s, c, h) in &ALL {
            for k in 0..=100 {
                let p = m.p_fill(s, c, h, k as f64 / 100.0);
                assert!((0.0..=1.0).contains(&p), "p={p} out of range for {s:?} {c:?} {h:?}");
            }
        }
    }

    #[test]
    fn fill_prob_is_monotone_non_increasing_in_queue_frac() {
        // The isotonic constraint: being further back never helps.
        let m = CalibratedFillModel::new();
        for &(s, c, h) in &ALL {
            let mut prev = f64::INFINITY;
            for k in 0..=100 {
                let p = m.p_fill(s, c, h, k as f64 / 100.0);
                assert!(p <= prev + 1e-12, "not monotone at frac={k} for {s:?} {c:?} {h:?}");
                prev = p;
            }
        }
    }

    #[test]
    fn longer_horizon_never_lowers_fill_prob() {
        // P(fill within 60s) >= P(within 10s) >= P(within 1s) at the same position.
        let m = CalibratedFillModel::new();
        for s in [Side::Bid, Side::Ask] {
            for c in [FillCriterion::Trade, FillCriterion::Any] {
                for k in 0..=100 {
                    let q = k as f64 / 100.0;
                    let p1 = m.p_fill(s, c, Horizon::S1, q);
                    let p10 = m.p_fill(s, c, Horizon::S10, q);
                    let p60 = m.p_fill(s, c, Horizon::S60, q);
                    assert!(p10 >= p1 - 1e-12 && p60 >= p10 - 1e-12, "horizon order broke at q={q}");
                }
            }
        }
    }

    #[test]
    fn any_criterion_dominates_trade_only() {
        // Counting cancels as queue advancement can only raise fill probability.
        let m = CalibratedFillModel::new();
        for s in [Side::Bid, Side::Ask] {
            for h in [Horizon::S1, Horizon::S10, Horizon::S60] {
                for k in 0..=100 {
                    let q = k as f64 / 100.0;
                    let trade = m.p_fill(s, FillCriterion::Trade, h, q);
                    let any = m.p_fill(s, FillCriterion::Any, h, q);
                    assert!(any >= trade - 1e-9, "any < trade at q={q} {s:?} {h:?}");
                }
            }
        }
    }

    #[test]
    fn interpolates_between_grid_points() {
        let m = CalibratedFillModel::new();
        // Halfway between grid nodes 0 and 1 is the mean of the two table values.
        let c = CalibratedFillModel::curve(Side::Bid, FillCriterion::Trade, Horizon::S1);
        let mid = m.p_fill(Side::Bid, FillCriterion::Trade, Horizon::S1, 0.005);
        assert!((mid - (c[0] + c[1]) / 2.0).abs() < 1e-12);
    }

    #[test]
    fn clamps_out_of_range_queue_frac() {
        let m = CalibratedFillModel::new();
        let lo = m.p_fill(Side::Ask, FillCriterion::Any, Horizon::S10, -5.0);
        let hi = m.p_fill(Side::Ask, FillCriterion::Any, Horizon::S10, 5.0);
        let at0 = m.p_fill(Side::Ask, FillCriterion::Any, Horizon::S10, 0.0);
        let at1 = m.p_fill(Side::Ask, FillCriterion::Any, Horizon::S10, 1.0);
        assert_eq!(lo, at0);
        assert_eq!(hi, at1);
    }
}

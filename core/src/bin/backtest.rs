//! P2.5 — Unified realism backtest: the Phase-2 integration step.
//!
//! Runs the Phase-0 naive market maker through *our* book/replay, but now with the
//! three calibrated realism layers wired in and individually toggleable, so one run
//! emits the ablation table that is the project's headline measurement:
//!
//!   { naive | +fill | +fill+latency | +fill+latency+impact }
//!
//! - **fill**    (P2.2): a resting quote fills, per 1 s interval, with the calibrated
//!                probability P(fill | queue_frac) instead of the naive touch rule.
//! - **latency** (P2.3): a (re)quote decided at T is not fillable until T + entry
//!                latency drawn from the calibrated (race-aware) model.
//! - **impact**  (P2.4): each of our fills pushes a propagator impulse that shifts the
//!                mid the strategy quotes around (replay stops being a frozen movie).
//!
//! First-cut modeling choices (documented as limitations in docs/JOURNAL.md Entry 9):
//! the strategy holds at most one resting order per side; queue_frac is estimated from
//! the resting depth at the quote price; fills are evaluated on the 1 s grid against the
//! 1 s fill curve; naive fills use interval trade-through (sell-trade <= bid / buy-trade
//! >= ask). Honest enough to rank the realism layers; Phase 4 refines it.
//!
//! Usage: backtest <file.npz> [--seed N] [--grid-ms N]

use std::env;

use lob_core::book::{L2Book, Side};
use lob_core::events::{
    BUY_EVENT, DEPTH_BBO_EVENT, DEPTH_CLEAR_EVENT, DEPTH_EVENT, DEPTH_SNAPSHOT_EVENT, LOCAL_EVENT,
    SELL_EVENT, TRADE_EVENT,
};
use lob_core::fill::{CalibratedFillModel, FillCriterion, Horizon};
use lob_core::impact::PropagatorKernel;
use lob_core::latency::{calibrated_race_latency, ConstantLatency, LatencyModel};
use lob_core::NpzEventReader;

const TICK_SIZE: f64 = 0.1;
const LOT_SIZE: f64 = 0.001;

// Strategy params — identical to scripts/p0_backtest.py so the ablation is comparable.
const HALF_SPREAD_TICKS: f64 = 60.0;
const SKEW_TICKS: f64 = 10.0;
const ORDER_QTY: f64 = 0.001; // BTC
const MAX_POSITION: f64 = 0.005; // BTC
const MAKER_FEE: f64 = 0.0002; // 2 bps, post-only (GTX)
const NAIVE_LATENCY_NS: i64 = 10_000_000; // 10 ms constant baseline

/// One resting virtual order.
#[derive(Clone, Copy)]
struct Order {
    px: f64,
    active_ts: i64, // fillable only once now >= active_ts (entry latency)
    queue_frac: f64,
}

#[derive(Clone, Copy)]
struct Config {
    name: &'static str,
    calibrated_fill: bool,
    calibrated_latency: bool,
    impact: bool,
}

#[derive(Default)]
struct Stats {
    fills: u64,
    buy_fills: u64,
    sell_fills: u64,
    fees: f64,
    final_equity: f64,
    final_position: f64,
    n_quotes: u64,
}

/// Inline xorshift64* — same generator the rest of the crate uses (book.rs soak,
/// latency.rs), kept local so the binary owns its own deterministic stream.
struct Xor64(u64);
impl Xor64 {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_f64(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn run(path: &str, cfg: Config, seed: u64, grid_ns: i64) -> Stats {
    let reader = NpzEventReader::open(path).expect("open npz");
    let mut book = L2Book::new(TICK_SIZE, LOT_SIZE);
    let fill_model = CalibratedFillModel::new();
    let mut latency: Box<dyn LatencyModel> = if cfg.calibrated_latency {
        Box::new(calibrated_race_latency(seed))
    } else {
        Box::new(ConstantLatency::new(NAIVE_LATENCY_NS, NAIVE_LATENCY_NS))
    };
    let mut kernel = PropagatorKernel::calibrated();
    let mut rng = Xor64::new(seed ^ 0x9E37_79B9_7F4A_7C15);

    let mut cash = 0.0_f64;
    let mut position = 0.0_f64;
    let mut bid: Option<Order> = None;
    let mut ask: Option<Order> = None;
    let mut stats = Stats::default();

    // Per-interval trade-through extremes for the naive fill rule.
    let mut min_sell_px = f64::INFINITY;
    let mut max_buy_px = f64::NEG_INFINITY;
    let mut next_grid = i64::MIN;
    let mut last_mid = f64::NAN;

    for ev in reader {
        let ev = ev.expect("read event");
        let is_local = ev.ev & LOCAL_EVENT != 0;
        if !is_local {
            continue;
        }
        let kind = ev.kind();
        let now = ev.local_ts;

        // ── settle the just-ended grid interval(s) ───────────────────────────
        if next_grid == i64::MIN {
            next_grid = now.div_euclid(grid_ns) * grid_ns + grid_ns;
        }
        while now > next_grid {
            let t = next_grid;
            settle_fills(
                t, &cfg, &fill_model, &mut rng, &mut kernel, &mut bid, &mut ask, &mut position,
                &mut cash, &mut stats, min_sell_px, max_buy_px,
            );
            requote(
                t, &cfg, &book, &mut latency, &mut kernel, &mut bid, &mut ask, position, &mut stats,
            );
            min_sell_px = f64::INFINITY;
            max_buy_px = f64::NEG_INFINITY;
            next_grid += grid_ns;
        }

        // ── apply the event to the book (mirrors bin/replay.rs) ──────────────
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
            TRADE_EVENT => {
                if ev.ev & BUY_EVENT != 0 {
                    max_buy_px = max_buy_px.max(ev.px);
                } else {
                    min_sell_px = min_sell_px.min(ev.px);
                }
                if cfg.calibrated_latency {
                    latency.on_trigger(now); // a trade is a race trigger (P2.3)
                }
            }
            _ => {}
        }
        if let Some(m) = book.mid() {
            last_mid = m;
        }
    }

    let final_mid = last_mid;
    stats.final_position = position;
    stats.final_equity = cash + position * final_mid;
    stats
}

#[allow(clippy::too_many_arguments)]
fn settle_fills(
    now: i64,
    cfg: &Config,
    model: &CalibratedFillModel,
    rng: &mut Xor64,
    kernel: &mut PropagatorKernel,
    bid: &mut Option<Order>,
    ask: &mut Option<Order>,
    position: &mut f64,
    cash: &mut f64,
    stats: &mut Stats,
    min_sell_px: f64,
    max_buy_px: f64,
) {
    if let Some(o) = *bid {
        if now >= o.active_ts {
            let filled = if cfg.calibrated_fill {
                rng.next_f64() < model.p_fill(Side::Bid, FillCriterion::Any, Horizon::S1, o.queue_frac)
            } else {
                min_sell_px <= o.px // naive: a sell traded through our bid
            };
            if filled {
                *position += ORDER_QTY;
                *cash -= o.px * ORDER_QTY;
                let fee = MAKER_FEE * o.px * ORDER_QTY;
                *cash -= fee;
                stats.fees += fee;
                stats.fills += 1;
                stats.buy_fills += 1;
                if cfg.impact {
                    kernel.push(now, 1.0, ORDER_QTY);
                }
                *bid = None;
            }
        }
    }
    if let Some(o) = *ask {
        if now >= o.active_ts {
            let filled = if cfg.calibrated_fill {
                rng.next_f64() < model.p_fill(Side::Ask, FillCriterion::Any, Horizon::S1, o.queue_frac)
            } else {
                max_buy_px >= o.px // naive: a buy traded through our ask
            };
            if filled {
                *position -= ORDER_QTY;
                *cash += o.px * ORDER_QTY;
                let fee = MAKER_FEE * o.px * ORDER_QTY;
                *cash -= fee;
                stats.fees += fee;
                stats.fills += 1;
                stats.sell_fills += 1;
                if cfg.impact {
                    kernel.push(now, -1.0, ORDER_QTY);
                }
                *ask = None;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn requote(
    now: i64,
    cfg: &Config,
    book: &L2Book,
    latency: &mut Box<dyn LatencyModel>,
    kernel: &mut PropagatorKernel,
    bid: &mut Option<Order>,
    ask: &mut Option<Order>,
    position: f64,
    stats: &mut Stats,
) {
    let Some(mid0) = book.mid() else { return };
    // Impact shifts the price the strategy quotes around (kernel returns ticks).
    let eff_mid = if cfg.impact { mid0 + kernel.impact(now) * TICK_SIZE } else { mid0 };
    let skew = (position / MAX_POSITION) * SKEW_TICKS * TICK_SIZE;
    let half = HALF_SPREAD_TICKS * TICK_SIZE;

    let entry = latency.entry_ns(now);
    let active_ts = now + entry;

    if position < MAX_POSITION {
        let bid_px = ((eff_mid - half - skew) / TICK_SIZE).floor() * TICK_SIZE;
        if bid_px.is_finite() && bid_px > 0.0 {
            let depth = book.qty_at_tick(Side::Bid, book.px_to_tick(bid_px));
            let queue_frac = depth / (depth + ORDER_QTY); // ~1 behind real depth, 0 if empty
            *bid = Some(Order { px: bid_px, active_ts, queue_frac });
            stats.n_quotes += 1;
        }
    } else {
        *bid = None;
    }
    if position > -MAX_POSITION {
        let ask_px = ((eff_mid + half - skew) / TICK_SIZE).ceil() * TICK_SIZE;
        if ask_px.is_finite() && ask_px > 0.0 {
            let depth = book.qty_at_tick(Side::Ask, book.px_to_tick(ask_px));
            let queue_frac = depth / (depth + ORDER_QTY);
            *ask = Some(Order { px: ask_px, active_ts, queue_frac });
            stats.n_quotes += 1;
        }
    } else {
        *ask = None;
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: backtest <file.npz> [--seed N] [--grid-ms N]");
        std::process::exit(2);
    }
    let path = &args[1];
    let mut seed = 42u64;
    let mut grid_ms = 1000i64;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                seed = args[i + 1].parse().expect("bad seed");
                i += 2;
            }
            "--grid-ms" => {
                grid_ms = args[i + 1].parse().expect("bad grid-ms");
                i += 2;
            }
            a => {
                eprintln!("unknown arg: {a}");
                std::process::exit(2);
            }
        }
    }
    let grid_ns = grid_ms * 1_000_000;

    let configs = [
        Config { name: "naive", calibrated_fill: false, calibrated_latency: false, impact: false },
        Config { name: "+fill", calibrated_fill: true, calibrated_latency: false, impact: false },
        Config { name: "+fill+latency", calibrated_fill: true, calibrated_latency: true, impact: false },
        Config { name: "+fill+latency+impact", calibrated_fill: true, calibrated_latency: true, impact: true },
    ];

    println!("HFT-SimLab P2.5 ablation backtest — {path}");
    println!("PnL is quote-currency (USDT) mark-to-market: cash + position * last_mid.");
    println!(
        "{:>22} | {:>7} | {:>7} | {:>7} | {:>12} | {:>8} | {:>8}",
        "config", "fills", "buys", "sells", "pnl(USDT)", "fees", "end_pos"
    );
    println!("{}", "-".repeat(82));
    for cfg in configs {
        let s = run(path, cfg, seed, grid_ns);
        println!(
            "{:>22} | {:>7} | {:>7} | {:>7} | {:>12.4} | {:>8.4} | {:>8.4}",
            cfg.name, s.fills, s.buy_fills, s.sell_fills, s.final_equity, s.fees, s.final_position
        );
    }
    println!(
        "\nnote: at a {grid_ms}ms decision grid the calibrated latency layer is a no-op vs +fill —\n\
         ms-scale entry latency never crosses a 1s boundary, so it cannot change which interval\n\
         an order becomes fillable in. Latency acts at the sub-second race scale (JOURNAL Entry 7);\n\
         exposing its PnL effect needs event-level (not grid-level) fill evaluation — a Phase 4 refinement."
    );
}

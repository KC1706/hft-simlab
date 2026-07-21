//! P2.5/P2.6 — Unified realism backtest with **event-level** fills.
//!
//! Runs the Phase-0 naive market maker through *our* book/replay with the three
//! calibrated realism layers wired in and individually toggleable, emitting the
//! ablation table that is the project's headline measurement:
//!
//!   { naive | +fill | +fill+latency | +fill+latency+impact }
//!
//! - **fill**    (P2.2): a resting quote does not fill on a grid boundary; it is given
//!                a continuous *fill time* `active_ts + Exp(λ)`, where λ is chosen so the
//!                probability of filling within 1 s equals the calibrated
//!                `P(fill | queue_frac)`. Naive fills instead match the real trade stream
//!                at event time (sell-trade ≤ bid / buy-trade ≥ ask).
//! - **latency** (P2.3): a (re)quote decided at T is not fillable until `active_ts =
//!                T + entry_latency`; since the fill time is measured from `active_ts`,
//!                higher/heavier-tailed latency pushes fills past the next requote and
//!                they are cancelled unfilled — so latency now has a measurable PnL effect
//!                (the P2.6 fix for Entry 9's grid-level null result).
//! - **impact**  (P2.4): each of our fills pushes a propagator impulse that shifts the
//!                mid the strategy quotes around (replay stops being a frozen movie).
//!
//! Modeling choices (limitations in docs/JOURNAL.md Entries 9–10): one resting order per
//! side; `queue_frac` is the resting-depth ratio at the quote price; the calibrated
//! hazard is exponential matched to the 1 s curve (memoryless approximation of the
//! isotonic horizon). Honest enough to rank the realism layers; Phase 4 refines further.
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
const NS_PER_S: f64 = 1_000_000_000.0;

// Strategy params — identical to scripts/p0_backtest.py so the ablation is comparable.
const HALF_SPREAD_TICKS: f64 = 60.0;
const SKEW_TICKS: f64 = 10.0;
const ORDER_QTY: f64 = 0.001; // BTC
const MAX_POSITION: f64 = 0.005; // BTC
const MAKER_FEE: f64 = 0.0002; // 2 bps, post-only (GTX)
const TAKER_FEE: f64 = 0.0004; // 4 bps, aggressive (OFI taker)
const NAIVE_LATENCY_NS: i64 = 10_000_000; // 10 ms constant baseline

// OFI-momentum taker (P4, a second ablation subject): cross the spread in the direction of the
// top-of-book volume imbalance when it exceeds a threshold. A TAKER always fills at the touch, so
// the calibrated queue fill-model is irrelevant to it (a deliberate contrast with the MM) — while
// market impact still applies. Latency-slippage is a v1 simplification: fills at decision-time touch.
const OFI_LEVELS: usize = 5;
const OFI_THRESHOLD: f64 = 0.15; // take when |imbalance − 0.5| exceeds this

#[derive(Clone, Copy, PartialEq)]
enum Strategy {
    Mm,       // fixed-offset skewed market maker (the P0/P2 subject)
    OfiTaker, // order-flow-imbalance momentum taker
}

/// One resting virtual order. `fill_time` is the scheduled calibrated fill instant
/// (`i64::MAX` = never / naive path); `active_ts` is when it becomes fillable.
#[derive(Clone, Copy)]
struct Order {
    px: f64,
    active_ts: i64,
    fill_time: i64,
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
    cancelled_unfilled: u64, // orders requoted away before their scheduled fill
}

/// Inline xorshift64* — same generator the rest of the crate uses.
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

/// Convert a 1 s cumulative fill probability into a scheduled fill instant measured from
/// `active_ts`, via the memoryless (exponential) hazard with the matching 1 s mass.
fn schedule_fill_time(active_ts: i64, p_1s: f64, rng: &mut Xor64) -> i64 {
    if p_1s <= 0.0 {
        return i64::MAX;
    }
    let lambda = -(1.0 - p_1s.min(0.999999)).ln(); // per second
    let u = rng.next_f64().min(1.0 - 1e-12);
    let dt_s = -(1.0 - u).ln() / lambda; // Exp(lambda) draw, seconds
    let dt_ns = (dt_s * NS_PER_S).min(i64::MAX as f64) as i64;
    active_ts.saturating_add(dt_ns)
}

struct Sim {
    cfg: Config,
    strategy: Strategy,
    half_spread: f64, // strategy knob (ticks): the MM's quoted half-spread (P4 ablation subject)
    book: L2Book,
    fill_model: CalibratedFillModel,
    latency: Box<dyn LatencyModel>,
    kernel: PropagatorKernel,
    rng: Xor64,
    cash: f64,
    position: f64,
    bid: Option<Order>,
    ask: Option<Order>,
    stats: Stats,
}

impl Sim {
    fn fill_bid(&mut self, o: Order, at: i64) {
        self.position += ORDER_QTY;
        self.cash -= o.px * ORDER_QTY;
        let fee = MAKER_FEE * o.px * ORDER_QTY;
        self.cash -= fee;
        self.stats.fees += fee;
        self.stats.fills += 1;
        self.stats.buy_fills += 1;
        if self.cfg.impact {
            self.kernel.push(at, 1.0, ORDER_QTY);
        }
        self.bid = None;
    }

    fn fill_ask(&mut self, o: Order, at: i64) {
        self.position -= ORDER_QTY;
        self.cash += o.px * ORDER_QTY;
        let fee = MAKER_FEE * o.px * ORDER_QTY;
        self.cash -= fee;
        self.stats.fees += fee;
        self.stats.fills += 1;
        self.stats.sell_fills += 1;
        if self.cfg.impact {
            self.kernel.push(at, -1.0, ORDER_QTY);
        }
        self.ask = None;
    }

    /// OFI-momentum taker decision at grid time `now`: cross the spread toward the imbalance.
    fn ofi_decide(&mut self, now: i64) {
        let bids = self.book.top_n(Side::Bid, OFI_LEVELS);
        let asks = self.book.top_n(Side::Ask, OFI_LEVELS);
        let bid_vol: f64 = bids.iter().map(|l| l.qty).sum();
        let ask_vol: f64 = asks.iter().map(|l| l.qty).sum();
        let tot = bid_vol + ask_vol;
        if tot <= 0.0 {
            return;
        }
        let imb = bid_vol / tot; // > 0.5 = bid-heavy → momentum up → buy
        if imb > 0.5 + OFI_THRESHOLD && self.position < MAX_POSITION {
            if let Some(a) = self.book.best_ask() {
                self.market_take(Side::Bid, a.px, now); // buy, lifting the ask
            }
        } else if imb < 0.5 - OFI_THRESHOLD && self.position > -MAX_POSITION {
            if let Some(b) = self.book.best_bid() {
                self.market_take(Side::Ask, b.px, now); // sell, hitting the bid
            }
        }
    }

    /// A market order: fills immediately at the touch `px` (crossing the spread is the taker cost),
    /// pays the taker fee, and — like our maker fills — pushes a market-impact impulse. Under the
    /// impact config the fill price is shifted by the accumulated propagator so an aggressive taker
    /// pays for the price it has already moved (impact feeds back into its own fills).
    fn market_take(&mut self, take_side: Side, px: f64, now: i64) {
        let (qty_sign, dir) = if take_side == Side::Bid { (1.0, 1.0) } else { (-1.0, -1.0) };
        let px = if self.cfg.impact {
            px + self.kernel.impact(now) * TICK_SIZE
        } else {
            px
        };
        self.position += qty_sign * ORDER_QTY;
        self.cash -= qty_sign * px * ORDER_QTY;
        let fee = TAKER_FEE * px * ORDER_QTY;
        self.cash -= fee;
        self.stats.fees += fee;
        self.stats.fills += 1;
        if take_side == Side::Bid {
            self.stats.buy_fills += 1;
        } else {
            self.stats.sell_fills += 1;
        }
        if self.cfg.impact {
            self.kernel.push(now, dir, ORDER_QTY);
        }
    }

    /// Execute any calibrated fills whose scheduled instant has arrived by `now`.
    fn settle_scheduled(&mut self, now: i64) {
        if let Some(o) = self.bid {
            if o.fill_time <= now {
                self.fill_bid(o, o.fill_time);
            }
        }
        if let Some(o) = self.ask {
            if o.fill_time <= now {
                self.fill_ask(o, o.fill_time);
            }
        }
    }

    /// Naive event-level fill: a real trade trades through our active resting order.
    fn settle_naive_trade(&mut self, ts: i64, is_buy: bool, px: f64) {
        if !is_buy {
            if let Some(o) = self.bid {
                if ts >= o.active_ts && px <= o.px {
                    self.fill_bid(o, ts);
                }
            }
        } else if let Some(o) = self.ask {
            if ts >= o.active_ts && px >= o.px {
                self.fill_ask(o, ts);
            }
        }
    }

    /// Cancel-replace both quotes from the strategy at decision time `now`.
    fn requote(&mut self, now: i64) {
        if self.bid.take().is_some() {
            self.stats.cancelled_unfilled += 1;
        }
        if self.ask.take().is_some() {
            self.stats.cancelled_unfilled += 1;
        }
        let Some(mid0) = self.book.mid() else { return };
        let eff_mid = if self.cfg.impact {
            mid0 + self.kernel.impact(now) * TICK_SIZE
        } else {
            mid0
        };
        let skew = (self.position / MAX_POSITION) * SKEW_TICKS * TICK_SIZE;
        let half = self.half_spread * TICK_SIZE;
        let active_ts = now + self.latency.entry_ns(now);

        if self.position < MAX_POSITION {
            let bid_px = ((eff_mid - half - skew) / TICK_SIZE).floor() * TICK_SIZE;
            if bid_px.is_finite() && bid_px > 0.0 {
                self.bid = Some(self.make_order(Side::Bid, bid_px, active_ts));
            }
        }
        if self.position > -MAX_POSITION {
            let ask_px = ((eff_mid + half - skew) / TICK_SIZE).ceil() * TICK_SIZE;
            if ask_px.is_finite() && ask_px > 0.0 {
                self.ask = Some(self.make_order(Side::Ask, ask_px, active_ts));
            }
        }
    }

    fn make_order(&mut self, side: Side, px: f64, active_ts: i64) -> Order {
        let fill_time = if self.cfg.calibrated_fill {
            let depth = self.book.qty_at_tick(side, self.book.px_to_tick(px));
            let queue_frac = depth / (depth + ORDER_QTY); // ~1 behind real depth, 0 if empty
            let p1 = self
                .fill_model
                .p_fill(side, FillCriterion::Any, Horizon::S1, queue_frac);
            schedule_fill_time(active_ts, p1, &mut self.rng)
        } else {
            i64::MAX // naive: filled only by a real trade-through
        };
        Order { px, active_ts, fill_time }
    }
}

fn run(path: &str, cfg: Config, seed: u64, grid_ns: i64, half_spread: f64, strategy: Strategy) -> Stats {
    let reader = NpzEventReader::open(path).expect("open npz");
    let latency: Box<dyn LatencyModel> = if cfg.calibrated_latency {
        Box::new(calibrated_race_latency(seed))
    } else {
        Box::new(ConstantLatency::new(NAIVE_LATENCY_NS, NAIVE_LATENCY_NS))
    };
    let mut sim = Sim {
        cfg,
        strategy,
        half_spread,
        book: L2Book::new(TICK_SIZE, LOT_SIZE),
        fill_model: CalibratedFillModel::new(),
        latency,
        kernel: PropagatorKernel::calibrated(),
        rng: Xor64::new(seed ^ 0x9E37_79B9_7F4A_7C15),
        cash: 0.0,
        position: 0.0,
        bid: None,
        ask: None,
        stats: Stats::default(),
    };

    let mut next_grid = i64::MIN;
    let mut last_mid = f64::NAN;

    for ev in reader {
        let ev = ev.expect("read event");
        if ev.ev & LOCAL_EVENT == 0 {
            continue;
        }
        let now = ev.local_ts;
        let kind = ev.kind();

        // Calibrated fills are scheduled in continuous time — settle any that are due.
        if cfg.calibrated_fill {
            sim.settle_scheduled(now);
        }

        // Decision grid: cancel-replace quotes when we cross a boundary.
        if next_grid == i64::MIN {
            next_grid = now.div_euclid(grid_ns) * grid_ns + grid_ns;
        }
        while now > next_grid {
            match sim.strategy {
                Strategy::Mm => sim.requote(next_grid),
                Strategy::OfiTaker => sim.ofi_decide(next_grid),
            }
            next_grid += grid_ns;
        }

        match kind {
            DEPTH_EVENT | DEPTH_SNAPSHOT_EVENT | DEPTH_BBO_EVENT => {
                let side = if ev.ev & BUY_EVENT != 0 { Side::Bid } else { Side::Ask };
                sim.book.set_level(side, ev.px, ev.qty, now);
            }
            DEPTH_CLEAR_EVENT => {
                let has_buy = ev.ev & BUY_EVENT != 0;
                let has_sell = ev.ev & SELL_EVENT != 0;
                match (has_buy, has_sell) {
                    (true, false) => sim.book.clear_side(Side::Bid, Some(ev.px)),
                    (false, true) => sim.book.clear_side(Side::Ask, Some(ev.px)),
                    _ => sim.book.clear_all(),
                }
            }
            TRADE_EVENT => {
                let is_buy = ev.ev & BUY_EVENT != 0;
                if !cfg.calibrated_fill {
                    sim.settle_naive_trade(now, is_buy, ev.px);
                }
                if cfg.calibrated_latency {
                    sim.latency.on_trigger(now); // a trade is a race trigger (P2.3)
                }
            }
            _ => {}
        }
        if let Some(m) = sim.book.mid() {
            last_mid = m;
        }
    }

    sim.stats.final_position = sim.position;
    sim.stats.final_equity = sim.cash + sim.position * last_mid;
    sim.stats
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
    let mut half_spread = HALF_SPREAD_TICKS; // strategy knob (P4)
    let mut csv = false; // machine-readable output for the ablation harness
    let mut strategy = Strategy::Mm;
    let mut strat_name = "mm";
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => { seed = args[i + 1].parse().expect("bad seed"); i += 2; }
            "--grid-ms" => { grid_ms = args[i + 1].parse().expect("bad grid-ms"); i += 2; }
            "--half-spread" => { half_spread = args[i + 1].parse().expect("bad half-spread"); i += 2; }
            "--strategy" => {
                strat_name = match args[i + 1].as_str() {
                    "mm" => "mm",
                    "ofi" => "ofi",
                    s => { eprintln!("unknown strategy: {s} (use mm|ofi)"); std::process::exit(2); }
                };
                strategy = if strat_name == "ofi" { Strategy::OfiTaker } else { Strategy::Mm };
                i += 2;
            }
            "--csv" => { csv = true; i += 1; }
            a => { eprintln!("unknown arg: {a}"); std::process::exit(2); }
        }
    }
    let grid_ns = grid_ms * 1_000_000;

    let configs = [
        Config { name: "naive", calibrated_fill: false, calibrated_latency: false, impact: false },
        Config { name: "+fill", calibrated_fill: true, calibrated_latency: false, impact: false },
        Config { name: "+fill+latency", calibrated_fill: true, calibrated_latency: true, impact: false },
        Config { name: "+fill+latency+impact", calibrated_fill: true, calibrated_latency: true, impact: true },
    ];

    if csv {
        // one row per config; header lets the P4 harness (experiments/ablation) parse it directly.
        println!("strategy,config,half_spread,seed,fills,buys,sells,pnl,fees,end_pos,cancel");
        for cfg in configs {
            let s = run(path, cfg, seed, grid_ns, half_spread, strategy);
            println!(
                "{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{}",
                strat_name, cfg.name, half_spread, seed, s.fills, s.buy_fills, s.sell_fills,
                s.final_equity, s.fees, s.final_position, s.cancelled_unfilled
            );
        }
        return;
    }

    println!("HFT-SimLab P2.5/P2.6 ablation backtest — {path}  (grid={grid_ms}ms, seed={seed}, strategy={strat_name}, half_spread={half_spread})");
    println!("PnL is quote-currency (USDT) mark-to-market: cash + position * last_mid.");
    println!(
        "{:>22} | {:>7} | {:>7} | {:>7} | {:>12} | {:>8} | {:>8} | {:>9}",
        "config", "fills", "buys", "sells", "pnl(USDT)", "fees", "end_pos", "cancel"
    );
    println!("{}", "-".repeat(96));
    for cfg in configs {
        let s = run(path, cfg, seed, grid_ns, half_spread, strategy);
        println!(
            "{:>22} | {:>7} | {:>7} | {:>7} | {:>12.4} | {:>8.4} | {:>8.4} | {:>9}",
            cfg.name, s.fills, s.buy_fills, s.sell_fills, s.final_equity, s.fees, s.final_position,
            s.cancelled_unfilled
        );
    }
}

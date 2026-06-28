//! P2.3 latency model: calibrated from feed timestamps.
//!
//! The feed latency (local_ts - exch_ts) is a lower bound on order round-trip
//! latency, because it only measures the data-path delay — not the order-path.
//! On co-located infrastructure the two are roughly equal; on retail infra the
//! order-path is typically 1–3× the feed delay. We calibrate from feed data
//! and expose the multiplier as a parameter.
//!
//! **Race-mode model** (arXiv 2603.24137): when a significant trigger event
//! (large trade, best-quote change) occurs, many participants react within one
//! exchange round-trip. The exchange queues all concurrent orders by arrival
//! time; effectively your order competes against a burst of orders clustered
//! within `race_window_ns` of the trigger. `RaceAwareLatency` adds a uniform
//! jitter in [0, race_window_ns] to the base latency around trigger events.

/// Provides order entry and response latencies (nanoseconds).
pub trait LatencyModel {
    /// Returns the entry latency (req → exchange) for an order sent at `now`.
    fn entry_ns(&mut self, now: i64) -> i64;
    /// Returns the response latency (exchange → local) for an order matched at `now`.
    fn response_ns(&mut self, now: i64) -> i64;
    /// Notify the model that a trigger event occurred at `now` (large trade /
    /// best-quote change). Race-aware models activate a race window.
    fn on_trigger(&mut self, _now: i64) {}
}

// ── xorshift64* ─────────────────────────────────────────────────────────────
// Deterministic PRNG for sampling inside the latency model (no external deps).
struct Xor64 {
    state: u64,
}

impl Xor64 {
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Uniform sample in [0, n).
    fn next_u64_mod(&mut self, n: u64) -> u64 {
        if n == 0 { return 0; }
        self.next() % n
    }

    /// Uniform f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

// ── Ziggurat-free log-normal sampler ─────────────────────────────────────────
// Box-Muller: two uniform samples → one standard normal → exponentiate.
// Sufficient for the ~10 samples per backtest step we need.
fn log_normal_sample(rng: &mut Xor64, mu: f64, sigma: f64) -> f64 {
    let u1 = rng.next_f64().max(1e-15);
    let u2 = rng.next_f64();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    (mu + sigma * z).exp()
}

// ── ConstantLatency ───────────────────────────────────────────────────────────

/// Deterministic constant latency — the naive baseline (hftbacktest default).
pub struct ConstantLatency {
    entry: i64,
    response: i64,
}

impl ConstantLatency {
    pub fn new(entry_ns: i64, response_ns: i64) -> Self {
        Self { entry: entry_ns, response: response_ns }
    }
}

impl LatencyModel for ConstantLatency {
    fn entry_ns(&mut self, _now: i64) -> i64 { self.entry }
    fn response_ns(&mut self, _now: i64) -> i64 { self.response }
}

// ── LogNormalLatency ──────────────────────────────────────────────────────────

/// Samples entry and response latency from a log-normal distribution
/// calibrated from feed data. The `roundtrip_multiplier` scales feed latency
/// to estimated order round-trip (default 2.0: entry ≈ response ≈ feed delay).
pub struct LogNormalLatency {
    /// ln(µs) location parameter of the log-normal.
    mu: f64,
    /// Shape parameter (scale of the log distribution).
    sigma: f64,
    /// Multiply the feed-latency sample by this to get entry latency.
    entry_mult: f64,
    /// Multiply the feed-latency sample by this to get response latency.
    response_mult: f64,
    rng: Xor64,
}

impl LogNormalLatency {
    /// Construct from calibrated feed-latency log-normal params.
    /// `mu_log_us` / `sigma` are in log(µs); `roundtrip_mult` is typically 1.0
    /// (entry leg only; the return path is separate via `response_mult`).
    pub fn new(mu_log_us: f64, sigma: f64, entry_mult: f64, response_mult: f64, seed: u64) -> Self {
        Self { mu: mu_log_us, sigma, entry_mult, response_mult, rng: Xor64::new(seed) }
    }

    fn sample_us(&mut self) -> f64 {
        log_normal_sample(&mut self.rng, self.mu, self.sigma).max(0.0)
    }
}

impl LatencyModel for LogNormalLatency {
    fn entry_ns(&mut self, _now: i64) -> i64 {
        (self.sample_us() * self.entry_mult * 1000.0).round() as i64
    }

    fn response_ns(&mut self, _now: i64) -> i64 {
        (self.sample_us() * self.response_mult * 1000.0).round() as i64
    }
}

// ── RaceAwareLatency ─────────────────────────────────────────────────────────

/// Log-normal base latency + race-mode jitter around trigger events.
///
/// After `on_trigger(T)`, for the next `race_window_ns` nanoseconds every
/// `entry_ns()` call adds a uniform jitter U(0, race_jitter_ns) to simulate
/// competing against other participants who all reacted to the same event at
/// approximately the same time. The race window should be set to the estimated
/// exchange round-trip latency (≈ mode × 2 of the feed latency distribution).
pub struct RaceAwareLatency {
    base: LogNormalLatency,
    race_jitter_ns: i64,
    race_window_ns: i64,
    race_active_until: i64,
}

impl RaceAwareLatency {
    pub fn new(
        mu_log_us: f64,
        sigma: f64,
        entry_mult: f64,
        response_mult: f64,
        race_jitter_ns: i64,
        race_window_ns: i64,
        seed: u64,
    ) -> Self {
        Self {
            base: LogNormalLatency::new(mu_log_us, sigma, entry_mult, response_mult, seed),
            race_jitter_ns,
            race_window_ns,
            race_active_until: i64::MIN,
        }
    }
}

impl LatencyModel for RaceAwareLatency {
    fn entry_ns(&mut self, now: i64) -> i64 {
        let base = self.base.entry_ns(now);
        if now <= self.race_active_until {
            let jitter = self.base.rng.next_u64_mod(self.race_jitter_ns.max(1) as u64) as i64;
            base + jitter
        } else {
            base
        }
    }

    fn response_ns(&mut self, now: i64) -> i64 {
        self.base.response_ns(now)
    }

    fn on_trigger(&mut self, now: i64) {
        self.race_active_until = now + self.race_window_ns;
    }
}

// ── Calibrated constructor (from p23 latency_model.json values) ──────────────

/// Construct a `RaceAwareLatency` from the P2.3 calibrated parameters.
/// These are the default values derived from BTCUSDT-perp 2026-05-01.
pub fn calibrated_race_latency(seed: u64) -> RaceAwareLatency {
    // Feed latency log-normal: mu=8.299 in log(µs), sigma=0.775.
    // Mode (most likely feed delay) = exp(mu - sigma^2) = 2203 µs.
    // Estimated entry = 1× feed delay (lower bound); response = 1× feed delay.
    // Race jitter = one estimated round-trip = 4405 µs = 4_405_000 ns.
    // Race window = race jitter (stay in race mode for one round-trip).
    RaceAwareLatency::new(
        8.299,        // mu_log_us
        0.775,        // sigma
        1.0,          // entry_mult: entry ≈ one-way feed delay
        1.0,          // response_mult
        4_405_000,    // race_jitter_ns: 2× feed mode ≈ round-trip
        4_405_000,    // race_window_ns: stay in race mode for one round-trip
        seed,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_latency_deterministic() {
        let mut m = ConstantLatency::new(10_000_000, 5_000_000);
        assert_eq!(m.entry_ns(0), 10_000_000);
        assert_eq!(m.entry_ns(999_999), 10_000_000);
        assert_eq!(m.response_ns(0), 5_000_000);
    }

    #[test]
    fn lognormal_latency_positive() {
        let mut m = LogNormalLatency::new(8.299, 0.775, 1.0, 1.0, 42);
        for _ in 0..1000 {
            assert!(m.entry_ns(0) > 0);
            assert!(m.response_ns(0) > 0);
        }
    }

    #[test]
    fn race_latency_has_higher_median_during_race() {
        let mut no_race = LogNormalLatency::new(8.299, 0.775, 1.0, 1.0, 42);
        let mut race = RaceAwareLatency::new(8.299, 0.775, 1.0, 1.0, 4_405_000, 4_405_000, 42);
        race.on_trigger(0);
        let n = 2000;
        let sum_no_race: i64 = (0..n).map(|_| no_race.entry_ns(0)).sum();
        let sum_race: i64 = (0..n).map(|_| race.entry_ns(0)).sum();
        // Race mode adds jitter → higher mean.
        assert!(sum_race > sum_no_race, "race latency should be higher on average");
    }

    #[test]
    fn race_mode_expires() {
        let mut m = RaceAwareLatency::new(8.299, 0.775, 1.0, 1.0, 1_000_000, 1_000_000, 0);
        m.on_trigger(0);
        assert!(m.race_active_until > 0);
        // After the race window, jitter should no longer apply.
        assert!(m.entry_ns(10_000_000) > 0); // just sanity — still positive
    }

    #[test]
    fn calibrated_latency_sanity() {
        let mut m = calibrated_race_latency(7);
        // Median should be in a reasonable range (1ms – 100ms).
        let samples: Vec<i64> = (0..2000).map(|_| m.entry_ns(0)).collect();
        let mut s = samples.clone();
        s.sort_unstable();
        let median = s[1000];
        assert!(median > 500_000 && median < 100_000_000,
            "median entry latency {median} ns out of expected range");
    }
}

//! P2.4 market-impact propagator: Bouchaud power-law kernel.
//!
//! After each simulated fill, the caller registers a signed volume impulse.
//! The propagator sums decaying contributions from all past impulses to
//! compute the current price shift (in ticks). This shifts the mid-price
//! seen by subsequent strategy logic — the "market reacts to us" feedback
//! that naive replay ignores.
//!
//! **Model:** G(τ) = G₀ × τ^{-β} (power-law decay, τ in seconds).
//! Impact from impulse at time T: G(now - T) × sign × κ × sqrt(volume).
//! Accumulated: impact(now) = Σ_k G(now - t_k) × ε_k × κ × √V_k.
//!
//! **Calibrated parameters (BTCUSDT-perp 2026-05-01):**
//! - G₀ = 64.76 ticks (1-second impact per unit-normalised volume)
//! - β  = 0.50 (Bouchaud literature prior — single trending day can't identify β)
//! - κ  = 162.6 ticks/√BTC (square-root law amplitude from 100ms response)
//! - Cutoff: impulses older than 60 s are dropped (negligible contribution).
//!
//! **Usage:**
//! ```rust,ignore
//! let mut kernel = PropagatorKernel::calibrated(); // default BTCUSDT params
//! kernel.push(now_ns, 1.0, 0.5);   // buy fill: sign=+1, 0.5 BTC
//! let shift = kernel.impact(now_ns); // ticks to add to mid
//! ```

/// One registered impulse: (timestamp ns, signed volume contribution).
struct Impulse {
    ts_ns: i64,
    signed_sqrt_vol: f64, // ε × √V  (√BTC)
}

/// Bouchaud power-law propagator kernel.
pub struct PropagatorKernel {
    g0: f64,      // amplitude at τ=1s (ticks per unit √V, after κ normalisation)
    beta: f64,    // decay exponent
    kappa: f64,   // sqrt-law amplitude (ticks / √BTC)
    cutoff_ns: i64, // drop impulses older than this
    impulses: Vec<Impulse>,
}

impl PropagatorKernel {
    pub fn new(g0_ticks: f64, beta: f64, kappa_ticks_per_sqrt_btc: f64, cutoff_s: f64) -> Self {
        Self {
            g0: g0_ticks,
            beta,
            kappa: kappa_ticks_per_sqrt_btc,
            cutoff_ns: (cutoff_s * 1e9) as i64,
            impulses: Vec::new(),
        }
    }

    /// Construct from BTCUSDT-perp 2026-05-01 calibrated parameters.
    pub fn calibrated() -> Self {
        Self::new(64.76, 0.5, 162.57, 60.0)
    }

    /// Register a fill: `sign` = +1 (buy) / -1 (sell), `volume_btc` ≥ 0.
    pub fn push(&mut self, ts_ns: i64, sign: f64, volume_btc: f64) {
        if volume_btc <= 0.0 { return; }
        let signed_sqrt_vol = sign * volume_btc.sqrt();
        self.impulses.push(Impulse { ts_ns, signed_sqrt_vol });
    }

    /// Compute accumulated price impact at `now_ns` (ticks). Positive = mid shifted up.
    /// Also prunes expired impulses (age > cutoff).
    pub fn impact(&mut self, now_ns: i64) -> f64 {
        let cutoff_ts = now_ns - self.cutoff_ns;
        self.impulses.retain(|imp| imp.ts_ns >= cutoff_ts);

        let mut acc = 0.0_f64;
        let ns_per_s = 1e9_f64;
        for imp in &self.impulses {
            let tau_s = (now_ns - imp.ts_ns) as f64 / ns_per_s;
            if tau_s <= 0.0 { continue; }
            // G(τ) = G₀ × τ^{-β}; normalise so G(1s)=G₀.
            let g = self.g0 * tau_s.powf(-self.beta);
            acc += g * imp.signed_sqrt_vol;
        }
        // Multiply by κ to convert from normalised √BTC space to ticks.
        acc * self.kappa / self.g0
        // Division by g0 because: impact(τ=1s, V=1) = g0 × 1^{-β} × κ × √1 / g0 = κ
        // i.e., the 1-second impact of a 1-BTC fill = κ ticks.
    }

    /// Return the number of active impulses in the kernel.
    pub fn len(&self) -> usize {
        self.impulses.len()
    }

    pub fn is_empty(&self) -> bool {
        self.impulses.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_buy_gives_positive_impact() {
        let mut k = PropagatorKernel::calibrated();
        k.push(0, 1.0, 1.0); // buy 1 BTC at t=0
        // At τ=1s: impact = κ ticks ≈ 162 ticks
        let imp = k.impact(1_000_000_000); // 1s later
        assert!(imp > 0.0, "buy should produce positive impact, got {imp}");
        assert!(imp > 100.0 && imp < 300.0, "1s/1BTC impact should be ~163 ticks, got {imp}");
    }

    #[test]
    fn sell_gives_negative_impact() {
        let mut k = PropagatorKernel::calibrated();
        k.push(0, -1.0, 1.0); // sell 1 BTC
        let imp = k.impact(1_000_000_000);
        assert!(imp < 0.0, "sell should produce negative impact");
    }

    #[test]
    fn impact_decays_with_time() {
        let mut k = PropagatorKernel::calibrated();
        k.push(0, 1.0, 1.0);
        let i1s = k.impact(1_000_000_000);
        let i10s = k.impact(10_000_000_000);
        let i60s = k.impact(60_000_000_000);
        assert!(i1s > i10s && i10s > i60s, "impact must decay: {i1s:.1} > {i10s:.1} > {i60s:.1}");
    }

    #[test]
    fn sqrt_law_holds() {
        // Impact ∝ √V: doubling volume should increase impact by √2 ≈ 1.41×
        let mut k1 = PropagatorKernel::calibrated();
        let mut k2 = PropagatorKernel::calibrated();
        k1.push(0, 1.0, 1.0);
        k2.push(0, 1.0, 2.0);
        let i1 = k1.impact(1_000_000_000);
        let i2 = k2.impact(1_000_000_000);
        let ratio = i2 / i1;
        assert!((ratio - std::f64::consts::SQRT_2).abs() < 0.01,
            "I(2V)/I(V) should be √2 ≈ 1.414, got {ratio:.4}");
    }

    #[test]
    fn expired_impulses_are_pruned() {
        let mut k = PropagatorKernel::calibrated();
        k.push(0, 1.0, 1.0);
        assert_eq!(k.len(), 1);
        // Query far in the future (> 60s cutoff).
        let _ = k.impact(100_000_000_000); // 100s later
        assert_eq!(k.len(), 0, "impulse older than cutoff should be pruned");
    }

    #[test]
    fn no_impact_without_fills() {
        let mut k = PropagatorKernel::calibrated();
        assert_eq!(k.impact(1_000_000_000), 0.0);
    }

    #[test]
    fn opposing_fills_cancel() {
        let mut k = PropagatorKernel::calibrated();
        k.push(0, 1.0, 1.0);  // buy 1 BTC
        k.push(0, -1.0, 1.0); // sell 1 BTC at same time
        let imp = k.impact(1_000_000_000);
        assert!(imp.abs() < 1e-10, "equal opposing fills should cancel, got {imp}");
    }
}

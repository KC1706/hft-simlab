# Anatomy of the Reality Gap: Measuring the Marginal Value of Realism Components in Limit Order Book Backtesting

*(working title — alternatives logged in paper/NOTES.md)*

**Target venue:** ICAIF '26/'27 workshop → main track (ACM sigconf, 8 pages) · arXiv q-fin.TR preprint first.
**Format note:** content is drafted here in Markdown; converted to ACM LaTeX template at submission time.
**Status legend:** ◻ not started · ◐ drafted-with-project · ✓ frozen

---

## Abstract ◻
*(written LAST, after results exist. One sentence per: problem, gap, method, headline number, implication.)*

**Placeholder thesis:** Backtest profitability of high-frequency strategies is known to transfer
poorly to live trading, yet the simulation community lacks a quantitative decomposition of *which*
simulator deficiencies cause how much of the error. We implement a LOB backtesting stack in which
queue position, latency, and market-impact feedback are independently calibrated and independently
removable, and measure — across strategies, symbols, and held-out days — the marginal reduction in
PnL-prediction error attributable to each component. [Headline number goes here.]

## 1. Introduction ◻ — *written during Phase 0–1*
- The backtest-live gap as an industrial problem (cite the 888-strategy study on backtest
  metrics' lack of predictive power).
- Existing simulators model fidelity components ad hoc; no published component-wise attribution.
- Contributions (draft list, refine later):
  1. An open-source, calibrated, ablatable LOB simulation stack on free reproducible data.
  2. Calibration recipes for fill-probability, latency-race, and impact-feedback models on
     public crypto L2 feeds.
  3. The first systematic ablation quantifying each realism component's contribution to
     backtest fidelity.

## 2. Related Work ◻ — *seeded now, grown each time we use a paper*
- **Reality gap & simulation fidelity:** arXiv 2603.24137 (Madet et al., recipe we extend
  from large-tick equities to crypto; their latency-race timing analysis motivates §3.3's
  race-mode model); ABIDES (arXiv 1904.12066).
- **Queue & fills:** Moallemi & Yuan; arXiv 2403.02572 (Hamdan & Sirignano, fill
  probability estimation from L2 data with state-dependent conditioning; we adapt their
  method to crypto perpetual futures and extend the state vector with OFI).
- **Impact & order flow:** Bouchaud propagator literature; arXiv 2601.23172 unified theory.
  Cont, Kukanov & Stoikov (arXiv 1011.6402) define order-flow imbalance and show it linearly
  explains roughly half of short-horizon mid-price variance; we use OFI both as a pipeline
  validation statistic (§3.1) and later as a market-state feature in the fill model (§3.2).
- **Generative LOB models:** TRADES (2502.07071); DiffLOB (2602.03776); TradeFM (2602.23784).
- **Tools:** hftbacktest (the base we extend); CoinTossX (precedent for software papers).
- *(Each project journal entry that uses a paper must also add 1–2 sentences here.)*

## 3. The Simulation Stack ◻ — *written during Phases 1–2, section per component*
*(opens with [FIG-1] architecture; intro/§2 carries [FIG-2] four-lies schematic, [FIG-3] LOB primer if space)*
- 3.1 Data and book reconstruction (Phase 1 → here) ◐
  > **Draft prose (v0, 2026-06-10):** We use publicly available tick-level market data from
  > Binance USDT-margined perpetual futures, comprising every level-2 depth update and trade
  > print with exchange and receive timestamps (Tardis.dev first-of-month open datasets;
  > BTCUSDT 2026-05-01 alone contains 1.01×10⁸ depth events). Events are normalized into a
  > fixed-width binary representation and replayed deterministically. Two data-handling
  > details materially affect simulation fidelity and are enforced throughout: (i) trade
  > events are sequenced before their induced depth deltas to prevent double-decrementing
  > simulated queue positions; (ii) the venue's published event timestamps reflect message
  > *send* time rather than match time, biasing measured feed latency downward — we treat
  > this as a known limitation (§6). The naive baseline configuration (constant bilateral
  > 10 ms latency, worst-case queue placement) yields strongly negative market-making
  > performance (≈ −3 bps per trade), consistent with adverse-selection-dominated fills;
  > this cell anchors the ablation grid of §5.
  >
  > **Draft prose (v1 addition, 2026-06-12 — book reconstruction):** Book state is maintained
  > as a market-by-price structure keyed on integer price ticks, with absolute per-level
  > quantity updates and explicit best-quote tracking. Our implementation is independent of,
  > but semantics-equivalent to, the reference reconstruction in hftbacktest — including its
  > treatment of transiently locked or crossed updates, which are resolved by advancing the
  > opposing best-quote pointer while retaining stale levels for subsequent feed refresh
  > rather than deleting them. Equivalence is enforced by a level-for-level comparison of the
  > two reconstructions over a full trading day (§3.1, validation). The simulator additionally
  > records the incidence of locking and crossing updates as a per-day feed-quality
  > diagnostic, since elevated rates indicate degraded or reordered feed data that would
  > otherwise contaminate downstream calibration unobserved.
  >
  > **Draft prose (v2 addition, 2026-06-13 — validation & throughput):** The reconstruction
  > was validated against the reference implementation by comparing the top five price
  > levels of both books — integer price ticks and aggregate quantities — at randomly
  > sampled timestamps of the evaluation day; all sampled states agree exactly. On the
  > 2026-05-01 BTCUSDT session (2.67×10⁷ events, 416 min), the local-view book exhibits
  > 3,809 transiently crossing and 319 locking updates (≈1 per 6.5×10³ events), consistent
  > with ordinary feed staleness rather than data corruption. Single-threaded replay
  > sustains ≈8.6×10⁶ events/s on commodity hardware (Apple M2), comparable to the
  > reference stack; at ~10³ events/s average feed rate this leaves four orders of
  > magnitude of headroom for the per-event model computations introduced in §§3.2–3.4.
  >
  > **Draft prose (v3 addition, 2026-06-13 — data characterization, [FIG-10]):** The
  > evaluation data exhibits the canonical microstructure stylized facts: the spread is
  > pinned at one tick 99.9% of the time — BTCUSDT perpetual futures trade firmly in the
  > large-tick regime, making queue position the dominant execution variable — returns are
  > heavy-tailed (1 s excess kurtosis ≈ 10²) with pronounced volatility clustering, and
  > trade signs are long-memory with a sign-autocorrelation decay exponent ≈ 0.63 over four
  > decades of lag. As a pipeline validation, order-flow imbalance computed from our
  > reconstruction (following Cont et al., arXiv 1011.6402) explains 48% of one-second
  > mid-price variance (β ≈ 0.94 ticks per unit), consistent with published estimates.
  > These statistics, logged as reusable artifacts during replay, later serve as the
  > fidelity targets for the generative order-flow model (§3.5).
- 3.2 Queue-position and fill-probability model + calibration (P2.2 → here) — [FIG-4] ◐
  > **Draft prose (v0, 2026-06-13 — model survey and motivation):**
  > An order submitted to a price level joins the back of a FIFO queue; it fills only when
  > all volume ahead of it has been consumed. Naive simulators treat any touch of a price
  > level as an immediate fill, ignoring queue position entirely. The reference implementation
  > (hftbacktest) provides two L2-compatible queue models: a *risk-adverse* model that
  > advances queue position only on trades (ignoring cancellations), and a *probability-based*
  > model that heuristically attributes a fraction of cancellation-driven level decreases to
  > orders ahead of the simulated position, with the fraction determined by a parametric
  > function of the queue volumes ahead and behind (power-law or log variants). Both models
  > share a structural limitation: their probability functions are not fitted to data. They
  > encode qualitative assumptions — cancellations are roughly uniform across the queue —
  > without conditioning on any observable market state. On BTCUSDT perpetual futures, where
  > the spread is pinned at one tick 99.9% of the time (§3.1), queue position is the dominant
  > execution variable; a miscalibrated fill model directly translates to misstated strategy
  > profitability.
  >
  > We replace these heuristic models with a calibrated fill-probability estimator following
  > arXiv 2403.02572. For each resting order placed at queue position q (measured in units of
  > level volume at arrival), we estimate P(fill within τ | q, s), where s is a market-state
  > vector comprising spread, short-horizon realized volatility, and order-flow imbalance at
  > arrival. Estimation proceeds from L2 data alone: a simulated passive order is inserted
  > at a given queue position; subsequent tape events determine whether a fill would have
  > occurred within horizon τ, using the conservative rule that a fill requires the level to
  > be fully consumed past the order's position by trade prints (cancellations are inferred
  > from level-quantity decreases not attributable to trades). This yields a labeled dataset
  > of (q, s, τ, outcome) pairs from which a nonparametric fill-probability surface is fitted.
  > [FIG-4a] shows the resulting empirical fill curves and fitted model; [FIG-4b] shows the
  > reliability diagram on the second half of the evaluation day (expected calibration error
  > 2–4% across horizons and sides). Calibration details and the labeled-dataset construction
  > procedure are described in §A.1 (appendix).
  >
  > **Draft prose (v1 addition, 2026-06-13 — empirical fill statistics):** On the BTCUSDT
  > perpetual evaluation day, the unconditional fill probability for a passive order placed at
  > the back of the best-bid queue (normalized position q/Q = 1) is approximately 10% within
  > 1 second, 40% within 10 seconds, and 53% within 60 seconds under the any-decrease
  > criterion. The corresponding trade-only rates are 9%, 30%, and 38%, implying that
  > roughly 27% of queue-position advancement at the best level comes from cancellations
  > rather than executions — a contribution invisible to the risk-adverse queue model.
  > The fill-probability surface decreases steeply from q/Q ≈ 0 to q/Q ≈ 0.4 and then
  > flattens, reflecting the convex value of front-of-queue priority: a passive order at
  > 5% depth has approximately 2× the fill probability of one placed at 100% depth over
  > a 60-second horizon.
- 3.3 Latency model with race dynamics + calibration (P2.3 → here) — [FIG-5] ◐
  > **Draft prose (v0, 2026-06-13 — feed latency calibration and race model):**
  > The entry latency distribution is calibrated from the feed's own timestamps: for each
  > recorded depth and trade event, `feed_latency = local_ts − exch_ts` is a directly
  > observable proxy for order one-way latency on the same network path. Over 1.9 million
  > events on the BTCUSDT-perp evaluation day, the feed latency distribution is well
  > approximated by a log-normal (µ = 8.30, σ = 0.78 in log µs) with a mode of 2.2 ms,
  > a median of 4.0 ms, and a heavy tail reaching 137 ms at the 99th percentile [FIG-5a].
  > We treat the feed latency as the entry-path lower bound; the response leg is assumed
  > equal (symmetric network path), giving an estimated order round-trip mode of 4.4 ms.
  >
  > We additionally implement a race-mode component following arXiv 2603.24137: after each
  > trigger event (best-quote change or trade exceeding a volume threshold), all orders
  > submitted within one estimated round-trip window are assigned a uniform arrival-time
  > jitter U(0, 4.4 ms) on top of the base log-normal sample. This models the empirical
  > observation that simultaneous reactions to the same trigger arrive at the exchange in a
  > burst within one round-trip, creating a race whose outcome depends on sub-millisecond
  > ordering. The three latency configurations — constant (baseline), log-normal (captures
  > intraday variation), race-aware (adds burst dynamics) — are independently removable in
  > the ablation of §5.
- 3.4 Impact-feedback propagator + calibration (P2.4 → here) — [FIG-6] ◐
  > **Draft prose (v0, 2026-06-13 — propagator calibration):**
  > We model the market-impact feedback following Bouchaud et al. (*Trades, Quotes &
  > Prices*; arXiv 2603.24137): after a fill of signed volume εV, the mid-price is
  > shifted by G(τ) = G₀ × τ^{−β} × ε × κ√V, where G(τ) is the power-law propagator
  > and κ√V is the square-root impact law amplitude. The square-root coefficient is
  > calibrated empirically from 100-millisecond trade response on the evaluation day:
  > E[ε × Δmid(100ms)] as a function of √V gives κ = 162.6 ticks/√BTC [FIG-6b].
  > The propagator decay exponent β = 0.5 is taken from the Bouchaud literature for
  > large-tick assets; single-day calibration is insufficient to identify β because the
  > evaluation day exhibits a strongly trending regime in which R(τ) is still rising at
  > τ = 60 s, precluding the deconvolution that would recover β. This limitation is
  > stated explicitly and multi-day calibration is deferred to the held-out evaluation
  > design of §4. The propagator amplitude is anchored to R(1s) = 64.8 ticks per unit
  > signed volume [FIG-6a]. In simulation, the kernel accumulates all fills in a 60-second
  > rolling window and applies the aggregate price shift as an additive offset to the
  > mid-price seen by subsequent strategy logic; fills older than 60 seconds contribute
  > at most G(60)/G(1) = 60^{−0.5} ≈ 13% of their initial impact and are pruned.
- 3.5 Composing the stack: the event-level ablation simulator (P2.5–P2.6 → here) ◐
  > **Draft prose (v0, 2026-06-28 — integration and fill execution):**
  > The three calibrated components of §§3.2–3.4 are assembled into a single backtest in
  > which each is independently removable, yielding the ablation lattice evaluated in §5.
  > A market-making strategy is replayed against the reconstructed book; at each decision
  > epoch it cancels and re-quotes a passive order per side, and each resting order is
  > subject to the calibrated fill, latency, and impact models according to the active
  > configuration. Fills are resolved in **continuous event time** rather than on the
  > decision grid: when an order is placed it is assigned a fill instant
  > t_fill = t_active + X, where t_active = t_decision + ℓ is the order's arrival time under
  > a latency draw ℓ from the model of §3.3, and X is an exponential waiting time whose rate
  > λ = −ln(1 − p₁) is chosen so that the probability of filling within one second equals the
  > calibrated fill probability p₁ = P(fill | q, s) of §3.2. An order is executed if its fill
  > instant precedes its cancellation at the next re-quote; otherwise it is cancelled unfilled.
  > Under the naive baseline, fills are instead resolved deterministically against the trade
  > tape — an order fills the instant a print crosses its price — reproducing the touch rule
  > that defines the fill lie.
  >
  > Resolving fills in event time rather than per grid interval is what makes the latency
  > component observable: because the fill clock is measured from t_active, entry latency
  > translates one-to-one into a later fill instant, and a latency draw exceeding the
  > re-quote interval removes the fill entirely. A grid-level fill check, by contrast, is
  > insensitive to millisecond latency whenever the decision interval exceeds the latency
  > scale — the effect of the latency component then vanishes by construction, an artifact we
  > observed and corrected. The construction is exact in the limit of fine decision grids and
  > degrades gracefully; the marginal PnL contribution of the latency component strengthens
  > monotonically as the re-quote interval approaches the latency scale. We note two
  > first-cut approximations, refined in §4's evaluation engine: the strategy maintains a
  > single resting order per side, and the exponential fill law honours only the one-second
  > calibrated horizon (a memoryless approximation of the multi-horizon isotonic surface of
  > §3.2). Quantitative ablation results appear in §5.
- 3.6 Generative counterfactual order flow (Phase 3 → here) — [FIG-7] ◐
  > **Draft prose (v0, 2026-06-28 — model and data adaptation):**
  > The parametric impact kernel of §3.4 lets the recorded market respond to the agent's
  > volume but not to its *presence*: counterparties cannot withdraw or re-quote in reaction
  > to the agent's orders. To model that final feedback channel we introduce a generative
  > order-flow model — a conditional denoising diffusion model over order events (TRADES;
  > Berti et al., arXiv 2502.07071) — that samples each successive market event from a
  > learned distribution conditioned on the recent event history and the current book, so
  > that altering the conditioning by inserting the agent's orders changes the simulated
  > market's subsequent behaviour. A key adaptation is required: the model is formulated for
  > level-3 message data (per-order submissions, cancellations, deletions and executions),
  > whereas public cryptocurrency feeds are level-2 (aggregate per-level quantities). We
  > therefore synthesize a level-3-equivalent event stream from level-2 dynamics — a level
  > quantity increase is mapped to a submission, a decrease to a cancellation (or deletion if
  > the level empties), and a trade print to an execution — and pair each synthesized event
  > with the contemporaneous top-ten book snapshot. Because order identities are unobservable
  > at level 2, the cancellation-versus-execution attribution is inferred rather than
  > observed, making the synthesized cancellation rate an upper bound; we treat the realism of
  > the resulting stream as an empirical question answered by the stylized-fact validation of
  > §3.7 rather than asserted. The model is scoped to approximately ten million parameters,
  > trained on free-tier cloud GPUs and run at inference on commodity hardware, keeping the
  > entire pipeline reproducible at zero cost.
  >
  > **Draft prose (v1 addendum, 2026-06-29 — normalization and its inverse):** Training
  > operates on a per-feature standardized representation: each continuous order and book
  > feature is centred and scaled to zero mean and unit variance, while the categorical event
  > type and side are left in their native encoding. Because standardization is affine it is
  > exactly invertible from the stored per-feature means and standard deviations, and we
  > recover real-unit events by applying that inverse; we verify the inversion by a round-trip
  > on held-out data, which reproduces the original to floating-point precision. The four
  > convenience observables — mid-price, spread, top-of-book imbalance and volume-weighted
  > average price — are not de-normalized as independent quantities but recomputed from the
  > recovered book by the identical arithmetic used to construct the real data, guaranteeing
  > internal consistency and ensuring any divergence measured in §3.7 reflects book geometry
  > rather than a mismatch in metric definitions. Event timestamps are reconstructed by
  > cumulating the generated inter-arrival times; since every stylized fact is computed from
  > first differences, the unobserved absolute time origin is immaterial.
- 3.7 Stylized-fact validation of the generative model — [FIG-7] ◐
  > **Draft prose (v0, 2026-06-28 — quantitative realism scoring):**
  > Generative market models are easily made to *appear* realistic, so we hold ours to a
  > quantitative standard: a generated event stream is scored against held-out real data by
  > the two-sample Kolmogorov–Smirnov distance on each microstructure stylized fact —
  > per-event spread, standardized event-time mid-return (heavy tails), order size,
  > inter-arrival time, and top-of-book volume imbalance — together with the gap in excess
  > kurtosis and in the power-law slope of the trade-sign autocorrelation (long-memory order
  > flow). The KS statistic is distribution-free and binning-free, which matters because every
  > fact here is heavy-tailed; it reduces "looks realistic" to a per-fact number bounded in
  > [0,1] with an interpretable floor (real-vs-real ≈ 0). Facts are measured in event time
  > rather than on a wall-clock grid, so the realism of the sequence is assessed separately
  > from the realism of the inter-arrival clock (itself scored as the dt distance). A model is
  > deemed to reproduce a stylized fact when its KS is near that floor; the per-fact table is
  > the generative model's realism scorecard.

*All figure content, axes, captions, and the raw data each phase must log are fully specified
in `paper/FIGURES.md` — figures are designed before the data exists.*

## 4. Experimental Design ◻ — *written during Phase 4 setup, BEFORE results* — [FIG-8]
- Strategies under test; symbols/days; calibration vs held-out split protocol;
  ablation grid; error metric (PnL-gap distribution vs real replay); significance testing.
- *(Pre-registering the design in this section before running experiments is what makes the
  results credible — and is good scientific hygiene reviewers reward.)*

## 5. Results ◻ — *Phase 4* — [FIG-9] waterfall + [TAB-1] ablation table
- The ablation table (the paper's centerpiece) + per-component marginal fidelity gains.

## 6. Limitations & Conclusion ◻
- Crypto-L2-only validation; single-venue; inference-based fill ground truth — be loud about
  these, reviewers respect it.

---

## Writing protocol (mirrors CLAUDE.md)
After each project phase step, the matching paper section gets its first-draft prose **in the
same session** as the journal entry. Journal = teaching voice; paper = academic voice. Same
facts, two registers — writing both back-to-back is deliberate practice for the user in
academic writing.

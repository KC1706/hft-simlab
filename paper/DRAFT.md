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
  >
  > **Addendum (v1, 2026-07-20 — first-pass conditional scoring):** We first evaluate the model
  > in a *conditional* mode that isolates the generated order flow from the generated book: each
  > sampled event is produced from a real 256-event context and paired with the real
  > contemporaneous book snapshot. This design has a deliberate consequence — every stylized
  > fact that is a deterministic function of the book (spread, mid-return and its kurtosis,
  > volume imbalance) is matched *by construction*, so its KS is trivially zero and reports
  > nothing about the model. The informative tests in this mode are exactly the two quantities
  > the model emits, order size and inter-arrival time. On a 3,000-event held-out sample from a
  > model trained for five epochs (validation simple-loss 0.42→0.37, still descending), these
  > stand at KS(size)=0.43 and KS(dt)=0.55 — far from the real-vs-real floor — with roughly half
  > of generated sizes falling negative before post-hoc rejection. We report this as an explicit
  > lower-bound checkpoint rather than a headline: it validates the full generate→de-normalize→
  > score pipeline while showing that flow realism requires substantially more training, and it
  > motivates the *unconditional* evaluation (§3.8, forthcoming), where the model rolls its own
  > book forward autoregressively and the book-derived facts finally become genuine tests.

*All figure content, axes, captions, and the raw data each phase must log are fully specified
in `paper/FIGURES.md` — figures are designed before the data exists.*

## 4. Experimental Design ◑ — *written during Phase 4 setup, BEFORE results* — [FIG-8]

**Strategies under test (the probes).** A backtest's realism is only observable through a strategy
that uses it: the strategy is the instrument and its PnL the reading. We deliberately test strategies
that stress *different* parts of the stack, because a realism component that dominates one strategy's
error can be irrelevant to another — and that contrast is itself a result, not noise. Three subjects:
1. **Fixed-offset market maker (MM)** — quotes ±k ticks around mid with inventory skew (k ∈
   {20,40,60,80} ticks as strategy variants). A *maker*: it rests in the queue, so it is sensitive to
   the fill-probability model, to latency, and to impact.
2. **OFI-momentum taker** — crosses the spread toward top-of-book order-flow imbalance past a
   threshold. A *taker*: it fills at the touch with certainty, so the queue/fill and latency-race
   models are **structurally inapplicable** to it; only impact bites. It is the maker's opposite.
3. **Learned market maker (RL-MM)** — a small policy (linear softmax over standardized features:
   book imbalance, inventory, spread, short-horizon return) selecting a half-spread/skew action each
   decision step, trained by REINFORCE with an inventory-penalized reward. It is a *learned*, state-
   adaptive probe: the finding must not depend on hand-picked parametric rules, so a policy chosen by
   optimization rather than by the authors is the harder-to-dismiss third subject.

   *Method caveat (stated, not buried):* the RL policy is trained once, under the **naive-fill**
   configuration, and then evaluated **frozen** across every ablation configuration. Latency and
   impact are intentionally excluded from training so the policy is not tuned to a particular realism
   toggle — otherwise the ablation would measure the policy's adaptation rather than the component's
   effect. Training in the simplest configuration and evaluating in the full stack is a standard
   sim-to-sim transfer; we report it as such. The policy is trained on the calibration day only; the
   held-out day is never seen during training.

**Symbols / days.** Calibrate every component on day A; evaluate on a held-out day B (both free
Tardis.dev first-of-month BTCUSDT L2+trades). The split is the point: an in-sample gap measures
overfitting, not fidelity.

**Ablation grid.** Each strategy × each simulator configuration
`{naive → +fill → +fill+latency → +fill+latency+impact}` × seeds (the calibrated fill and latency
models are stochastic; seeds give a PnL distribution, not a point).

**Error metric.** Ground truth is the naive-fill backtest on the *real* replayed book of day B — a
documented proxy, since crypto L2 lacks true order-ID queue positions. The per-configuration gap is
`g(S,D,cfg) = |PnL_sim − PnL_real|`; we keep the full signed-error distribution over seeds, not just
the mean.

> **Pre-registration amendment (recorded on first running the experiment).** The ground-truth proxy
> above — "naive-fill on the real book" — is, in our implementation, *identically* the `naive`
> ablation configuration (both fill an order only when a real trade prints through its level).
> Hence `PnL_sim(naive) ≡ PnL_real` and `g(naive) = 0` by construction, and a "marginal gap
> reduction as components switch on" is not well-defined against this baseline: without level-3
> queue data there is no truth independent of the naive proxy for the calibrated models to be
> *closer to*. We therefore report the quantity the design can honestly measure — the **signed
> correction** each realism component applies to the naive-real baseline, its statistical
> significance, and, decisively, **whether that correction replicates on the held-out day**. "Does
> adding realism change the backtested PnL, by how much, for which strategies, and does it transfer
> across days?" replaces "does realism close a gap?". §5 reports this.

**Significance.** Paired (per-seed) bootstrap confidence intervals on each component's marginal
effect; a component "matters" when its CI excludes zero. Deterministic effects (e.g. the taker's
exactly-zero fill/latency sensitivity) collapse to a point, which is the honest report there.

- *(Pre-registering the design in this section before running experiments is what makes the
  results credible — and is good scientific hygiene reviewers reward.)*

## 5. Results ◐ — *Phase 4* — [FIG-9] waterfall + [TAB-1] ablation table

We calibrate every component on day A (`2026-05-01`) and evaluate the frozen stack on a held-out day
B (`2026-06-01`), both BTCUSDT; the days are matched by event budget (≈26–27 M events) rather than
wall-clock, since day B is ≈1.5× more active per unit depth (§4, day B is a 4.46 h window). All
strategies lose money net of adverse selection and maker fees on both days (passive making is
unprofitable on this instrument — a property of the data, not a result we claim), so we read the
ablation through the *correction* each component applies, not through profitability.

**[TAB-1] Marginal effect of each realism component on backtested PnL (USDT), paired per-seed
bootstrap 95% CI, 8 seeds; "sig" = CI excludes zero.** The *fill* column is the headline.

| subject | fill ΔPnL (day A) | fill ΔPnL (day B, held-out) | impact ΔPnL (A / B) | latency ΔPnL (A / B) |
|---|---|---|---|---|
| OFI-taker | **+0.00** det. | **+0.00** det. | −3.80 / −2.16 (sig both) | 0.00 / 0.00 |
| MM(½=20) | +2.16 [1.37,2.98] sig | **+25.72** [24.23,26.96] sig | −4.21 / −4.50 (sig both) | +0.87 / +1.15 (ns) |
| MM(½=40) | +1.93 [0.27,3.53] sig | **+21.35** [19.64,22.80] sig | −1.04 / −1.20 (ns both) | +0.36 / −0.07 (ns) |
| MM(½=60) | **−3.22** [−4.83,−1.51] sig | **+14.85** [14.23,15.33] sig | −1.62 / +0.07 | +1.34 / −1.80 |
| MM(½=80) | **−8.10** [−9.36,−6.82] sig | **+9.05** [8.07,9.96] sig | −0.01 / −0.79 (ns) | +1.12 / −0.25 |
| RL-MM (learned) | +7.52 [6.69,8.42] sig | **+23.11** [22.42,23.78] sig | −3.19 / −2.91 (sig both) | +0.51 / −1.30 (ns) |

Three findings, separated by how well they survive the held-out day.

**(i) The fill/queue model is the dominant correction, and its significance transfers; its magnitude
is large and regime-dependent.** For every maker, on both days, switching the naive trade-through
rule to the calibrated P(fill | queue) model is a statistically significant PnL correction — never
negligible. On day B it is a *large positive* correction for every maker (+9 to +26 USDT against
naive PnL bases of −24 to −65), i.e. the naive backtest **understates** maker PnL by 15–40 %. The
economic reading (§3.2, Bouchaud TQP ch. 6): the queue model fills resting orders probabilistically
while they wait, not only on a full trade-through, yielding more fills at the maker's favorable
quoted price. The magnitude scales with how much a strategy rests in the queue — largest for the
tight fixed maker and the learned maker, smallest for the widest maker.

**(ii) The taker's structural immunity is exact and transfers perfectly.** The OFI-taker's fill and
latency corrections are *identically zero* on both days: a strategy that crosses the spread fills at
the touch regardless of queue position, so neither the queue model nor the latency race can move it;
only its own market impact bites (−3.80 day A, −2.16 day B, significant both). This is the sharpest
cross-strategy contrast in the paper and it transfers exactly, because it is a property of the
strategy's interaction with the book, not of the day.

**(iii) The *sign* of the fill correction for wide makers does *not* transfer — an honest negative
result.** On day A the fill correction flipped sign with maker width (helping tight makers, +2.16,
but *hurting* wide makers, −8.10). On day B it is positive for *every* width, decreasing
monotonically (+25.72 → +9.05) but never crossing zero. The clean day-A sign-flip is therefore a
day-specific artifact, not a robust finding; we demote it accordingly. What survives is the weaker,
truer statement: the fill correction's **magnitude** is strategy- and regime-dependent (largest for
tight/adaptive makers, smallest for wide ones), it is **exactly zero** for takers, and its **sign**
for wide makers is not stable across days. Catching this is precisely what the held-out day is for.

*Impact* is a consistent small negative correction wherever significant (taker −2 to −4; tight and
learned makers −3 to −4.5) and transfers in sign and rough magnitude. *Latency* is second-order:
rarely significant on either day. The learned RL-MM behaves as a tight, aggressive maker — it anchors
the high-fill-sensitivity end of the spectrum (+7.52 day A, +23.11 day B) and tracks the tight fixed
maker, confirming the fingerprint is a property of the quoting behavior the policy learned, not of
hand-chosen parameters. It is a *probe*, not a profitability result (§4 caveat).

> **[FIG-9]** waterfall: PnL vs config `{naive→+fill→+fill+latency→+fill+latency+impact}`, one panel
> per strategy, day A and day B overlaid, seed-dispersion bands — to be rendered from
> `experiments/ablation/results.parquet` (day A) and `results_dayB.parquet` (day B).

*Headline sentence (candidate, wording TBD): the fill-model correction alone moves backtested
market-maker PnL by 15–40 %, is significant for every maker on a held-out day, and is exactly zero
for takers — so a backtest's realism budget should be spent where the strategy actually interacts
with the book.*

## 6. Limitations & Conclusion ◻
- Crypto-L2-only validation; single-venue; inference-based fill ground truth — be loud about
  these, reviewers respect it.

---

## Writing protocol (mirrors CLAUDE.md)
After each project phase step, the matching paper section gets its first-draft prose **in the
same session** as the journal entry. Journal = teaching voice; paper = academic voice. Same
facts, two registers — writing both back-to-back is deliberate practice for the user in
academic writing.

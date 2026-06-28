# Figure & Diagram Specifications

Every figure the paper needs, specified in full detail BEFORE the data exists — so each phase
knows exactly what artifacts it must produce, and no experiment gets run twice for plotting.
Convention: `FIG-<n>` referenced from `DRAFT.md` as `[FIG-n]`. Each spec has: purpose, exact
visual content, data source (which phase produces it), draft caption, and design notes.
Status: ◻ not produced · ◐ data exists, plot drafted · ✓ camera-ready.

---

## FIG-1 ◻ — System architecture / the ablatable simulation stack
**Section:** 3 (opening figure) · **Produced by:** drawn manually (diagrams.net/TikZ), Phase 2
**Purpose:** One-glance understanding of the whole system and of what "ablatable" means.
**Content (block diagram, left→right data flow):**
- Left: cylinder "Recorded L2 feed + trades (Tardis / own collector)" → "Book reconstruction".
- Center: "Event-driven simulator core" box containing FOUR stacked plug-in slots, each drawn
  with an explicit ON/OFF toggle switch icon: (a) Queue/fill model, (b) Latency model,
  (c) Impact-feedback kernel, (d) Generative order flow. Each slot annotated with its paper
  (2403.02572 / 2603.24137 / 2601.23172+Bouchaud / 2502.07071).
- Right: "Strategy under test (MM / taker / RL)" → "PnL ledger".
- Bottom rail: "Real replay of held-out days" → comparator node "Δ = sim PnL − real PnL"
  feeding the ablation table. This rail is the experiment's punchline — make it visually
  distinct (dashed, accent color).
**Caption draft:** "Architecture of the simulation stack. Each realism component is
independently calibrated and independently removable; the sim-to-real comparator (bottom)
measures the PnL-prediction error of every configuration against held-out real replay."
**Design notes:** single-column width; max ~10 boxes total; toggles are the key visual idea.

## FIG-2 ◻ — The four lies of naive backtesting (didactic schematic)
**Section:** 1 or 2 · **Produced by:** drawn manually, can be made early (Phase 1)
**Purpose:** Make the gap mechanisms visceral for non-microstructure readers; this is the
figure people screenshot.
**Content:** 2×2 panel grid of mini-schematics, one per lie:
- (a) *Fill lie:* price-level box with a FIFO queue of small order rectangles; "you" highlighted
  at the back; arrow shows price touching the level, consuming only the front of the queue,
  then retreating. Naive backtest verdict "FILLED ✓" vs reality "NOT FILLED ✗" stamped on top.
- (b) *Latency lie:* timeline with two book snapshots; signal observed at t, order lands at
  t+δ into a visibly different book (best level gone). Annotate δ with "feed + order latency".
- (c) *Impact lie:* mid-price path flat in replay vs the same path bending against a buy
  program in reality; shaded area = execution window.
- (d) *Feedback lie:* quotes on opposite side widening/pulling in response to "your" order;
  replay shows them frozen.
**Caption draft:** "Four mechanisms by which naive replay backtests misstate performance:
(a) queue-position-blind fills, (b) zero-latency action, (c) absent market impact,
(d) absent counterparty reaction. Sections 3.2–3.5 instrument each mechanism."
**Design notes:** consistent icon language across panels; no real data needed — pure schematic.

## FIG-3 ◻ — LOB mechanics + queue position primer (likely appendix or cut)
**Section:** 2/appendix · **Produced by:** drawn manually
**Content:** classic two-sided book ladder (bids green left, asks red right, volume bars per
price), spread/mid annotated; inset zoom of one bid level showing FIFO queue with arrival-time
ordering and "queue position q" marker.
**Caption draft:** "Limit order book structure and intra-level FIFO queue. An order's fill
hazard depends on both its price level and its position q within the level."
**Design notes:** only include if reviewer audience needs it (workshop: yes, ICAIF main: cut).

## FIG-4 ◐ — Fill-probability calibration (the P2.2 money plot) — data produced 2026-06-13
**Section:** 3.2 · **Produced by:** Phase 2.2 validation run
**Content:** two panels.
- (a) Empirical fill curves: x = queue position ahead (normalized by level volume), y =
  P(fill within τ) for τ ∈ {1s, 10s, 60s}, one line each, with 95% CI bands; overlaid dashed
  lines = our fitted model. Optionally split by spread state (1-tick vs >1-tick) as line styles.
- (b) Reliability diagram on a HELD-OUT day: x = model-predicted fill prob (binned deciles),
  y = realized fill frequency; identity diagonal; point size = bin count; report ECE
  (expected calibration error) in the corner.
**Caption draft:** "(a) Empirical fill probability vs normalized queue position with fitted
model. (b) Out-of-sample calibration: predicted vs realized fill rates hug the diagonal
(ECE = X.XX). Estimation follows arXiv 2403.02572 adapted to public L2 crypto feeds."
**Data to log during P2.2:** per simulated order: queue estimate, market state vector,
predicted p, realized outcome, τ. Store as parquet so the figure regenerates from one script.

## FIG-5 ◐ — Latency distribution and race mode (the P2.3 evidence) — data produced 2026-06-13
**Section:** 3.3 · **Produced by:** Phase 2.3 calibration
**Content:** two panels.
- (a) Histogram (log y) of inter-event reaction times around trigger events (large trades /
  best-level changes) measured from recorded feed timestamps; annotate the mode at exchange
  round-trip latency with a vertical line + label "race mode" (the 2603.24137 signature).
- (b) QQ-plot or overlaid density: recorded reaction-time distribution vs our fitted latency
  model samples — shows the model reproduces both the race mode and the heavy tail.
**Caption draft:** "(a) Reaction-time distribution around trigger events exhibits a pronounced
mode at exchange round-trip latency, consistent with simultaneous reactions and latency races
(cf. arXiv 2603.24137). (b) Our fitted latency model reproduces both the race mode and tail."
**Data to log during P2.3:** trigger-event timestamps, subsequent event timestamps, venue,
symbol — raw inter-arrival table.

## FIG-6 ◐ — Impact and reversion (the P2.4 money plot) — data produced 2026-06-13
**Section:** 3.4 · **Produced by:** Phase 2.4 validation
**Content:** two panels.
- (a) Average mid-price trajectory (in spreads or bps) around a standardized simulated buy
  program: shaded execution window, rise during execution (concave), partial post-trade
  reversion after; three lines: naive replay (flat — the lie), our kernel, and the
  empirical/literature reference shape.
- (b) Peak impact vs participation/size on log-log axes with fitted slope ≈ 0.5 annotated
  (square-root law check); points = simulated programs of varying size.
**Caption draft:** "(a) Simulated execution exhibits concave impact and post-trade reversion
under the calibrated power-law propagator (Bouchaud et al.); naive replay shows none.
(b) Peak impact scales ≈ √size, matching the empirical square-root law."
**Data to log during P2.4:** per synthetic program: size, schedule, mid path ±5 min, kernel
params used.

## FIG-7 ◻ — Stylized-facts panel: real vs generated market (Phase 3 gate)
**Section:** 3.5 · **Produced by:** Phase 3.3 validation suite
**Content:** 2×3 grid, each cell overlaying REAL (solid) vs GENERATED (dashed) for one
stylized fact: (a) 1s-return distribution, log-density, fat tails visible; (b) autocorrelation
of returns (≈0 beyond lag 1); (c) autocorrelation of |returns| (slow decay = vol clustering);
(d) autocorrelation of signed order flow (long memory, log-log); (e) spread distribution;
(f) average book-depth profile by level. Each cell prints its KS distance / relevant statistic.
**Caption draft:** "Generated order flow (TRADES-style model, arXiv 2502.07071, retrained on
crypto L2) reproduces the six target stylized facts of the real feed; quantitative distances
in each panel."
**Data to log during P3.3:** the validation suite must dump every statistic to parquet —
figure is a pure rendering of the suite's output.

## FIG-8 ◻ — Sim-to-real experimental protocol (design diagram)
**Section:** 4 · **Produced by:** drawn manually during Phase 4 setup
**Content:** horizontal calendar/timeline: calibration window (days 1..k, shaded) →
embargo gap → held-out evaluation days (k+e..n, hatched — "never touched during development");
below it, the evaluation loop as arrows: {simulator config c × strategy s × day d} →
sim PnL_csd vs real-replay PnL_sd → error metric ε_csd → aggregation into the ablation table.
**Caption draft:** "Evaluation protocol. Models are calibrated on past days only; PnL
prediction error is measured on embargoed held-out days, per simulator configuration,
strategy, and symbol."
**Design notes:** this figure pre-registers the design — draw it BEFORE running Phase 4.

## FIG-9 ◻ — THE HEADLINE: marginal value of each realism component
**Section:** 5 (and the abstract's number comes from it) · **Produced by:** Phase 4.3
**Content:** waterfall chart: x = simulator configurations left→right (naive → +queue →
+latency → +impact → +generative), y = median |PnL prediction error| (bps or % of gross PnL);
each bar drop annotated with the marginal error reduction; whiskers = IQR across
strategy×symbol×day cells; significance stars on each marginal step (paired test).
Companion (same data, appendix): violin/box distributions per config.
**Caption draft:** "Marginal reduction in backtest PnL-prediction error per realism component.
[Component X] accounts for the largest share of the reality gap (−YY%), while [component Z]'s
contribution is not statistically distinguishable."
**Data to log during P4:** the full ε_csd tensor (config × strategy × symbol × day) — one
parquet file, from which FIG-9, TAB-1, and all significance tests regenerate.

## FIG-10 ◐ — Dataset stylized-facts panel (the P1.3 baseline)
**Section:** 3.1 (or appendix data description) · **Produced by:** Phase 1.3 — DONE 2026-06-13
**Purpose:** establish that the evaluation data exhibits the standard microstructure
stylized facts (so later claims aren't artifacts of a weird dataset), and fix the project's
measurement conventions. Doubles as the REAL-side reference for FIG-7.
**Content:** 2×3 grid: (a) spread distribution, log-y (P(1 tick)=0.999 — large-tick regime);
(b) average depth profile, top-10 levels per side; (c) standardized 1s-return log-density vs
N(0,1) (excess kurtosis ≈ 108); (d) ACF of returns vs |returns| to 300 s (vol clustering);
(e) trade-sign ACF, log-log, with power-law fit (slope ≈ −0.63 — long-memory flow);
(f) response function R(τ) = E[sign·Δmid(τ)] to 60 s, log-x. Stats also saved to
`experiments/data/p13/stats.json`, incl. OFI→return regression (β=0.94 ticks/u, R²=0.48 @1s).
**Caption draft:** "Stylized facts of the BTCUSDT-perp evaluation data (2026-05-01): pinned
one-tick spread, fat-tailed returns, volatility clustering, long-memory order flow, and a
concave trade-response function — the targets the generative market of §3.5 must reproduce."
**Data:** `experiments/data/p13/{samples,trades}.parquet` (100 ms grid + full trade tape);
regenerates via `experiments/figures/p13_stylized_facts.py`. FIG-7 reuses these as its
solid/"real" lines.

## TAB-1 ◻ — Ablation table (centerpiece table, pairs with FIG-9)
**Section:** 5 · Rows = simulator configs; columns = per-strategy median error, aggregate
median, IQR, p-value vs previous row. One command (`make ablation`) regenerates it.

---

## Production rules
- Every data figure regenerates from a single script in `experiments/figures/` reading logged
  parquet — no hand-edited plots, ever (reproducibility is a stated contribution).
- Log the figure's raw data DURING the phase (specs above say what to log) — never re-run
  experiments for plotting.
- Schematics (FIG-1,2,3,8) live as editable sources (`.drawio`/`.tex`) in `paper/figs-src/`.
- Color scheme: real data = solid/dark, simulated = dashed/accent, naive baseline = grey.
  Color-blind-safe palette; all panels readable in grayscale print.

# Phase 4 — Sim-to-real PnL ablation (design)

**The headline research contribution.** Everything so far built realism *components* (calibrated
fills P2.2, latency P2.3, impact P2.4, generative market P3). Phase 4 *measures* them: **by how
much does each component close the gap between a backtest's PnL and the PnL you'd actually have
gotten?** The per-component marginal gap reduction is the publishable number (PLAN.md).

## The metric (define it precisely — this is the crux)

For a strategy S on a day D:
- **Ground truth** `PnL_real(S, D)` — S run against the REAL replayed book of D. Without true L3
  queue position we can't know real fills exactly, so ground truth is the *naive-fill* backtest on
  real data (fill only when a real trade crosses S's quote). It is a proxy, documented as such.
- **Simulated** `PnL_sim(S, D, cfg)` — S run under simulator config `cfg`, an ablation over the
  realism stack: `naive → +fill → +fill+latency → +fill+latency+impact → +generative-market`.
- **Gap** `g(S, D, cfg) = |PnL_sim(S, D, cfg) − PnL_real(S, D)|` (also keep signed error + the
  distribution over seeds/windows, not just the point value).
- **Headline** = mean gap per `cfg`, and the **marginal reduction** `g(prev cfg) − g(cfg)` as each
  component is switched on — a waterfall ([FIG-9], [TAB-1]).

Sim-to-real protocol (P4.3): **calibrate every component on day A, then test on a HELD-OUT day B**
— the gap must be measured out-of-sample or it just reports overfitting.

## What already exists (extend, don't rebuild)
- `core/src/bin/backtest.rs` — the Phase-2 ablation: a naive MM through `{naive | +fill |
  +fill+latency | +fill+latency+impact}`, emitting `Stats{final_equity, fills, fees,
  cancelled_unfilled, ...}`. `final_equity` is PnL. The realism toggles (`Config.calibrated_latency`,
  `.impact`, calibrated fill model) are already wired.
- Calibrated components: fill (`core/src/fill*.rs`), latency (`latency.rs`), impact (`impact.rs`).
- The generative market (P3): `p36_autoregressive.py` produces a self-consistent generated book.

## The pieces Phase 4 needs

### 1. Strategies (the test subjects)
Have: naive fixed-offset MM. Add two more *parametric* subjects (cheap, no learning) so the ablation
isn't single-strategy — a realism component that helps one strategy may hurt another, and that
contrast is part of the result:
- **fixed-offset MM** (widen/narrow variants) — quote ±k ticks, already ~the naive one; parameterize k.
- **OFI-momentum taker** — cross the spread when order-flow imbalance exceeds a threshold (uses the
  Phase-1 OFI signal); a *taker* stresses fill/latency realism differently than a maker.
- **(stretch) RL MM** — small policy net via a Gym-style wrapper (ABIDES-Gym pattern, `refs/`);
  CPU-trainable. Defer until the parametric ablation runs end-to-end.

### 2. The generative-market config (+generative)
Add a 5th ablation config: run the strategy against the P3 generated book instead of the real
replay. Wire `p36`'s output (or a Rust reader of it) as a market source the backtest can consume,
so `PnL_sim(S, D, +generative)` is a real cell in the table.

### 3. The sim-to-real harness
`experiments/ablation/` — for each (strategy × cfg × day × seed): run the backtest, collect
`final_equity` + fills, write a tidy parquet (`strategy, cfg, day, seed, pnl, fills, ...`). Wrap
the Rust `backtest` binary (fast) driven by a Python orchestrator that sweeps the grid and computes
gaps. Emit the data for [FIG-9]/[TAB-1] as it runs (per CLAUDE.md's figure track).

### 4. Statistics
- Gap **distribution** over seeds & sub-windows, not a point — the components' effects are noisy.
- **Paired significance**: bootstrap CIs on the marginal reduction per component; a paired test that
  `g(+fill) < g(naive)` etc. Report effect sizes, not just p-values.
- Guard against the multiple-comparisons / seed-cherrypicking trap; fix seeds, log every cell.

### 5. Reproducibility — `make ablation`
A `Makefile` target that runs the full grid and regenerates the table + waterfall from one command
(PLAN's manual test: reproduce the table, spot-check three cells, confirm the significance tests run).

## The gating prerequisite (be honest): MULTI-DAY DATA
We have **one day** (`btcusdt_20260501`). The sim-to-real headline REQUIRES ≥2 days (calibrate on A,
test held-out B); a real result wants several. Budget is $0, so:
- **Tardis.dev free first-of-month** CSVs give ~1 day/month/symbol — slow to accumulate.
- **Self-recording collector** (`refs/hftbacktest/collector`, PLAN P0.2) accumulates our own days
  from live websockets starting now — the right long-game move; start it in parallel.
Until ≥2 days exist, Phase 4 can only be validated **in-sample** (calibrate and test on the same
day / disjoint windows of it) — useful for building and debugging the harness, but NOT the headline
out-of-sample number. Treat "acquire a second day" as a hard gate for the publishable result.

## Key decisions
| decision | recommendation |
|---|---|
| ground truth without L3 | naive-fill backtest on real data, documented as a proxy |
| strategies first | 2–3 parametric (MM + OFI taker); RL MM is a stretch, defer |
| gap over one number | keep the full distribution (seeds × windows) + signed error |
| data | start the self-recording collector NOW; in-sample harness meanwhile |

## Phased plan
1. **Harness + metric on 1 day, in-sample** — add the 2 parametric strategies, the tidy-parquet
   sweep, the gap computation, `make ablation`; validate the fill/latency/impact waterfall reproduces
   Phase-2 behavior. (No new data needed.)
2. **+generative config** — plug the P3 book in as the 5th cfg; sanity-check its gap vs the parametric configs.
3. **Second day** — acquire via Tardis free-of-month or the collector; run the true held-out
   sim-to-real; produce [FIG-9]/[TAB-1].
4. **(stretch) RL MM**, more days/symbols, significance hardening → the paper's §4–§5.

## Definition of done
`make ablation` reproduces a strategy × config PnL-gap table on ≥2 days, with the per-component
marginal gap reduction and its confidence interval — the sentence "adding component X reduces
backtest error by Y% (95% CI …)" backed by the harness. That sentence is the paper's headline.

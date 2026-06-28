# HFT-SimLab — Master Plan

**Objective (one sentence):** Build a market simulator whose backtest PnL provably matches reality —
and measure exactly which realism components (queue model, latency model, impact feedback) close
that gap, by how much. The measurement itself is the publishable research contribution.

**Working protocol (non-negotiable):**
1. Claude implements a step.
2. Claude writes a journal entry in `docs/JOURNAL.md`: what was done, the theory behind it,
   which paper/book chapter it comes from and how exactly it was used.
3. In the same session, the matching section of `paper/DRAFT.md` gets its academic-voice
   draft (parallel paper track — section mapping lives inside DRAFT.md).
4. The user runs the **manual test** for that step and confirms understanding.
5. Only then does work proceed to the next step.

**Hardware budget:** Apple M2, 8 GB RAM, ~24 GB free disk. Everything below fits.
**Money budget:** $0. Data = Tardis.dev free first-of-month CSVs + self-recorded websocket feeds.
Heavy GPU training (Phase 3 only) = Kaggle free tier (30 GPU-h/week).

---

## Reference repos (in `refs/`, shallow clones — read/copy from these, never edit them)

| Repo | What we take from it |
|---|---|
| `refs/hftbacktest` | **Our base.** Rust backtesting core: L2/L3 book reconstruction (`hftbacktest/src/backtest/data/`), queue models (`models/queue.rs`), latency models (`models/latency.rs`), exchange processors (`proc/*.rs`), Tardis/Binance data tooling (`collector/`, `examples/Data Preparation.ipynb`) |
| `refs/DeepMarket` | Official TRADES code: diffusion-transformer LOB generator (Phase 3) |
| `refs/abides-jpmc-public` | ABIDES: agent-based NASDAQ-like exchange + ABIDES-Gym RL wrapper (Phases 3–4) |
| `refs/LOBCAST` | Deep-learning LOB prediction baselines + LOB preprocessing patterns |
| `refs/liquibook` | Clean C++ matching-engine design reference for the Phase 1 warm-up |

## Paper map (which paper powers which step)

| Paper | Used in | What it gives us |
|---|---|---|
| Bridging the Reality Gap in LOB Simulation — arXiv 2603.24137 | P2.3, P2.4, P4 | The master recipe: project/estimate/validate/adapt; latency-race timing; power-law feedback kernel |
| Fill Probabilities in a LOB with State-Dependent Dynamics — arXiv 2403.02572 | P2.2 | How to estimate fill probability conditional on queue position + market state |
| Moallemi & Yuan, Queue Position Valuation | P2.2 | Why queue position has economic value; adverse-selection vs spread-capture decomposition |
| Unified theory of order flow, impact, volatility — arXiv 2601.23172 | P2.4 | Core-flow vs reaction-flow Hawkes decomposition; impact consistency checks |
| TRADES — arXiv 2502.07071 (+ `refs/DeepMarket`) | P3 | Diffusion-transformer architecture for generating realistic LOB event streams |
| DiffLOB — arXiv 2602.03776 | P3 (stretch) | Regime-conditioned counterfactual generation |
| TradeFM — arXiv 2602.23784 (PDF already in HFT folder) | P3 | Scale-invariant features, universal tokenization of trade flow |
| Attn-LOB MM — arXiv 2305.15821; Hawkes MM — arXiv 2207.09951 | P4 | RL market-maker baselines (action space, reward shaping) |
| Books on hand: Bouchaud *Trades, Quotes & Prices* (propagator/impact), Hasbrouck (microstructure econometrics), Cartea (optimal MM math), Harris (market mechanics) | throughout | Theory backbone — journal entries cite specific chapters |

---

## Phase 0 — Environment + first working backtest (~1 week)

**P0.1 Install Rust toolchain** (`rustup`, free), create `uv`-managed Python env.
**P0.2 Get data:** download Tardis.dev free first-of-month incremental L2 + trades CSVs
(Binance Futures BTCUSDT + one mid-cap alt for contrast); convert to hftbacktest `.npz` format
using their Data Preparation tooling. Start the self-recording websocket collector
(`refs/hftbacktest/collector`) so our own dataset accumulates from day one.
**P0.3 Run hftbacktest's example market-making backtest** end-to-end on that day of data.
- *Concepts taught:* market-by-price vs market-by-order feeds; incremental book updates vs
  snapshots; why tick data ≠ bar data; the anatomy of one backtest event loop.
- **Manual test (you):** run the example notebook, see a PnL curve, change one strategy
  parameter (spread width), rerun, observe the PnL change. Confirms toolchain + data pipeline work.

## Phase 1 — Foundations: build your own book + replay (warm-up, ~2–3 weeks)

**P1.1 Write a minimal L2 order book in Rust from scratch** (in `core/`): apply incremental
depth updates, maintain best bid/ask, depth arrays, detect crossed/locked books.
**P1.2 Replay harness:** stream a full day through it; assert book state consistency
(sequence numbers, no negative volumes); benchmark events/sec vs hftbacktest's reader.
**P1.3 Microstructure measurement notebook:** from your replayed book compute spread
distribution, depth profiles, order-flow imbalance, trade signature plots, autocorrelation
of returns and of signed order flow.
- *Concepts taught:* price-time priority; tick size and large-tick vs small-tick assets;
  stylized facts (fat tails, vol clustering, long-memory order flow — Bouchaud ch. on
  order-flow correlations); OFI as the workhorse microstructure signal.
- **Manual test (you):** pick 3 random timestamps; verify my book's top-5 levels exactly match
  hftbacktest's reconstruction at the same timestamps. Run the measurement notebook and
  visually check the stylized facts against the textbook plots.
- *Deliverable doubles as:* the "built an order book from scratch" portfolio piece +
  benchmark writeup.

## Phase 2 — The realism layer (the heart, ~4–6 weeks)

**P2.1 Read and document hftbacktest's existing models** (`models/queue.rs` — RiskAdverse,
probability queue models; `models/latency.rs` — constant/intp latency). Journal entry maps
each to its assumption and failure mode.
**P2.2 Calibrated fill-probability / queue model.** Estimate empirical fill curves from data:
for orders at queue position q with market state s (spread, imbalance, volatility), what is
P(fill within τ)? Implement as a new `QueueModel` in our fork; validate against *observed*
fills in L3-style data (Bybit provides order-level detail on some feeds; else infer from
trade prints vs book).
- *Papers:* arXiv 2403.02572 (estimation method), Moallemi-Yuan (interpretation).
**P2.3 Calibrated latency model with race dynamics.** Estimate the distribution of
feed→action→exchange round-trip from our recorded data timestamps; implement latency-race
clustering (the Reality Gap paper's finding: event-timing mode at exchange round-trip latency
= everyone reacting to the same trigger). New `LatencyModel` that samples from the fitted
distribution, with a race mode around signal events.
**P2.4 Market-impact feedback kernel.** The big one: replay no longer ignores us. Our filled
volume feeds a propagator: G(t) ~ power-law decay (Bouchaud propagator model — *Trades,
Quotes & Prices* part on impact), shifting subsequent simulated book prices; calibrate decay
exponent and amplitude from public impact estimates + our data; verify square-root impact law
and post-trade reversion emerge.
- *Papers:* Reality Gap recipe (arXiv 2603.24137), unified theory (arXiv 2601.23172) for
  consistency checks.
- **Manual tests (you), one per step:** (2.2) compare predicted vs realized fill rates on a
  held-out day — calibration plot should hug the diagonal; (2.3) histogram of simulated
  reaction times vs recorded ones; (2.4) run a buy-program strategy, plot impact during
  execution and reversion after — shapes must match the textbook curves.

## Phase 3 — Generative counterfactual market (~4 weeks, GPU on Kaggle free tier)

**P3.1 Adapt DeepMarket/TRADES** to our crypto data format; scope model to ~5–15 M params.
**P3.2 Train on Kaggle** (free 30 GPU-h/wk); run inference locally on M2 (MPS).
**P3.3 Stylized-fact validation suite:** generated streams must reproduce fat tails, vol
clustering, OFI autocorrelation, spread distribution — quantified (KS distances), not eyeballed.
**P3.4 Couple generator to the simulator:** market events come from the model *conditioned on
our order flow* → a market that reacts to us beyond the parametric kernel of P2.4.
- *Papers:* TRADES, DiffLOB, TradeFM tokenization ideas.
- **Manual test (you):** side-by-side dashboards — real day vs generated day; you should
  *not* be able to tell which is which from the stylized-fact panels.

## Phase 4 — Agent + the ablation experiment (the paper, ~4 weeks)

**P4.1 RL market maker** (small policy net, CPU-trainable) in our sim via a Gym-style wrapper
(pattern from ABIDES-Gym); baselines from Attn-LOB / Hawkes MM papers; also 2–3 simple
parametric strategies (fixed-offset MM, OFI-momentum taker) as non-RL test subjects.
**P4.2 Ablation harness:** run every strategy under simulator configs
{naive fills, +queue model, +latency model, +impact kernel, +generative market} × symbols × days.
**P4.3 Sim-to-real protocol:** calibrate on past days → simulate held-out future days →
compare to actual replay of those days → report PnL-gap distribution per config.
**The headline number:** marginal reduction in backtest error per realism component.
- **Manual test (you):** reproduce the full ablation table from one command
  (`make ablation`), spot-check three cells, confirm statistical significance tests run.

## Phase 5 — Write-up & release (~2–3 weeks)

arXiv preprint (q-fin.TR, endorsement via authors we build on) → open-source release with
docs → ICAIF workshop submission (deadlines late 2026) → upstream PR of the impact-feedback
model to hftbacktest for public credibility.

---

## Repo layout (ours)

```
hft-simlab/
├── PLAN.md            ← this file
├── CLAUDE.md          ← working protocol for every session
├── docs/JOURNAL.md    ← the learning journal (theory + papers + what changed)
├── refs/              ← read-only reference clones
├── data/              ← raw + converted market data (zstd-compressed, ~5 GB cap)
├── core/              ← Phase 1: our own Rust book + replay
├── simlab/            ← Phase 2+: our fork/extension of hftbacktest
├── genmarket/         ← Phase 3: TRADES adaptation
├── experiments/       ← Phase 4: ablation configs + results + analysis notebooks
└── paper/             ← Phase 5
```

## Risk register (honest)

- **8 GB RAM:** Phase 3 model must stay small; mitigated by Kaggle training + local inference.
- **Disk 24 GB:** cap data at ~5 GB compressed; rotate raw recordings after conversion.
- **L3 data scarcity in crypto:** fill-model validation may rely on inference from trades+book
  rather than true order IDs; the journal will document the assumption explicitly (reviewers ask).
- **Scope creep:** DiffLOB conditioning and equities (LOBSTER) are stretch goals — cut first.

# HFT-SimLab

**A market simulator whose backtest PnL provably matches reality — and a measurement of
exactly which realism components (queue/fill model, latency model, impact feedback) close
that gap, and by how much.** The measurement itself is the intended research contribution.

Naive backtests lie in four mechanical ways: they over-fill passive orders (no queue),
act on stale-free book state (no latency), replay a frozen tape (no impact), and ignore that
other participants would react (no feedback). HFT-SimLab models each lie honestly, **calibrates
it from real data**, and ablates it — quantifying the PnL-prediction error each one costs.

See [`PLAN.md`](PLAN.md) for the full roadmap and [`docs/JOURNAL.md`](docs/JOURNAL.md) for the
learning journal (theory + papers + every design decision, written as the project progresses).

## Status

| Phase | Description | State |
|-------|-------------|:-----:|
| **P0** | Data pipeline (Tardis.dev L2+trades → `.npz`) + naive market-making baseline | ✅ |
| **P1.1** | Minimal L2 order book in Rust, from scratch (`lob-core`) | ✅ |
| **P1.2** | Streaming `.npz` replay + level-for-level parity vs hftbacktest + benchmark | ✅ |
| **P1.3** | Microstructure measurement: stylized facts (OFI, fat tails, long-memory flow) | ✅ |
| **P2.1** | Reconnaissance: document hftbacktest's queue + latency models and their failure modes | ✅ |
| **P2.2** | Calibrated fill-probability model (virtual-order labeling + isotonic regression) | ✅ |
| **P2.3** | Calibrated latency model with race dynamics (log-normal + race-aware jitter) | ✅ |
| **P2.4** | Market-impact feedback kernel (Bouchaud propagator, square-root law) | ✅ |
| **P2.5** | Phase-2 integration: unified ablation backtest `{naive\|+fill\|+latency\|+impact}` | ✅ |
| **P2.6** | Event-level fills (continuous-time scheduling) — latency layer now measurable | ✅ |
| **P3.1** | Generative market: L2→TRADES/LOBSTER adapter + model scoping | ✅ |
| **P3.2** | Train TRADES on Kaggle (scaffold + driver; ~10M-param model trains) | ◐ |
| **P3.3** | Stylized-fact validation suite (KS distances, real-vs-real floor verified) | ✅ |
| **P3.4** | Generate from checkpoint, score realism, couple generator to simulator | ⏳ |
| **P4** | RL market-maker + the ablation experiment (the headline result) | ⏳ |
| **P5** | Write-up & open-source release | ⏳ |

## Layout

```
hft-simlab/
├── core/          Rust crate `lob-core`: order book, streaming replay, latency & impact models
│   ├── src/         book.rs · events.rs · npz.rs · latency.rs · impact.rs
│   └── src/bin/     replay.rs · fill_labeler.rs · latency_profiler.rs
├── scripts/       Python (uv): data conversion, calibration, parity, benchmarks (p0…p24)
├── experiments/   figure scripts + rendered figures (intermediate parquet is gitignored)
├── docs/          JOURNAL.md — the learning journal
├── paper/         DRAFT.md · FIGURES.md · NOTES.md — the parallel paper track
├── refs/          read-only reference clones (not committed — see refs/README.md)
└── data/          market data (not committed — regenerable, see scripts/p0_convert.py)
```

## Reproduce

```bash
# 1. Rust core (zero deps beyond flate2)
cd core && cargo test && cargo build --release

# 2. Python analysis env (uv)
uv venv && uv pip install -r requirements.txt   # polars, scikit-learn, matplotlib, scipy

# 3. Reference repos (Phase 3+ needs these)
#    see refs/README.md for the five clone commands

# 4. Data: download Tardis.dev free first-of-month BTCUSDT L2+trades, then
python scripts/p0_convert.py    # → data/*.npz
```

Hardware target: Apple M2, 8 GB RAM. Budget: $0 (free Tardis.dev data + Kaggle free-tier GPU for Phase 3).

## License

MIT (`core/`). Reference repositories under `refs/` retain their own licenses.

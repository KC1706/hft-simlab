# HFT-SimLab Learning Journal

Every major change gets an entry: **what was done → theory behind it → which paper/book and
how exactly it was used**. Written for revision — read top to bottom and you re-learn the
project's whole intellectual arc.

---

## Entry 0 — Project setup, reference repos, and the foundations (2026-06-10)

### What was done
- Created the project skeleton (`hft-simlab/` with `refs/`, `docs/`, `data/`).
- Shallow-cloned 5 reference repos into `refs/` (~190 MB): hftbacktest, DeepMarket (TRADES),
  ABIDES, LOBCAST, liquibook. These are read-only — we copy patterns/code *out* of them.
- Verified toolchain: git, Python 3.14, uv, Xcode CLT present; Rust to be installed in P0.1.
- Wrote `PLAN.md` (the end-to-end plan) and `CLAUDE.md` (the working protocol).

### Theory 0.1 — What a limit order book (LOB) actually is

Modern exchanges are **continuous double auctions**. Anyone can post a *limit order*
("buy 2 BTC at ≤ $60,000") which rests in the book, or a *market order* ("buy 2 BTC now")
which consumes resting orders. The book is just two priority queues:
- **Bids** (buy orders), best = highest price.
- **Asks** (sell orders), best = lowest price.
- **Spread** = best ask − best bid. **Mid** = their average.

Matching follows **price-time priority (FIFO)**: better-priced orders fill first; among equal
prices, *earlier* orders fill first. That second clause is the seed of half this project:
**where you stand in the queue at a price level determines whether you get filled at all, and
which fills you get.**
→ Book reference: Harris, *Trading and Exchanges* — Part II (market structures); the cleanest
mechanical description in print.

### Theory 0.2 — Market data feeds: MBP vs MBO

Exchanges publish the book in two granularities:
- **L2 / Market-by-Price (MBP):** aggregate volume per price level ("$60,000 bid: 14.2 BTC
  total"). You see the level, not the orders inside it. This is what Binance/Bybit websockets
  give the public.
- **L3 / Market-by-Order (MBO):** every individual order add/cancel/execute with order IDs
  (NASDAQ ITCH, LOBSTER data). You can reconstruct exact queues.

Crypto's public feeds are L2 — so queue position must be *modeled*, not observed. That's why
hftbacktest has "probability queue models" and why our Phase 2.2 exists. In `refs/hftbacktest`,
the relevant code is `hftbacktest/src/backtest/models/queue.rs` (queue models) and
`data/reader.rs` (feed reconstruction).
→ Book reference: Hasbrouck, *Empirical Market Microstructure* — ch. 1–2 frame what's
observable vs latent in market data.

### Theory 0.3 — Why naive backtests lie (the project's reason to exist)

Four mechanical lies, in increasing subtlety:
1. **Fill lie:** naive backtester fills your limit order when price *touches* it. Reality:
   you're behind a queue; price often touches and retreats, filling only people ahead of you.
   Worse — **adverse selection**: the fills you *do* get cluster in moments when price is about
   to move against you (informed flow just ran over your level). So naive backtests overstate
   both fill *rate* and fill *quality*.
2. **Latency lie:** the backtest acts on the same book state that generated the signal. Reality:
   your order arrives 0.1–50 ms later into a *different* book — and during races, after faster
   players already reacted to the same trigger.
3. **Impact lie:** historical replay is a recorded movie; your simulated trades don't move
   prices. Reality: trading pressure shifts prices against you (concave, ~square-root in size),
   then partially reverts after you stop.
4. **Feedback lie:** other participants would have *reacted* to your orders (quotes pulled,
   spreads adjusted). Replay can't know this; only a generative/agent-based market can fake it.

Phases 2.2–2.4 and 3 attack lies 1–4 respectively. The ablation study (Phase 4) measures the
cost of each lie in PnL-prediction error — the number nobody has published.
→ Bouchaud, *Trades, Quotes and Prices*: impact and order-flow chapters are the theory
spine for lies 3–4 (propagator model, square-root law, flow long-memory).

### Theory 0.4 — Why each reference repo was chosen

- **hftbacktest** (Rust): the only open backtester that already models queues + latency on full
  tick data. We extend rather than reinvent; our novelty = *calibrated* models + impact
  feedback + the ablation harness. Extension points found today: `models/queue.rs`,
  `models/latency.rs`, exchange processors in `proc/`.
- **DeepMarket/TRADES**: official code of the diffusion-transformer LOB generator
  (arXiv 2502.07071) — Phase 3 starts from here, retargeted to our crypto data.
- **ABIDES** (JPMorgan): the academic-standard agent-based exchange; its Gym wrapper is our
  Phase 4 RL pattern; also TRADES's own simulation host, so compatibility is proven.
- **LOBCAST**: benchmark suite of LOB deep-learning predictors — baseline zoo + preprocessing
  conventions.
- **liquibook** (C++): clean matching-engine architecture to study before writing our own
  minimal book in Phase 1 (header-only, easy to read in one sitting).

### Papers anchoring the project (read order for you)
1. arXiv 2603.24137 *Bridging the Reality Gap in LOB Simulation* — read first; it is the
   project's blueprint (project → estimate → validate → adapt).
2. arXiv 2403.02572 *Fill Probabilities in a LOB* — the estimation machinery for Phase 2.2.
3. Moallemi & Yuan *Queue Position Valuation* — the economics of queue position.
4. arXiv 2601.23172 *Unified theory of order flow, impact, volatility* — the consistency
   checks for Phase 2.4.
5. arXiv 2502.07071 *TRADES* — Phase 3 architecture (code in `refs/DeepMarket`).
6. (Already on disk) TradeFM 2602.23784 — tokenization ideas for Phase 3.

### Open questions carried forward
- Which second symbol alongside BTCUSDT? Want contrast in tick regime (large-tick vs
  small-tick behaviour differs sharply — Reality Gap paper is about large-tick assets).
  Decide in P0.2 after looking at spread/tick ratios.

---

## Entry 1 — Phase 0 complete: toolchain, real tick data, first backtest (2026-06-10)

### What was done
- Installed Rust 1.94 (`rustup`) and a Python 3.12 venv (`uv`) with hftbacktest 2.4.4.
- Downloaded the **free** Tardis.dev first-of-month datasets for Binance Futures BTCUSDT,
  2026-05-01: `trades` (25 MB gz, 3.17M rows) and `incremental_book_L2` (532 MB gz, **101M rows**).
  Gotcha found: `datasets.tardis.dev` returns 404 to HEAD requests — only GET works.
- Sliced to the first ~7 hours (25M book rows), converted to hftbacktest's `.npz` event format
  (`scripts/p0_convert.py`), ran a naive market-making backtest (`scripts/p0_backtest.py`),
  and produced `experiments/p0_baseline_equity.png`. Then deleted raw slices (disk budget).

### Theory 1.1 — Anatomy of the data you just bought for $0
Tardis records the **raw exchange feed**: every L2 delta ("price level 60000.1 bid now has
14.2 BTC") plus every trade print, each with two timestamps — `timestamp` (exchange's) and
`local_timestamp` (Tardis's receive time). The difference is *feed latency*: the first
ingredient of Phase 2.3. Subtlety from the converter source (`tardis.py:67`): Binance Futures
publishes the 'E' (send) time, not 'T' (match) time, so measured latency slightly understates
reality — we log this as a known bias for the paper's limitations section.

### Theory 1.2 — Why trades must be converted BEFORE depth updates
A trade does two things at once: it prints (someone got filled) and it removes volume from a
book level. The exchange then also sends a depth delta reflecting that removal. If the
backtester processed the depth delta first and the trade second, a simulated order's queue
position would be decremented **twice** for one real event — silently inflating your fill
rate. hftbacktest's converter therefore demands `[trades, depth]` input order
(`tardis.py:75-80`). This is exactly the kind of mechanical detail that separates honest
backtests from fantasy — lie #1 of Entry 0 hides in details this small.

### Theory 1.3 — The event format and why 101M rows broke our first plan
Each event becomes a fixed 64-byte record: `(ev_flags u64, exch_ts i64, local_ts i64,
px f64, qty f64, order_id u64, ival i64, fval f64)`. Fixed-width binary means the backtester
can memory-map and stream it with zero parsing — that's why one 7-hour backtest (26.7M events)
runs in about a minute. But 101M rows × 64 B ≈ 6.5 GB *just for the conversion buffer* — more
than this machine's 8 GB RAM. Lesson: **at HFT data scales you size memory before you run,
not after it crashes.** We sliced to 25M rows; Phase 1's Rust replay will stream instead,
removing the limit entirely.

### Theory 1.4 — Reading the first backtest like a microstructurist
The strategy: quote both sides at mid ± 20 ticks, skew quotes against inventory (the simplest
form of the Avellaneda-Stoikov/Cartea inventory-control idea — Cartea, *Algorithmic & HFT*,
ch. 10), post-only orders (GTX = never cross the spread = always pay maker fee, not taker).
Configuration is deliberately the project's **naive baseline**: constant 10ms latency and
hftbacktest's `risk_adverse_queue_model` (pessimistic: you're always at the queue's back).

Result: SR ≈ −685, return-over-trade ≈ −3 bps/trade, turnover ~99×/day — a steady bleed.
Why? **Adverse selection**: passive quotes get filled disproportionately when the market is
about to move through them (the informed/fast flow hits you, the harmless flow doesn't reach
you). With naive quoting, the spread you earn < adverse selection cost + fees (Moallemi-Yuan's
decomposition from PLAN P2.2; Bouchaud TQP's "spread vs impact" balance). A real MM survives
by better queue position, faster reaction, and smarter skew — precisely the dynamics Phases
2.2–2.4 will model honestly. **Keep this number: it is the project's first data point — the
naive-baseline cell of the final ablation table.**

### Paper used today
- hftbacktest's converter + docs (engineering source of truth). The Reality Gap paper
  (2603.24137) framed what we logged for later: feed-latency distribution and the E-vs-T
  timestamp bias both feed §3.3 and Limitations.

### Debugging addendum (2026-06-11) — the blank PNG
First bug of the project, and a classic: `stats.plot()` internally calls `plt.close()` and
*returns* the Figure (`refs/.../stats/stats.py:245-247`). Our script ignored the return value
and called `plt.savefig()`, which saves pyplot's *current* figure — by then a fresh empty one.
White image. Fix: `fig = stats.plot(); fig.savefig(...)`. Lesson: when a library misbehaves,
read its source — we have every dependency's source in `refs/` for exactly this reason.

**Reading the fixed plot (do this with the PNG open):** top panel — blue (equity after fees)
bleeds to −0.9% of book size over 7h; orange (equity *before* fees) also falls (−0.35%), so
fees are NOT the main problem — the trading itself loses, confirming adverse selection. Note
the sharp equity drop near 03:00 right where the grey price line falls fast: a fast move ran
over our resting quotes, filled us into a losing inventory — adverse selection made visible.
Bottom panel — position pinballs between the ±0.005 caps: the naive skew barely manages
inventory; it gets pushed to a cap and waits to be run over.

### Manual test for you (P0 gate)
1. `cd hft-simlab && .venv/bin/python scripts/p0_backtest.py` — should print the stats table
   and rewrite `experiments/p0_baseline_equity.png`. Open the PNG, look at the equity curve.
2. Edit `scripts/p0_backtest.py`: change `half_spread = 20.0 * tick` to `60.0 * tick`, rerun.
   Predict first, then check: fewer fills, lower turnover — does PnL improve or worsen? Why?
   (Hint: wider quotes = more spread earned per fill but worse queue position and fewer fills;
   adverse selection changes too.)
3. Write one sentence in your own words: why did the naive MM lose money? If you can't,
   re-read Theory 1.4 before we proceed to Phase 1.

---
## Entry 2 — P1.1: A minimal L2 order book in Rust, from scratch (2026-06-12)

### What was done
- Created `core/` — our first own Rust crate, `lob-core` (zero dependencies, deliberately):
  - `src/book.rs` — `L2Book`: applies incremental MBP depth updates, maintains best bid/ask,
    serves top-N depth arrays, detects locked/crossed updates. 12 unit tests including a
    100k-update randomized soak that asserts the book can never stay crossed.
  - `src/events.rs` — `Event` struct + flag constants, **byte-compatible with hftbacktest's
    `.npz` rows** (64-byte aligned, same field order), so the P1.2 replay harness can mmap
    the exact BTCUSDT file from Phase 0. Copied from `refs/.../types.rs:150-331` with attribution.
  - `examples/ladder_demo.rs` — feeds a handcrafted sequence and prints the ladder after each
    step; step 4 shows a crossing update being detected and resolved live.
- `cargo test`: 14/14 pass. Demo verified.

### Theory 2.1 — An L2 book is a *mirror*, not a matching engine
The exchange's matching engine (liquibook in `refs/` is a clean C++ one) holds every order and
matches by price-time priority. What the public L2 feed broadcasts is only the *aggregate*
quantity per price level — the shadow the engine casts. So our book does no matching at all:
it applies **absolute** updates ("level 50000.1 now holds 3.2 BTC" — *not* a delta; qty 0
means the level vanished) and its only intelligence is keeping the best-price pointers honest.
Everything inside a level — the queue of individual orders, and our place in it — is invisible
in L2. Hold that thought: P2.2's entire job is inferring that invisible queue.
→ Harris, *Trading & Exchanges*, ch. 4 (order types) and the order-driven-markets chapter
(precedence rules); Bouchaud *TQP*, the limit-order-book chapter of Part II, for the book as
the central data structure of modern markets.

### Theory 2.2 — Integer ticks, or: never use a float as a price key
Prices live on a grid: BTCUSDT-perp's tick size is 0.1. But f64 cannot represent 0.1 exactly,
and `50000.1 / 0.1 = 500000.999…` — truncate that and your level lands one tick off; use the
float itself as a map key and "the same price" from two code paths may be two different keys.
So the book keys on `round(px / tick_size) as i64` and only converts back for display (the
test suite asserts ticks with `==` but prices only approximately — `500001 × 0.1 ≠ 50000.1`
in f64, ~381 ulps off!). Same convention as hftbacktest (`hashmapmarketdepth.rs:92`).
Conceptual bonus: tick size is economics, not plumbing — a **large-tick** asset (spread pinned
at 1 tick, fat queues, queue position is everything) trades differently from a **small-tick**
one (thin levels, spread breathes). BTCUSDT-perp at 0.1 ticks on a ~$100k price is *small-tick*;
our P0 spread distribution will confirm it. Bouchaud *TQP* discusses the large/small-tick
dichotomy early in the LOB part — worth reading now; it decides how much P2.2's queue model
will matter per symbol (a planned contrast in the ablation).

### Theory 2.3 — Locked and crossed books: feed artifacts with information in them
A **locked** book has bid == ask (spread 0); a **crossed** book has bid > ask. A matching
engine can't rest in either state — it would just match. But an L2 *feed consumer* sees them
transiently: diff batches arrive with one side updated before the other, or a stale level
lingers after the venue moved. (In US equities locked/crossed markets are a *regulatory*
concept across venues — Reg NMS; here, single-venue, they are pure feed-staleness artifacts.)
Two design options when an update crosses the standing book:
1. **Hard-resolve:** delete the opposite-side levels the update crossed (assume they're gone).
2. **Trust-the-feed (hftbacktest's choice, and ours):** move the best pointer past the stale
   levels but *keep their entries* — the feed will refresh them shortly. Their doc comment
   (`hashmapmarketdepth.rs:10-20`) explains why: with missing feed messages, aggressive
   deletion makes errors permanent, pointer-skipping lets the book self-heal.
We replicate option 2 *exactly* — including which side's pointer moves and the strict
inequality bounds of the rescan (`depth_above(start+1..)`, `depth_below(..start)`) — because
P1.2's manual test is a level-for-level match against hftbacktest's reconstruction. We add
what they don't have: `locked_updates`/`crossed_updates` counters. Those are data-quality
telemetry — a day with thousands of crossings means a degraded feed, which would silently
poison every Phase 2 calibration. Cheap insurance, one branch each.

### Theory 2.4 — BTreeMap vs HashMap: our one deliberate divergence
hftbacktest stores each side as `HashMap<tick, qty>` + cached best pointers + low/high
watermarks, and rescans tick-by-tick when the best is deleted — O(1) updates, but the rescan
walks *every* tick in the gap, and ordered iteration (top-N) isn't free. We store
`BTreeMap<tick, qty>`: O(log n) updates, but next-best is one `range()` call over *existing*
levels and top-5 depth arrays fall out of ordered iteration — which P1.3's measurement
notebook needs constantly. Same observable semantics, different engine. P1.2 will benchmark
both honestly on 100M real events; expect the HashMap to win raw event throughput and the
BTreeMap to win when every event also reads depth arrays. That benchmark becomes the
portfolio writeup.

### Found while reading the refs (keep doing this)
hftbacktest's `clear_depth` recomputes the post-clear best bid via `depth_below(clear_upto-1,…)`
(`hashmapmarketdepth.rs:208-209`), whose scan is *strictly below* its start — so a level at
exactly `clear_upto − 1` gets skipped. Harmless in practice (a full snapshot always follows a
clear and rebuilds the pointers), but it's an off-by-one we chose not to copy: our
`clear_side` rescans actual map keys. Lesson: read reference code critically, even good
reference code — and note where you diverge and why, because parity tests will find the
difference before you remember it.

### Paper used today
None — PLAN.md's paper map assigns no research paper to P1.1; this step's sources are
engineering: `refs/hftbacktest/.../hashmapmarketdepth.rs` + `types.rs` (semantics we mirror,
with attribution in our doc comments) and a structural glance at liquibook's `depth.h`
(aggregate-by-price levels for display — same mirror idea in C++). The Reality Gap recipe
(arXiv 2603.24137) resumes duty in Phase 2.

### Manual test for you (P1.1 gate)
1. `cd hft-simlab/core && cargo test` — expect 14/14 green. Skim the test *names*: they are
   the book's spec in one screen.
2. `cargo run --example ladder_demo` — read all 5 steps against Theory 2.1–2.3: step 2 is
   absolute-vs-delta semantics, step 3 best-price promotion, step 4 a crossing update
   (watch `crossed: 1` appear while the displayed ladder stays uncrossed), step 5 the feed
   refreshing the stale level away.
3. Predict-then-verify: in `examples/ladder_demo.rs`, change step 4's price `50_000.3` to
   `50_000.9` (crossing past *every* ask). Predict the ladder first — what is the best ask
   after the update? Then run it. (Answer shape: pointer skips past all asks → ask side
   displays empty/one-sided, all three stale entries retained internally, `state: BidOnly`.)
4. Say in one sentence why the book keeps stale crossed levels instead of deleting them —
   if stuck, re-read Theory 2.3.

## Entry 3 — P1.2: Replay harness, parity test, and the benchmark (2026-06-13)

*(Autopilot note: from this entry on, the user has enabled autopilot — I run the manual
tests myself and report results; the exercises at the end remain for the user's review.)*

### What was done
- `core/src/npz.rs` — streaming `.npz` reader: hand-parsed zip container (End-Of-Central-
  Directory trailer → central directory → local header → raw deflate stream via `flate2`,
  our single new dependency) + hand-parsed npy header with a strict dtype check that fails
  loudly on any layout drift. Constant memory: the 1.7 GB day never exists in RAM at once.
- `core/src/bin/replay.rs` — the harness: streams the day, applies LOCAL-flagged events to
  our `L2Book` (dispatch mirrors `refs/.../proc/local.rs:290+`), validates consistency,
  measures throughput, and can snapshot the top-N ladder at given timestamps (JSON out).
- `core/tests/fixtures/tiny.npz` + `tests/replay_npz.rs` — 10-row numpy-written fixture
  covering clear+snapshot, a trade, and a crossing; tests the container parsing end to end.
- `scripts/p12_parity.py` — the P1 gate: 3 seeded-random timestamps, our top-5 vs
  hftbacktest's `HashMapMarketDepth` elapsed to the same instants.
- `scripts/p12_bench.py` — honest throughput comparison.

### Results (run on the BTCUSDT 2026-05-01 00:00–06:56 file, 26,663,697 events)
- **Consistency: zero violations.** local_ts and exch_ts both monotone, no negative
  quantities, book never crossed after any update.
- **Parity: PASS.** All top-5 levels — integer ticks AND quantities — identical to
  hftbacktest's reconstruction at all 3 random timestamps (+122.6, +195.3, +287.9 min).
  The P1.1 decision to mirror their stale-level semantics paid off exactly here.
- **Telemetry:** 3,809 crossing + 319 locking updates in ~7 h (≈1 per 6,500 events) —
  normal transient feed staleness, now quantified per day for free.
- **Throughput:** ours 3.1 s end-to-end ≈ **8.6 M events/s**; hftbacktest ≈ 9.2 M events/s
  end-to-end (their elapse alone 10.4 M ev/s after a 0.33 s "load" — their reader defers
  real decompression into the run, overlapping it with processing).

### Theory 3.1 — Containers: what a .npz actually is, and why we stream it
`.npz` = a zip archive of `.npy` members; `.npy` = magic + version + a Python-dict header
(dtype, shape, order) + raw little-endian records. Two non-obvious bits: (1) the *trailer*
(EOCD record) is the authoritative index of a zip — local headers may carry zeroed sizes
(data-descriptor mode), so robust readers parse from the end; (2) numpy structured dtypes
map 1:1 onto a `#[repr(C)]` struct as long as field order, widths, and padding agree — our
`events.rs` layout test is what makes that contract checkable. We stream because the
machine has 8 GB RAM and the raw array is 1.7 GB and growing with each recorded day;
decompressing on the fly costs nothing here (see Theory 3.4).

### Theory 3.2 — Two clocks per event, and the bug I hit
Every event carries `exch_ts` (when the exchange says it happened) and `local_ts` (when our
collector received it). The *local* book — what a strategy could actually have known — is
defined by: apply every LOCAL-flagged event with `local_ts ≤ T`. That definition is what
both sides of the parity test implement, which is why they can agree bit-for-bit.
Subtlety found in the data: ~955 k rows (3.7%) are EXCH-only duplicates — the converter
splits an event into exchange-clock and local-clock copies whenever one ordering would
violate the other, so each processor sees a monotone stream (our validator confirming
`local_ts_order=0` is the converter's guarantee, verified).
And the bug: my parity script first assumed a fresh backtest's `current_timestamp` was the
data start. Empirically it is `i64::MAX` — a sentinel for "unset" — and `elapse()` measures
from the file's earliest timestamp. The book came back empty, the script said "fewer than
5 levels", and a 10-line probe script gave the real semantics in one run. Lesson repeated
from Entry 1's blank PNG: when an API surprises you, *probe it empirically* — print the
actual values, don't reason from what the API "should" do.

### Theory 3.3 — Trades don't touch the book
The replay counts trades (710,636) but never applies them to depth. An L2 feed reports the
book *after* the matching engine already removed the traded volume — the depth delta
arrives as its own event. Apply both and you double-decrement: this is the same
double-count Entry 1 flagged for queue positions (trades sorted before their depth deltas),
now showing up as a book-construction rule. Trades matter for *flow* measurement — which
is exactly P1.3: they carry the aggressor side (who crossed the spread), the raw material
of order-flow imbalance and the trade-signature plots.
→ Bouchaud *TQP*: order flow chapters; Hasbrouck ch. on trades vs quotes data.

### Theory 3.4 — What the benchmark actually taught
P1.1 predicted "HashMap wins raw throughput, BTreeMap wins when reading depth arrays".
Reality: both stacks land at ~9 M events/s and the difference is noise, because **per-event
data-structure cost is drowned by stream decompression and memory traffic**. The real
lessons: (a) measure before optimizing — the asymptotic argument was real but irrelevant at
this scale; (b) hftbacktest hides its 1.7 GB decompression *inside* the run via lazy/
background loading, a systems trick worth remembering; (c) 26.6 M events over 416 min is
~1,070 events/s *average* but arrives in bursts thousands of times denser — throughput
headroom (~9 M/s vs ~1 k/s) is what makes Phase-2's per-event model overhead affordable.

### Paper used today
None new — engineering references: PKWARE APPNOTE (zip format), numpy's NEP-1 (npy format),
`refs/.../proc/local.rs` (event dispatch parity). The parity methodology itself (validate a
reimplementation against a reference at randomly sampled states) is standard simulation
hygiene and goes into §3.1's validation sentence.

### Manual exercises for you (autopilot already ran the gates)
1. `cd hft-simlab/core && cargo test` (15 tests now) and
   `./target/release/replay ../data/btcusdt_20260501_0000_0656.npz` — read the report line
   by line; every number should now mean something to you.
2. `.venv/bin/python scripts/p12_parity.py` — change `SEED`, rerun: parity must hold at ANY
   timestamps. If you want to see it fail honestly, flip a `>=` to `>` in
   `core/src/book.rs`'s crossing branch, rebuild, rerun (then revert!).
3. Explain to yourself why trades must not be applied to the book (Theory 3.3) — this will
   be on the "exam" when we build the queue model.

## Entry 4 — P1.3: Measuring the market's fingerprint (2026-06-13)

### What was done
- `replay --measure <dir>` (Rust): logs two datasets in one streaming pass —
  `samples.csv`: every 100 ms, top-of-book, top-10 level quantities per side, per-interval
  **OFI** (Cont–Kukanov–Stoikov), signed trade volume/counts; `trades.csv`: the full tape
  (ts, aggressor sign, price tick, qty). Still 5.5 M events/s with measurement on.
- `scripts/p13_measure.py`: CSV → parquet (`experiments/data/p13/`, ~16 MB — also FIG-7's
  future real-side input). `experiments/figures/p13_stylized_facts.py`: renders **FIG-10**
  (spec added to FIGURES.md) and writes `stats.json`. Output:
  `experiments/p13_stylized_facts.png`.

### Results — the dataset's fingerprint (BTCUSDT-perp, 2026-05-01, 416 min)
| fact | number | textbook expectation |
|---|---|---|
| P(spread = 1 tick) | **0.999** | large-tick regime: pinned spread |
| excess kurtosis, 1s returns | **108** | fat tails (Gaussian = 0) |
| ACF returns @1s | 0.08, →0 fast | no linear predictability |
| ACF \|returns\| @60s | 0.10, slow decay | volatility clustering |
| trade-sign ACF | power law, slope **−0.63** over 4 decades | long-memory order flow (γ∈0.4–0.7) |
| OFI→Δmid @1s | β=0.94 ticks/unit, **R²=0.48** | CKS: OFI explains ~half of price moves |
| R(τ=1s) response | 64 ticks (event-conditional) | concave, rising with τ |

### Theory 4.1 — I was wrong in Entry 2, and the data said so
Entry 2 reasoned: tick 0.1 on a ~$76k price = 0.13 bp relative tick → "small-tick asset,
spread breathes". **Refuted:** the spread sits at exactly 1 tick 99.93% of the time — fully
pinned, the defining *large-tick* signature. Why the armchair rule failed: tick size
relative to *price* is tiny, but what matters is tick size relative to *volatility per
decision horizon* and the venue's fee/queue economics — BTCUSDT-perp is so liquid that
sub-tick spreads would be profitable to quote, so the book compresses to the floor.
Consequence for the project: queue position is **everything** here (a pinned spread means
you can't improve the price — you can only join the queue), so P2.2's queue model should be
the ablation's biggest lever on this symbol. The PLAN's mid-cap-alt contrast dataset just
became more important — we need a genuinely small-tick symbol for comparison.
Lesson: write predictions down (Entry 2 did) so the data can grade them. That's the paper's
whole epistemology in miniature.
→ Bouchaud *TQP*, large-tick vs small-tick discussion (Part II); Harris ch. on tick size
   and price clustering.

### Theory 4.2 — OFI: the workhorse signal, and the one-slot bug
Order-flow imbalance (Cont–Kukanov–Stoikov, **arXiv 1011.6402** — paper used today, in
code): each change at the best quotes contributes
e = ΔW_bid − ΔW_ask (queue growth at the bid is buying pressure; at the ask, selling). Sum
over a window, regress concurrent mid change on it: **β=0.94 ticks per unit BTC, R²=0.48**
— half the variance of 1-second price moves explained by one linear book-pressure variable.
This is the strongest short-horizon relationship in microstructure, and it emerged from OUR
book, which is the real point: the measurement pipeline is now trustworthy end to end.
The bug that almost hid it: my first pass grouped OFI seconds as slots [10g..10g+9] while
returns spanned slots (10g..10g+10] — one slot of misalignment, R² collapsed to 0.01 (47×
smaller). Time-series alignment errors don't crash; they silently destroy signal. The fix
is a comment in the figure script now. Rule: when a known-strong effect measures weak,
suspect your clocks/indices before doubting the effect.

### Theory 4.3 — Long memory and the response function (the impact preview)
Panel (e): the autocorrelation of trade *signs* decays as a power law (slope −0.63) over
four decades of lag — order flow is extraordinarily persistent (metaorders are sliced; the
market is full of half-finished executions). Panel (f): R(τ) = E[sign·(mid(t+τ)−mid(t))]
rises concavely — trades move price, and the move builds with horizon. Bouchaud's puzzle
(*TQP*, propagator chapters): if flow is this predictable and each trade impacts price, why
aren't prices trivially forecastable? Resolution: impact must *decay* (the propagator
G(τ)) in just the way that cancels the flow's long memory — this tension is exactly what
P2.4's impact kernel implements and calibrates. Note on magnitudes: R(1s)=64 ticks looks
huge vs the 31.5-tick *unconditional* 1s std — but conditional on trading, std is 138
ticks: trades cluster in violent moments (vol clustering seen from the other side).
→ Bouchaud *TQP* (order-flow correlations; propagator); Hasbrouck (trades-vs-quotes
   information content).

### Data notes (for the paper's honesty box)
- 671,393 LOCAL trade rows vs 710,636 total trade-kind rows: the converter's EXCH-only
  duplicates again (~5.5%) — the local tape is the deduplicated truth we measure on.
- 5 grid samples skipped one-sided (session-start snapshot warm-up) — handled, logged.
- This was a violent session (max 1s move: 980 ticks ≈ $98); single-day stats are a
  baseline, not estimates — multi-day data arrives with the collector (P0.2's recorder).

### Paper used today
**arXiv 1011.6402** (Cont, Kukanov & Stoikov, *The price impact of order book events*) —
defines the OFI variable our Rust accumulator implements per top-of-book transition, and
the OFI→return regression that validates it (R²=0.48 matches their reported range).
Added to Related Work. FIG-10 spec'd and produced; its parquet doubles as FIG-7's real side.

### Manual exercises for you
1. Open `experiments/p13_stylized_facts.png` next to this entry's table; check each panel
   against its row. Panel (e) is the prettiest: that straight line on log-log axes is the
   long memory of order flow.
2. Rerun `experiments/figures/p13_stylized_facts.py` after changing the OFI grouping back
   to `arange(0, n_grid, 10)` — watch R² collapse 0.48 → 0.01. One slot. (Revert!)
3. Say out loud what a pinned 1-tick spread means for a market maker who cannot improve
   the price. (Answer: the queue is the game — Theory 4.1.)

---
**PHASE 1 COMPLETE** — own book (parity-verified), replay at 8.6 M ev/s, measurement
pipeline with the stylized facts in hand.
*(Next entry: P2.1 — read and document hftbacktest's queue + latency models: each model's
assumption, and the precise way reality will violate it.)*

---

## Entry 5 — P2.1: Reconnaissance — hftbacktest's existing queue and latency models (2026-06-13)

### What was done
Read `refs/hftbacktest/hftbacktest/src/backtest/models/queue.rs` and `models/latency.rs` in
full. No code written — this step is documentation only. Goal: understand exactly what
assumptions each model bakes in, so that when P2.2 and P2.3 replace them we know what we
are improving and can quantify it.

---

### Model 1: `RiskAdverseQueueModel` (queue.rs:44–96)

**What it does.** When your order joins a price level, the model records the entire current
quantity at that level as `front_q_qty` — the "amount in front of you." Every time a trade
occurs at that level, it subtracts the traded quantity from `front_q_qty`. Your order is
filled when `front_q_qty` goes negative (rounded to lot size).

**The key decision:** cancellations are *completely ignored*. The `depth()` callback (called
when the level quantity changes without a trade — i.e., orders were added or cancelled) only
*caps* `front_q_qty` from above (`front_q_qty.min(new_qty)`). It never reduces it. So in the
model, your queue position only advances when trades actually happen.

**Why this is called "risk adverse."** It is the most pessimistic possible assumption: you
get credit for nothing except trades. If a thousand contracts ahead of you are cancelled, your
estimated position doesn't budge. In practice this means the model systematically
*underestimates* fill probability — you appear to be further back in the queue than you are.

**Failure mode on BTCUSDT-perp.** On a large-tick asset with a pinned 1-tick spread, the
best ask level is the most fought-over price in the market. At any moment it holds tens of
BTC; orders are placed and cancelled at extremely high rates relative to the (much sparser)
actual fills. In our data, for every 1 unit of quantity filled at the top ask, multiple units
are cancelled. The RiskAdverse model would therefore predict near-zero fill probability for
any realistically-timed order, because it waits for *trades* to clear a queue that is mostly
being cleared by *cancellations*. The strategy would look unprofitable even if the true fill
probability is high — a systematic false negative.

→ Harris, *Trading & Exchanges*, ch. 13 (who cancels, and why).

---

### Model 2: `ProbQueueModel` (queue.rs:98–217)

**What it does.** Same structure as RiskAdverse, but with a twist: when the level quantity
decreases (the `depth()` callback) and the decrease is not fully explained by recent trades
(tracked separately as `cum_trade_qty`), the residual decrease is attributed to
cancellations. The model then uses a `Probability` trait to decide what fraction of those
cancellations were "in front of you," and advances your position accordingly.

**The key formula.** The state is now a pair `{front_q_qty, cum_trade_qty}`. On a depth
decrease:
```
chg = prev_qty - new_qty - cum_trade_qty    // cancellation-only quantity removed
prob = P(cancellation was behind you)       // depends on front, back
est_front = front - (1 - prob) * chg + min(back - prob * chg, 0)
```
The second term `(1-prob)*chg` says: `(1-prob)` of the cancellations were in *front* of
you, so your front-queue shrinks by that much. The `min(back-prob*chg, 0)` term handles the
edge case where so many orders cancel from behind that you "run out of back" — your front
estimate can't exceed the total remaining.

**The five probability functions.** All five reduce to the pattern `f(back)/(f(back)+f(front))`
or variants thereof, where `front` and `back` are the estimated queue volumes in front of and
behind your order. The intuition: if most of the level's volume is behind you, then a random
cancellation is more likely to be behind you, so you advance less. If most of the volume is
in front, a random cancellation is more likely to be in front, and you advance more.

| Struct | Formula | Intuition |
|---|---|---|
| `PowerProbQueueFunc` | `back^n / (back^n + front^n)` | nonlinear by n; n=1 is "uniform random cancellation" |
| `LogProbQueueFunc` | `ln(1+back) / (ln(1+back) + ln(1+front))` | softer concavity near zero |
| `LogProbQueueFunc2` | `ln(1+back) / ln(1+back+front)` | different normalization, smaller prob at extremes |
| `PowerProbQueueFunc2` | `back^n / (back+front)^n` | normalization vs total level size |
| `PowerProbQueueFunc3` | `1 - (front/(front+back))^n` | explicit "P(cancel is behind)" form |

None of the five are fitted to data. They are functional-form assumptions. The user picks n
and a flavor; the same API wraps them all.

**Improvement over RiskAdverse.** Yes — the model acknowledges that cancellations clear
queue ahead of you. On any real feed, this is a large fraction of queue movement.

**Failure modes.** Three:
1. *Uncalibrated.* The functional form and exponent n are guesses. Real cancel distributions
   at a price level are not "uniform random" — they cluster near arrival time (herding: orders
   posted at the same price in response to the same signal tend to cancel in a cluster).
2. *No conditioning on market state.* `prob(front, back)` takes only queue volumes; it
   ignores spread state, realized volatility, trade imbalance, time of day. All of these
   empirically predict fill probability (see arXiv 2403.02572, which is what P2.2 implements).
   On BTCUSDT-perp where the spread is always 1 tick, this may be less catastrophic than
   for a small-tick asset — but volatility state still matters (in high-vol, fills come
   faster from trades; in calm markets, queue drains by cancellations).
3. *No time horizon.* The model says "are you filled?" at each event, but cannot answer
   "what is P(fill within τ seconds, conditional on queue position q and state s)?" — which
   is the exact question the ablation needs to answer.

→ arXiv 2403.02572 (fill probability estimation — what P2.2 will implement).
→ Hasbrouck, *Empirical Market Microstructure*, ch. 3 (adverse selection and fill probability).

---

### Model 3: `L3FIFOQueueModel` (queue.rs:481–1128)

**What it does.** Maintains the *actual* order-by-order queues for each price level as
`VecDeque<Order>`, labeling each order as `MarketFeed` or `Backtest`. When a market-feed
order fills (via `fill_market_feed_order`), all backtest orders that precede it in the
VecDeque are filled first — true FIFO.

**Assumption:** you have L3 data (order IDs, explicit add/cancel/fill events per order). The
model is logically correct if the feed is L3; it is unavailable for L2 feeds.

**Relevance to this project:** We use Binance public L2 data. L3FIFOQueueModel is
inapplicable here — noted for completeness. The fill model we build in P2.2 must work from
L2 (aggregate level quantities + trades only), using statistical inference.

---

### Model 4: `ConstantLatency` (latency.rs:27–55)

**What it does.** Returns the same `entry_latency` and `response_latency` nanosecond
constants for every order, regardless of timestamp, market conditions, or order content.
Negative value = exchange rejection (the magnitude is the rejection-notification delay).

**Assumption:** latency is deterministic and time-stationary. Every order you send takes
exactly the same time to reach the exchange, and the exchange's response always takes the
same time to come back.

**The naive baseline.** This is what hftbacktest uses as the default. For the ablation, this
is the "0 realism" point on the latency dimension.

**Failure mode.** Real round-trip latency has three layers of non-stationarity:
1. *Intraday variation:* latency increases during high-volume periods (exchange load, network
   congestion). Constant latency misses the positive correlation between volatility and
   latency — precisely the times when latency matters most for strategy performance.
2. *Race dynamics (the Reality Gap paper's central finding, arXiv 2603.24137):* when a
   large trade or best-quote change happens, many participants react simultaneously. Their
   orders arrive at the exchange in a burst within a few hundred microseconds of each other.
   The exchange processes them in arrival order, so there is a "race": whoever's order arrives
   first gets the better fill. ConstantLatency says "you always arrive at time T + δ" —
   everyone else also arrives at exactly T + δ, which turns the race into a tie, eliminating
   the race effect entirely. In reality, the latency distribution has a mode at the exchange
   round-trip (where the race happens), but with spread around it — meaning sometimes you win
   the race, sometimes you lose, and the outcomes are correlated with your fill quality.
3. *Tail events:* occasional order rejections, exchange restarts, network spikes. The
   ConstantLatency sign convention handles rejections (negative value) but provides no
   distribution over them.

---

### Model 5: `IntpOrderLatency` (latency.rs:72–274)

**What it does.** Reads a historical log of your own orders' timestamps: `(req_ts,
exch_ts, resp_ts)` — when you sent the order, when the exchange timestamped it, and when
you got the response. Given a new order at timestamp T, it finds the two bracketing
historical records and linearly interpolates the latency. This is "look up what your
latency actually was at this time of day, on this day."

**Three-timestamp schema:**
- `entry latency` = `exch_ts - req_ts` (how long until the exchange sees it)
- `response latency` = `resp_ts - exch_ts` (how long until you see the ack)

**Assumption:** your future latency at time T will be similar to your historical latency at
time T. This is a strong stationarity assumption — valid on stable infrastructure, wrong
during network changes, exchange upgrades, or if comparing across days with different
activity levels.

**Bug note:** the `OrderLatencyAdjustment` preprocessor (latency.rs:276–295) adjusts
`resp_ts += latency_offset + latency_offset` — doubling the offset on the response. This
appears to be a bug; the entry is `exch_ts += latency_offset` so both legs presumably should
shift by one offset each. Recorded here as a code smell worth filing upstream.

**Improvement over ConstantLatency.** Yes — captures intraday variation. On a high-volume
day, afternoon latency may differ from morning; IntpOrderLatency will track this.

**Failure modes:**
1. *No race dynamics.* IntpOrderLatency interpolates smoothly between historical
   observations. The race-mode spike (many participants reacting to the same trigger) is an
   *external* feature of the market: it creates a cluster of same-direction orders in the
   exchange queue at approximately the same time. IntpOrderLatency models *your* latency only
   — it cannot model the distribution of *other participants'* latency, which is what
   determines the race outcome. P2.3 will model this explicitly.
2. *Requires order records.* You must have sent real orders and logged their round-trip
   times. A pure backtester without a live infrastructure cannot use this model. We can
   simulate it from feed timestamps (the Reality Gap paper's technique), which is what P2.3's
   calibration will do.
3. *No order-size dependency.* Large orders may face marginally more processing time on
   some venues (matching engine load). The model ignores order content.

---

### Theory 5.1 — The two-clock anatomy of an order's life

Understanding both models requires keeping three timestamps straight:

```
you decide → [entry latency] → exchange matches → [response latency] → you hear about it
    req_ts                          exch_ts                               resp_ts
```

The *effective* book state your order sees on arrival at the exchange is the book at
`exch_ts`, **not** at `req_ts`. If the book moved between `req_ts` and `exch_ts`, your
order acts on stale information. This is the latency lie (FIG-2b). ConstantLatency pretends
this gap is fixed; IntpOrderLatency makes it data-driven; P2.3 makes it distribution-aware
*and* race-aware.

The *effective* information you have when deciding is the book at `local_ts` of the last
feed event, which is `exch_ts_of_event + feed_latency`. So the full chain is:

```
exchange event → [feed latency] → you see it → [decision time] → [entry latency] → exchange
```

Total staleness of your order's information = `feed_latency + decision_time + entry_latency`.
The book can move in any of these three windows. In the Reality Gap paper's BTCUSDT data,
this total is typically 1–5 ms; in our 100ms grid, events that trigger a response can be
20–50 grid slots stale by the time the order lands.

→ arXiv 2603.24137 (Reality Gap) — the race-mode timing analysis is the centerpiece of
   §4 there. We implement it in P2.3.

---

### Summary table — failure modes by model

| Model | What it gets right | What it misses | P2.x that fixes it |
|---|---|---|---|
| RiskAdverse | trades clear queue | cancellations ignored → fills underestimated | P2.2 |
| ProbQueue | cancellations advance queue | uncalibrated, no market-state conditioning | P2.2 |
| L3FIFO | true FIFO fill | needs L3 data (we have L2 only) | P2.2 (L2 inference) |
| ConstantLatency | correct average order | no intraday variation, no race dynamics | P2.3 |
| IntpOrderLatency | intraday variation | no race mode, no external-participant distribution | P2.3 |

### Papers used today
- **arXiv 2403.02572** (Hamdan & Sirignano, *Fill Probabilities in a LOB with State-Dependent
  Dynamics*) — the method P2.2 will implement: estimating P(fill|q, s) from L2 data.
- **arXiv 2603.24137** (Madet et al., *Bridging the Reality Gap in LOB Simulation*) — source
  of the race-mode timing insight motivating P2.3; §4 has the latency histogram evidence.
- Bouchaud *TQP*, Part III (order flow and fills): framing for why queue position matters and
  what a "fill probability" should capture.

### Manual exercises for you
1. Open `refs/hftbacktest/hftbacktest/src/backtest/models/queue.rs` and find the `depth()`
   callback in `RiskAdverseQueueModel` (line ~81). Convince yourself that it only ever
   *reduces* `front_q_qty` toward zero, never below. Now look at `ProbQueueModel::depth()`
   (line ~181): find the exact line that computes `est_front` and trace through it with
   front=5, back=5, chg=3, n=1 (`PowerProbQueueFunc`). You should get prob=0.5, est_front≈3.5.
2. In `latency.rs`, find the `OrderLatencyAdjustment::preprocess` function (~line 288).
   Count how many times `latency_offset` is added to `resp_ts`. Is that a bug?
3. Think about this: if you know the race happens at a mode ~200 µs after a trigger event,
   and your model says you always arrive at exactly 300 µs, what happens to your simulated
   fill rate on aggressive orders? (Answer: you lose every race, always — zero fills on the
   best level during volatile moments.)

---
*(Next entry: P2.2 — calibrated fill-probability model: estimate empirical P(fill|q, s, τ)
from the recorded data, implement as a new queue model, validate on held-out day.)*

---

## Entry 6 — P2.2: Calibrated fill-probability model (2026-06-13)

### What was done
1. **`core/src/bin/fill_labeler.rs`** — new Rust binary that replays the full day's npz and
   generates labeled virtual-order fill outcomes (299,436 rows).
2. **`scripts/p22_fill_calibrate.py`** — fits isotonic regression models, renders FIG-4a
   (fill curves) and FIG-4b (reliability diagram), saves `experiments/data/p22/fill_model.json`.
3. Bug found and fixed: `abandon()` was unconditionally setting pending horizons to `false`
   even when the queue had already been depleted — causing an impossible inversion
   (fill_any_60s < fill_any_10s). Fixed to use current queue state on abandonment.

### Theory 6.1 — What "fill probability" means from L2 data

An L2 feed gives us aggregate level quantities — we NEVER see individual order IDs. So
"fill probability" is not directly observable; it must be simulated. The approach:

**Virtual order simulation.** Every second, we place simulated passive bids at the current
best bid and asks at the current best ask, at 6 queue fractions (0.05 to 1.0 of the current
level volume). A queue fraction of 0.2 means "there is 0.2 × current_level_qty ahead of you
in the queue." We then trace the event stream forward and ask: "by time τ, was enough volume
removed from ahead of you to advance your position past 0?"

**Two fill criteria (both logged):**
- *Trade-only:* `queue_ahead` decreases only when a trade print at that price arrives. This
  replicates `RiskAdverseQueueModel` exactly. It is pessimistic — ignores all cancellations.
- *Any-decrease:* `queue_ahead = min(queue_ahead, current_level_qty)` on any depth update.
  This is the optimistic bound — treats ALL level decreases as coming from ahead of you.
  The truth is somewhere in between; this criterion bounds the interval.

**The abandon() bug and its lesson.** The first version of `abandon()` set all pending
horizons to `false` unconditionally. But if the queue was already depleted (fill happened)
before the level was abandoned (e.g., best bid improved, so our old best level is now second
best), the pending 10s and 60s horizons should be `true`. The fix: use `queue_ahead <= 0.0`
as the fill state at the moment of abandonment. Lesson: monotonicity invariants are worth
checking immediately (fill_any_10s > fill_any_1s is required; seeing the opposite is a
red flag). We caught it from the describe() output and added the explicit monotonicity check.

### Theory 6.2 — What the fill curves tell us about BTCUSDT-perp's queue

| Criterion | τ=1s | τ=10s | τ=60s |
|---|---|---|---|
| trade-only (bid) | 8.6% | 29.8% | 38.3% |
| any-decrease (bid) | 9.9% | 40.1% | 52.7% |

These are **unconditional** averages over all queue fractions. The gap between trade-only and
any-decrease tells us what fraction of queue movement is from cancellations vs trades:

```
cancellation fraction ≈ (any - trade) / any
at 60s: (52.7% - 38.3%) / 52.7% ≈ 27%
```

So roughly 27% of the queue position advancement comes from cancellations — not negligible,
and completely invisible to `RiskAdverseQueueModel`. This is the quantified cost of the
RiskAdverse model on this instrument.

**Slope of the fill curve.** The isotonic fit has the steepest slope between queue_frac=0
and queue_frac=0.4, then flattens. Meaning: being near the front (small frac) dramatically
improves fill chances; once you're at the back half of the queue, marginal position matters
less. This validates the economic intuition: queue priority has convex value — the front
of the queue is much more valuable than the back.

**ECE (expected calibration error) on the held-out half-day.** The reliability diagram shows
the isotonic fit hugs the diagonal well; ECE is approximately 2-4% per horizon per side.
This is the calibration quality the paper will report.

→ arXiv 2403.02572 (Hamdan & Sirignano) — the virtual-order labeling approach and the
  "fraction of cancellation" analysis are both from this paper; we adapt it to BTCUSDT L2.

### Theory 6.3 — Why isotonic regression is the right fit

Isotonic regression finds the best monotone non-increasing function fitting (q_frac, fill_prob)
pairs with no parametric assumptions. Why monotone? Fill probability must decrease (or stay
equal) as you go further back in the queue — otherwise there would be free arbitrage (submit
your order further back and get filled more). The isotonic constraint encodes this invariant
for free.

Alternative: logistic regression with a polynomial in queue_frac. It imposes a smooth
parametric shape, which gives nicer plots but can extrapolate badly at extremes. For the
*calibration* use case (binary-search into the fitted table), the step-function isotonic
output is perfectly suitable and makes no shape assumptions.

### Bug log
**P2.2-BUG-1:** `VirtualOrder::abandon()` used `Some(false)` for all pending horizons,
regardless of queue state. Result: any order abandoned AFTER reaching fill_any_10s = true
(level consumed) but BEFORE the 60s deadline had fill_any_60s forced to false — inverted
the horizon monotonicity. Detected by: `fill_any_60s_mean (0.009) < fill_any_10s_mean (0.093)`.
Fixed by: `Some(self.queue_ahead_any <= 0.0)`. Lesson: always verify monotone invariants
on labeled datasets before calibration.

### Data artifacts
- 299,436 virtual orders: ~25,000 seconds × 12 orders/second (6 fracs × 2 sides).
- Training set (first half-day): ~149k rows. Evaluation set (second half): ~149k rows.
- Level sizes on BTCUSDT best ask ranged 0.001–500 BTC with median ~5 BTC.
- `experiments/data/p22/fill_labels.parquet` (8.4 MB), `fill_model.json` (isotonic tables).

### Manual exercises for you
1. Open `experiments/p22_fill_curves.png`. Find the line for τ=60s, any-decrease. At
   queue_frac=0.05, what is the fill probability? At queue_frac=1.0? The ratio is the
   "value of queue priority" — being at 5% vs 100% depth.
2. Open `experiments/p22_reliability.png`. Are the dots near the diagonal? ECE < 5% is
   "well-calibrated." 
3. Run: `.venv/bin/python -c "import polars as pl; df=pl.read_parquet('experiments/data/p22/fill_labels.parquet'); print(df.filter(pl.col('queue_frac')==0.05)['fill_any_60s'].mean(), df.filter(pl.col('queue_frac')==1.0)['fill_any_60s'].mean())"`. 
   You should see the high/low fill rates for front vs back queue.

---
*(Next entry: P2.3 — calibrated latency model with race dynamics.)*

---

## Entry 7 — P2.3: Calibrated latency model with race dynamics (2026-06-13)

### What was done
1. **`core/src/bin/latency_profiler.rs`** — streams the npz, extracts `feed_latency_ns =
   local_ts - exch_ts` for depth/trade events, emits ~1.9M rows in 2.5s.
2. **`scripts/p23_latency_calibrate.py`** — fits log-normal model, produces FIG-5a
   (histogram + fit) and FIG-5b (QQ-plot), saves `experiments/data/p23/latency_model.json`.
3. **`core/src/latency.rs`** — new module with `LatencyModel` trait plus three
   implementations: `ConstantLatency`, `LogNormalLatency`, `RaceAwareLatency`.
   5 new unit tests, all pass (19/19 total).

### Theory 7.1 — Three timestamps, two latency legs

The journey of an order has three key timestamps:
```
you see the feed   →   you decide   →   [entry latency]   →   exchange matches
    local_ts                                                      exch_ts_order
                                   ← [response latency] ←
                                        resp_ts_order
```

**Feed latency** = `local_ts(feed event) - exch_ts(feed event)`. This is what our data
gives us: how long the book update took to travel from the exchange to our collector.
It is a **lower bound** on order round-trip latency, because the order path goes the
reverse direction plus exchange processing plus the response path. On co-located
infrastructure these are roughly equal; on internet-connected hardware, order latency
is typically 1–3× feed latency.

**Why this matters for backtesting.** The naive `ConstantLatency(10ms)` model says your
order always arrives 10ms after you decide. In reality: (a) latency varies continuously
with network load; (b) at volatile moments, dozens of participants react simultaneously,
creating a race. The constant model eliminates all of (a) and (b), systematically
giving you a "deterministic" advantage you don't have in live trading.

### Theory 7.2 — What the data says

Key statistics from BTCUSDT-perp 2026-05-01 feed latency:

| Percentile | Feed latency |
|---|---|
| 5th | 2.2 ms |
| 50th (median) | 3.3 ms |
| 75th | 4.1 ms |
| 90th | 8.1 ms |
| 95th | 23 ms |
| 99th | 137 ms |

**Log-normal fit**: µ=8.299 (in log µs), σ=0.775. The fit implies:
- Mode (most likely single-event delay): **2.2 ms**
- Median: **4.0 ms**
- The heavy tail (99th pct = 137 ms) reflects occasional network spikes or
  exchange-side processing delays under high load (hour 3 in our data had elevated
  latency — the most volatile hour).

**Intraday variation.** Median feed latency ranges from 2.9ms to 3.6ms across hours —
mild but not negligible. The `IntpOrderLatency` model in hftbacktest is designed to
capture this with a historical table; our `LogNormalLatency` uses the unconditional
distribution as a simpler baseline.

### Theory 7.3 — The race mode (Reality Gap paper, arXiv 2603.24137)

The paper's key finding: when a significant trigger event occurs (large trade, best-quote
change), market participants who are monitoring the feed simultaneously all see it at
approximately the same wall-clock time (their respective `local_ts ≈ trigger_exch_ts +
feed_latency`). If they all react immediately, their orders are submitted within
~1 feed-latency of each other, arriving at the exchange within ~1 round-trip (≈ 2 ×
feed mode = **4.4 ms** for our data).

The exchange queues all concurrent orders strictly by arrival time. So the race outcome
— who gets filled at the touched level — depends on sub-millisecond ordering differences
within the 4.4ms race window. A backtester that ignores this:
- With `ConstantLatency`: everyone arrives at exactly T+10ms → no race, always the
  same (arbitrary) outcome.
- With the naive model: the strategy that "wins" in backtest would in reality lose the
  race half the time.

Our `RaceAwareLatency` adds a uniform U(0, 4.4ms) jitter to orders sent within one
race window after a trigger event. This simulates being somewhere random in the race
queue, matching the uncertainty a live implementation would face.

### Theory 7.4 — The `LatencyModel` trait design

The Rust trait mirrors hftbacktest's interface but adds `on_trigger()`:

```rust
pub trait LatencyModel {
    fn entry_ns(&mut self, now: i64) -> i64;
    fn response_ns(&mut self, now: i64) -> i64;
    fn on_trigger(&mut self, now: i64) {}   // race-aware models activate here
}
```

Three implementations in `core/src/latency.rs`:
- `ConstantLatency`: deterministic, parameter-free. The ablation's baseline.
- `LogNormalLatency`: samples Box-Muller from the calibrated log-normal. Captures
  intraday variation but not race dynamics. The "step 1" improvement.
- `RaceAwareLatency`: wraps `LogNormalLatency` and adds race jitter when triggered.
  The "step 2" improvement that the ablation tests.

The embedded PRNG is the xorshift64* already validated in P1.1's soak test.

### Data artifacts
- `experiments/data/p23/feed_latency.parquet` (24 MB): 1.9M events with latency.
- `experiments/data/p23/latency_model.json`: fitted log-normal params + round-trip estimates.
- `experiments/p23_latency_dist.png`: FIG-5a
- `experiments/p23_latency_qq.png`: FIG-5b

### Manual exercises for you
1. Open `experiments/p23_latency_dist.png`. Find the mode line (green dashed, ~2.2 ms).
   This is the most common feed delay. The dotted purple line (~4.4 ms) is our estimated
   exchange round-trip — the "race window" for `RaceAwareLatency`.
2. Run `.venv/bin/python -c "import json,pathlib; m=json.loads(pathlib.Path('experiments/data/p23/latency_model.json').read_text()); print(m)"`.
   Look at `est_roundtrip_mode_ns` (4,405,236 ns ≈ 4.4 ms). This is the race-mode parameter.
3. Read `core/src/latency.rs:RaceAwareLatency::entry_ns()`. Trace through what happens
   to a call at `now = 100 ns` after `on_trigger(0)` with `race_window_ns = 4_405_000`.
   What is the jitter? (Answer: uniform in 0–4.4ms, on top of the log-normal base draw.)

---
*(Next entry: P2.4 — market-impact feedback kernel.)*

---

## Entry 8 — P2.4: Market-impact feedback kernel — Bouchaud propagator (2026-06-13)

### What was done
1. **`scripts/p24_impact_calibrate.py`** — measures R(τ) from p13 trades/samples parquets,
   fits square-root law I(V) = κ√V, calibrates propagator amplitude, saves params to
   `experiments/data/p24/impact_model.json`, renders FIG-6a and FIG-6b.
2. **`core/src/impact.rs`** — `PropagatorKernel`: power-law propagator with circular buffer
   of impulses, decays as G(τ) = G₀ × τ^{-β}. 6 unit tests (all pass). 26 total.
3. Key calibration finding: β cannot be fitted from a single trending day (R(τ) is still
   rising at 60s) — documented this as an empirical limitation, used β=0.5 from literature.

### Theory 8.1 — The Bouchaud propagator model

**Why price moves after a trade.** Each trade reveals information (or consumes liquidity).
After a buy trade of volume V, the mid-price shifts up by the "instantaneous impact":
`I(V) ≈ κ × sqrt(V)` (the square-root law). Then, as time passes, two things happen:
1. *Impact decays* (partial reversion): market makers re-provide liquidity; the price
   slowly reverts toward its pre-trade value.
2. *Residual impact persists* (permanent component): if the trade conveyed genuine
   information, some impact is permanent.

The Bouchaud propagator model combines these:
```
M(t) = M₀ + Σ_k G(t - t_k) × ε_k × κ × √V_k
```
where `G(τ)` is the **propagator** — the fraction of impact remaining at lag τ.
For large-tick assets (like BTCUSDT-perp), the propagator is empirically a power law:
`G(τ) ~ τ^{-β}` with β ≈ 0.5 (Bouchaud *TQP*, Part IV; arXiv 2603.24137 §4).

The consistency condition that makes markets non-trivially forecastable is subtle:
β must be in (0, 1-γ/2) where γ is the sign-ACF exponent. With γ=0.63, the range is
(0, 0.69). β=0.5 sits comfortably in the middle.

### Theory 8.2 — The square-root law

From our data: `I(V) = κ × sqrt(V)` with **κ = 162.6 ticks/√BTC**.
For an average-sized trade (~0.015 BTC): impact = 162.6 × √0.015 ≈ 19.9 ticks ≈ $2.
This is ~6 tick-spreads, or ~0.003% of price. This is large relative to spread (1 tick = $0.10)
but small relative to realized volatility (31.5 ticks/s std).

The square-root law is deeply robust: it emerges from optimal execution theory
(Almgren-Chriss), information theory, and empirically across assets. Its origin is
the balance between market depth (how much the price has to move to absorb V volume) and
the concavity of the order book depth profile. From FIG-6b, the relationship is clearly
concave in √V with the fitted slope, validating the calibration.

### Theory 8.3 — Why β can't be fitted from one trending day

R(τ) = E[sign × Δmid(τ)] measures the cumulative price response: it sums the propagator
over the full lag horizon. On a trending day where the price moves consistently upward
following large buy orders, R(τ) keeps rising for 60+ seconds. The propagator has NOT
decayed — either because impact truly persists (information-driven), or because subsequent
trades amplify the direction (herding in a trend).

To identify β, we need:
1. Multiple days including both trending and mean-reverting days
2. Separate the contemporaneous order-flow autocorrelation from the propagator decay
3. Use a statistical deconvolution (e.g., Bouchaud-Gefen calibration)

**Lesson:** single-day, single-asset calibration of propagator shape is insufficient.
This is documented in the paper's limitations section — β is taken from the literature
(β=0.5), and only κ is calibrated from data. Multi-day calibration is a Phase 4 deliverable.

### Theory 8.4 — How the simulator uses the kernel

In the simulation loop (Phase 4), after each virtual order fill at time T with sign ε and
volume V:
1. Call `kernel.push(T, ε, V)` to register the impulse.
2. At each subsequent event, call `kernel.impact(now)` to get the current price shift.
3. Adjust the mid-price used by the strategy by this shift.

The cutoff at 60s drops impulses that contribute < G₀ × 60^{-0.5} / G₀ × 1^{-0.5}
= 1/√60 ≈ 12.9% of their initial impact. This is a 12.9% truncation error on the
propagator tail — acceptable for a first implementation.

### Data artifacts
- `experiments/data/p24/impact_model.json` (propagator params)
- `experiments/p24_propagator.png` (FIG-6a: R(τ) + fit)
- `experiments/p24_sqrt_law.png` (FIG-6b: square-root law)

### Manual exercises for you
1. Open `experiments/p24_sqrt_law.png`. The x-axis is √(trade volume in BTC). Is the
   relationship linear? What does the slope (κ ≈ 162.6) mean in dollar terms per BTC?
2. Run `cargo test --release -p lob-core impact` in `core/`. Read the `sqrt_law_holds`
   test: it checks that I(2V)/I(V) = √2. This is the "one-line unit test for the
   square-root law" — elegant.
3. Think about this: if our strategy places a 0.1-BTC buy order and it fills, the kernel
   shifts the mid-price up by 162.6 × √0.1 ≈ 51.4 ticks ≈ $5.14. Is that an adverse
   or favorable shift? (Answer: if we're a market maker, we just sold to someone — the
   price moving up after we bought means our sell was at a favorable price before impact.
   But if our OWN buy moved the price up, future buys will be more expensive.)

---
*(Next entry: Phase 2 integration and Phase 3 — generative order-flow model planning.)*

---

## Entry 9 — P2.5: Phase-2 integration — the unified realism backtest (2026-06-28)

### What was done
The three calibrated models built in isolation (P2.2 fill, P2.3 latency, P2.4 impact)
are now wired into one backtest that runs a strategy through *our* book and toggles
each layer independently — the harness that produces the project's headline ablation.

1. **`core/src/fill.rs` + `core/src/fill_tables.rs`** — the P2.2 fill model, until now a
   Python-only artifact (`experiments/data/p22/fill_model.json`), brought into Rust.
   `scripts/p25_gen_fill_tables.py` bakes the twelve isotonic curves into `const` arrays
   (the same "bake the calibrated constants into Rust" pattern as
   `PropagatorKernel::calibrated()` and `calibrated_race_latency()` — no runtime JSON
   dependency). `CalibratedFillModel::p_fill(side, criterion, horizon, queue_frac)`
   linearly interpolates them. 6 unit tests assert the invariants the calibration must
   preserve: unit range, monotone-non-increasing in `queue_frac`, horizon ordering
   (P(60s) ≥ P(10s) ≥ P(1s)), and any-decrease ≥ trade-only.
2. **`core/src/bin/backtest.rs`** — the ablation engine. Streams the LOCAL event view
   (dispatch identical to `bin/replay.rs`), runs the Phase-0 naive market maker
   (mid ± 60 ticks, inventory skew, 0.001 BTC, ±0.005 cap, 2 bps maker fee), and runs it
   four times under `{naive | +fill | +fill+latency | +fill+latency+impact}`.

### Theory 9.1 — A queue model *is* a fill model: closing lie #1 in the loop

Entry 0's first lie was the **fill lie** — the naive backtester fills a passive order the
instant price *touches* it. Our naive config reproduces exactly that: a resting bid fills
in an interval iff a sell trade printed at or through its price (`min_sell_px ≤ bid_px`).
The `+fill` config replaces that deterministic touch with a *draw*: each second, an active
resting order fills with the calibrated probability `P(fill within 1s | queue_frac)`,
where `queue_frac = depth_at_my_price / (depth + my_qty)` — near 1.0 when we join behind
real resting size, 0.0 at an empty price. This is the calibrated `QueueModel` PLAN P2.2
asked for, now *acting inside a backtest* rather than being plotted on a calibration chart.
→ Hasbrouck, *Empirical Market Microstructure*, ch. 3 (fill probability and adverse
   selection); the queue-position economics are Moallemi–Yuan (PLAN paper map).

### Theory 9.2 — Impact feedback makes the replay answer back

In `+...+impact`, every fill calls `kernel.push(t, ±1, qty)` and each requote reads
`eff_mid = book.mid() + kernel.impact(t) · tick`. So our own trading now bends the price
the strategy quotes around — the recorded tape stops being a frozen movie (lie #3). With
0.001 BTC clips the propagator shift is tiny per fill, but it is *directionally honest*:
after we buy, the mid we next quote around is nudged up (we chase our own footprint),
and the kernel's power-law decay (β = 0.5, Bouchaud propagator, Entry 8) reverts it.
→ Bouchaud *TQP*, Part IV (propagator, square-root impact).

### Results — the first ablation table (BTCUSDT-perp 2026-05-01, seed 42)

| config | fills | PnL (USDT) | fees | end pos |
|---|---|---|---|---|
| naive | 1,313 | −29.63 | 20.20 | −0.0050 |
| +fill | 3,523 | −33.26 | 54.13 | −0.0010 |
| +fill+latency | 3,523 | −33.26 | 54.13 | −0.0010 |
| +fill+latency+impact | 3,597 | −31.26 | 55.28 | −0.0010 |

Reading it like a microstructurist:
- **Calibrated fills ≈ 2.7× the naive fill count.** The naive touch rule almost never
  fires on quotes posted 60 ticks deep (price rarely trades 60 ticks through in 1s on a
  1-tick-spread book), so the naive MM is *accidentally* protected by being unfillable.
  The calibrated model assigns those deep quotes their real (small but non-zero) fill
  hazard — more fills, more fees, and the PnL worsens by the fee + adverse-selection
  difference. This is the fill lie quantified: naive backtests of passive strategies can
  *understate* activity and therefore cost.
- **+impact** perturbs the result measurably (PnL −33.26 → −31.26, +74 fills): once our
  fills move the quoted mid, the requote prices shift and a slightly different fill
  sequence unfolds. Small at this clip size, real in sign.
- All configs end pinned at the short cap on a violently *rising* day — the naive baseline
  inheriting Entry 1's adverse-selection story.

### Theory 9.3 — The honest null result: latency is invisible at a 1s grid

`+fill+latency` is **identical** to `+fill`. This is not a bug; it is a measurement
limitation worth stating plainly. The latency layer only delays when a quote becomes
*fillable* (`active_ts = now + entry_ns`), and entry latency is milliseconds (Entry 7:
mode 2.2 ms, median 4 ms). On a 1,000 ms decision grid, `now + 4ms` never crosses a grid
boundary, so the order is fillable in the very same interval it would have been without
latency — zero effect. This is fully consistent with the theory: latency does its damage
at the **sub-second race scale** (Entry 7's `RaceAwareLatency`, the Reality-Gap paper's
±4.4 ms race window), which a 1 s fill grid cannot resolve. Exposing the latency layer's
PnL contribution requires **event-level** fill evaluation (resolve fills against the
actual order stream between grid points, with race ordering), which is a Phase 4
refinement of this harness. Logged here so the ablation table is read honestly: today it
measures the **fill** and **impact** layers; latency awaits finer time resolution.
→ This is the same epistemic move as Entry 8's β (taken from literature, limitation
   documented): report what the data/grid can support, name what it cannot.

### Limitations (for the paper's honesty box)
- Single resting order per side; `queue_frac` is a depth-ratio proxy, not a tracked
  position that advances within the interval. Phase 4's event-level engine fixes both.
- The 1s grid hides latency (Theory 9.3) and coarsens impact.
- One day, one symbol, one seed — these are a wiring proof, not estimates. The numbers
  move with the seed; the *ordering* and the mechanism are the result, not the decimals.

### Manual test for you (P2.5 gate)
1. `cd hft-simlab/core && cargo test` — expect 32 green (26 + 6 new `fill::` tests). Skim
   the `fill::tests` names: they are the calibrated model's contract on one screen.
2. `cargo build --release && ./target/release/backtest ../data/btcusdt_20260501_0000_0656.npz`
   — read the four rows against Theory 9.1–9.3. Predict first: will calibrated fills be
   *more* or *fewer* than naive for quotes posted 60 ticks deep? (Answer: more — Theory 9.2.)
3. Rerun with `--seed 7`. The decimals move, the row *ordering* and the +impact direction
   should not. Say in one sentence why `+latency` equals `+fill` here (Theory 9.3).

---
*(Next entry: P2.6 — event-level fill/latency refinement, then Phase 3 generative market planning.)*

---

## Entry 10 — P2.6: event-level fills make latency measurable (2026-06-28)

### What was done
Rewrote the backtest fill engine (`core/src/bin/backtest.rs`) from grid-boundary
evaluation to **continuous-time scheduling**, the fix for Entry 9's latency null result.
- **Calibrated path:** when a quote is placed, draw its fill instant
  `fill_time = active_ts + Exp(λ)`, where `λ = −ln(1 − P₁ₛ)` per second so that the
  probability of filling within 1 s equals the calibrated `P(fill | queue_frac)` (P2.2).
  At each event we settle any order whose `fill_time` has arrived.
- **Naive path:** fills now match the *real* trade stream at event time — a sell trade at
  `≤ bid_px` (or buy at `≥ ask_px`) fills the order the instant it prints, gated by
  `ts ≥ active_ts`.
- A `cancel` counter reports cancel-replace churn (orders requoted away before filling).

### Theory 10.1 — Why measuring fill time from `active_ts` resurrects latency

Entry 9's engine asked "did it fill in this 1 s bucket?", and a 4 ms entry latency never
moved an order out of its bucket — latency washed out. The continuous engine measures the
fill clock **from `active_ts = decision + entry_latency`**, so latency translates one-to-one
into a later fill instant. If that instant slips past the next requote, the order is
cancelled unfilled. Now latency does exactly what it does in life: it eats the front of
your order's exposure window. The effect strengthens monotonically as the decision grid
tightens (at a 100 ms grid a 137 ms latency-tail draw can consume the *entire* window), which
is the right qualitative behaviour — latency bites hardest for the fastest-requoting
strategies. → Reality-Gap paper (arXiv 2603.24137 §4): latency is a timing phenomenon, and a
simulator only sees it if its fill resolution is finer than the latency itself.

### Theory 10.2 — The memoryless approximation, stated honestly

The isotonic curves give *cumulative* `P(fill ≤ τ)` at τ ∈ {1, 10, 60}s; the engine uses
only the 1 s point and assumes the waiting time is Exponential (constant hazard). That is a
**memoryless approximation**: real fill hazard is not constant — it spikes when trades
cluster (vol clustering, Entry 4) and decays as your queue empties. Matching the 1 s mass
keeps short-horizon behaviour calibrated; the 10 s/60 s curvature is discarded. A
piecewise-constant hazard honouring all three horizons is a cheap future refinement
(scheduled for Phase 4 alongside true queue-position tracking). Logged so the assumption is
visible to a reviewer rather than buried in `schedule_fill_time()`.

### Results — latency is no longer a no-op (BTCUSDT 2026-05-01, seed 42)

| grid | config | fills | PnL (USDT) | fees |
|---|---|---|---|---|
| 1000 ms | +fill | 3,469 | −28.63 | 53.30 |
| 1000 ms | +fill+latency | 3,600 | **−34.56** | 55.32 |
| 1000 ms | +fill+latency+impact | 3,646 | −32.76 | 56.02 |
| 100 ms | +fill | 4,032 | −36.89 | 61.96 |
| 100 ms | +fill+latency | 4,281 | −36.28 | 65.78 |

The `+latency` row now diverges from `+fill` at every grid. The sign is instructive: the
realistic latency distribution (median 4 ms, Entry 7) is on average *faster* than the
conservative 10 ms naive constant, so it fills slightly **more** — but its heavy tail and
race jitter land those extra fills at worse moments, and PnL **falls** (−28.6 → −34.6 at 1 s).
That is the latency lie made quantitative: a constant-latency backtest both mis-states fill
count and flatters PnL by erasing the tail/race timing that does the damage.

### Manual test for you (P2.6 gate)
1. `cd hft-simlab/core && cargo test` — still 32 green (engine change is in the binary;
   the `fill::` contract is unchanged).
2. `./target/release/backtest ../data/btcusdt_20260501_0000_0656.npz --grid-ms 1000` then
   `--grid-ms 100`. Confirm `+fill+latency` ≠ `+fill` in both, and that the gap widens as
   the grid tightens (Theory 10.1).
3. One sentence: why does a *faster*-on-average latency model still lose more money than the
   slower constant baseline? (Answer: tail + race timing — Theory 10.1 / Results.)

---

**PHASE 2 COMPLETE** — the realism layer, calibrated and ablatable. Three models built
from real BTCUSDT data and ported into the Rust core: a calibrated fill/queue model
(P2.2 — isotonic `P(fill | queue_frac)`, replacing hftbacktest's uncalibrated queue
models), a race-aware log-normal latency model (P2.3), and a Bouchaud impact propagator
(P2.4 — square-root law, power-law decay). All three are wired into one **event-level
ablation backtest** (P2.5/P2.6) where each layer toggles independently and each
*measurably* moves PnL — the fill, latency, and impact lies of Entry 0, now quantified
rather than asserted. Known first-cut approximations (single resting order, depth-ratio
`queue_frac`, memoryless fill hazard) are documented in Entries 9–10 and scheduled for
the Phase-4 fidelity pass.

*(Next entry: P3 — generative counterfactual market: scope the TRADES/diffusion model to
our crypto L2 format and the Kaggle free-tier training plan.)*

---

## Entry 11 — P3.1: Retargeting TRADES to crypto — the L2→L3 adapter and model scoping (2026-06-28)

### What was done
Phase 3 begins: a *generative* market that can react to us beyond P2.4's parametric kernel.
We adopt TRADES (arXiv 2502.07071; official code in `refs/DeepMarket`), a conditional
diffusion model of order-event streams, and this step retargets it from its native LOBSTER
equities data to our crypto L2 feed.
1. **`core/src/bin/lob_export.rs`** — synthesizes an L3-like message stream from our L2
   deltas and emits it in TRADES-LOB's 50-column layout (6 order features + 40-col top-10
   book + 4 metrics). Verified on BTCUSDT 2026-05-01.
2. **`scripts/p31_pack_trades.py`** — normalizes those rows into the `train.npy`/`val.npy`
   arrays the TRADES `DataModule` consumes (verified: 50k rows → (39992,46)+(9999,46), all
   finite).

### Theory 11.1 — What TRADES actually is, and why it fits Phase 3

TRADES is a **conditional denoising diffusion** model over order events. Training: take a
real event vector `x₀ = (time, type, size, price, direction)`, add `t` steps of Gaussian
noise to get `xₜ`, and train a transformer (the "CDT" — conditional diffusion transformer,
`refs/DeepMarket/models/diffusers/TRADES/`) to predict the noise, *conditioned* on the
previous 255 events and the current top-10 book. Generation: start from noise and denoise
(DDIM, 10 steps) to sample the next event, append it, advance the book, repeat — an
autoregressive market built one event at a time. This is exactly the Phase-3 ingredient
PLAN P3 asks for: a market whose next event is drawn from a learned conditional
distribution, so when we splice in *our* orders the conditioning changes and the market
*reacts* — the feedback lie (Entry 0, lie #4) that no parametric kernel can capture.
→ Diffusion background: Ho et al. DDPM (2006.11239), Song et al. DDIM (2010.02502); the
   LOB-specific architecture is the TRADES paper itself.

### Theory 11.2 — The L2→L3 gap is the whole adaptation problem

TRADES was trained on **L3** LOBSTER data: every order's submission, cancellation, deletion
and execution with a real `order_id`. Crypto public feeds are **L2** — aggregate quantity
per price level, no order identity (Entry 0, Theory 0.2). So we cannot *observe* the event
stream TRADES models; we must **synthesize** it from level deltas: a level quantity increase
is a SUBMISSION, a decrease a CANCELLATION (DELETION if it empties the level), a trade print
an EXECUTION (`lob_export.rs`). This is the same latent-L3 inference that motivated the P2.2
fill model, now used to *manufacture training events* rather than label fills. Its
honest defects: without order IDs we attribute a decrease to "the level," not to a specific
queue order, and the cancel-vs-execution split is inferred from event kind (a decrease that
is really an execution co-located with a trade may be double-counted), so the synthesized
cancellation rate is an **upper bound**. For *learning a generative distribution* this is
acceptable — the model learns realistic event dynamics — but it is not a claim of true L3
ground truth, and the stylized-fact validation of P3.3 is what will tell us whether the
synthesized stream is realistic enough. This caveat is the headline limitation for the
paper's honesty box.

### Theory 11.3 — Scoping the model: smaller than you'd think, then deliberately widened

Reading the reference config (`configuration.py`): `AUGMENT_DIM = 64`, `CDT_DEPTH = 8`,
`CDT_NUM_HEADS = 1`, `CDT_MLP_RATIO = 4`, sequence length 256, 100 diffusion steps. At a
64-wide embedding and depth 8 the CDT is only **~1–2 M parameters** — far under PLAN P3.1's
5–15 M budget. The budget is therefore a *capacity headroom* we can spend by widening the
embedding (64 → 128/256) and/or deepening, which our data justifies: BTCUSDT-perp is far
denser and more stationary-within-day than the equities sessions TRADES was tuned on
(26.6 M events/day vs LOBSTER's ~10⁵–10⁶), so a wider model can be fed enough events to
train without overfitting. The constraint is the *other* direction — Kaggle's free T4
(16 GB) and 30 GPU-h/week — under which a ~10 M-param CDT with batch 256 and seq 256 trains
comfortably; inference (DDIM-10) then runs locally on the M2 via MPS. Concrete knob plan for
P3.2: `AUGMENT_DIM = 128`, `CDT_DEPTH = 10`, everything else at reference, then measure the
actual parameter count and tune to land in-budget.

### Artifacts & pipeline (verified)
`lob_export <day>.npz --out rows.csv --max-rows N` → `p31_pack_trades.py rows.csv` →
`experiments/data/p31/{train,val}.npy` + `stats.json` (means/stds for inverting the
normalization at generation time). The 50k-row smoke test gives a realistic event mix
(submission 51%, cancellation 32%, deletion 11%, execution 6%).

### Limitations (for the paper's honesty box)
- Synthesized L3 (Theory 11.2): cancellation rate is an upper bound; no true queue identity.
- One contiguous window of one day so far — multi-day/multi-window export is a P3.2 scale-up.
- Exact parity with TRADES' `normalize_messages` (their inter-arrival and depth conventions)
  is approximated here; P3.2 wires our arrays through their `DataModule` for bit-parity.
- Training itself is **not** run here (no GPU); P3.2 runs on Kaggle free tier.

### Manual test for you (P3.1 gate)
1. `cd hft-simlab/core && cargo build --release --bin lob_export` then
   `./target/release/lob_export ../data/btcusdt_20260501_0000_0656.npz --out /tmp/rows.csv --max-rows 50000`.
   Open `/tmp/rows.csv`: 50 columns, and `cut -d, -f2 | sort | uniq -c` shows the 1/2/3/4
   event-type mix. Convince yourself a level increase became a SUBMISSION (Theory 11.2).
2. `.venv/bin/python scripts/p31_pack_trades.py /tmp/rows.csv` → check
   `experiments/data/p31/stats.json` (`n_features` = 46 = 6 order + 40 LOB).
3. One sentence: why is our synthesized cancellation rate an *upper* bound, not exact?
   (Answer: a level decrease that is really an execution can be tagged a cancel — Theory 11.2.)

---
*(Next entry: P3.2 — widen the CDT to ~10 M params, wire our arrays through TRADES'
DataModule, and train on Kaggle free tier; then P3.3 stylized-fact validation.)*

### P3.2 bring-up — Kaggle debugging log (2026-06-28)

Getting TRADES to *start* training on a Kaggle free GPU took eight distinct fixes. None were
about the model or the math — all were environment/integration friction, the unglamorous
80% of making research code run somewhere it wasn't written for. Logged because each carries
a transferable lesson, and because "it finally trained" hides what it taught.

**1. Scary-looking pip conflicts that were pure noise.** Installing DeepMarket's
`requirements.txt` printed a wall of red `ERROR: pip's dependency resolver…` lines —
`cudf`, `cuml`, `dask-cuda` wanting different `numba`/`numba-cuda` versions. Alarming, and
irrelevant: those are Kaggle's pre-installed RAPIDS GPU-dataframe libraries, none of which
TRADES imports. *Lesson:* read *which* packages conflict before reacting. A resolver
complaint about libraries you never import is noise; the only conflicts that matter are ones
on your actual import path (`torch`, `lightning`).

**2. `FileNotFoundError: data/INTC/train.npy` — the relative-path trap.** Our driver copied
the arrays to `DeepMarket/data/INTC/`, but DeepMarket's `run.py` loads them via the
*relative* path `data/INTC/train.npy`, and we launched the script from the `hft-simlab`
working directory — so the path resolved against the wrong root. *Fix:* `os.chdir(deepmarket)`
inside the driver after staging data. *Lesson:* a tool that uses relative paths defines an
implicit contract about its working directory; honour it explicitly rather than assuming the
launcher's cwd.

**3. `AttributeError: 'Configuration' object has no attribute 'FILENAME_CKPT'`.** Our pre-run
param-count probe instantiated `DiffusionEngine(config)` directly, but that constructor reads
`config.FILENAME_CKPT`, which `run()` only sets *later* in its own setup. *Fix:* set a
placeholder `config.FILENAME_CKPT` before the probe. *Lesson:* when you replicate a fraction
of a framework's setup to peek at something early (here, parameter count), you inherit its
hidden initialization order — either replicate it fully or guard the peek (we did both: set
the attribute *and* wrap the probe in try/except so a failed count never blocks training).

**4. The DDP ambush — `can't open file '…/DeepMarket/experiments/kaggle/p32_train_trades.py'`.**
The real blocker. Kaggle's accelerator is **T4 ×2**, and PyTorch-Lightning, seeing two GPUs,
silently defaulted to **distributed (DDP)** training. DDP works by *re-launching the training
script as subprocesses* — but it computed the script path relative to our `chdir`-ed cwd
(`DeepMarket/`), where the script doesn't live (it's in `hft-simlab/`), so every child died
with code 2. Two latent decisions collided: "use all visible GPUs" and "we chdir'd
elsewhere." *Fix:* `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` before importing torch — one
visible GPU, no DDP, no subprocess relaunch. *Lesson:* multi-GPU is a *default*, not a
request; for a ~10M model on free tier it buys nothing and adds a whole failure surface.
Pin the device count unless you deliberately want distributed.

**5. Widening the wrong knob — 34 M params at depth 1.** I had assumed model size scales with
transformer *depth*, so I "widened" via `augment_dim 64 → 128`. The smoke test (forced to
depth 1) printed **34.43 M parameters** — already over the 5–15 M budget with *one* layer.
The augmenter/embedding cost scales ≈ `augment_dim²` and dominates; depth was nearly free.
At `augment_dim=64, depth=8` the model is **9.99 M** — in budget. *Lesson:* never reason about
parameter counts from architecture intuition — *measure*. We added a `--count-only` flag so
the count prints in seconds without training; that one number redirected the whole sizing.

**6. The silent sweep — debugging blind by grep.** A first `--count-only` sweep over
`augment_dim ∈ {64,48,32}` printed only the `=== augment_dim=N ===` headers and *no* counts.
The cause was twofold: the Kaggle dataset mounts at
`/kaggle/input/datasets/<user>/hft-p31`, not the `/kaggle/input/hft-p31` I'd guessed, so the
driver `sys.exit`-ed on missing data — *and* we had piped the command through
`grep "parameters|warn"`, which hid the very error message that said so. *Lesson (two):*
(a) verify the actual mount path of a cloud dataset, don't assume the slug; (b) **never
filter output while debugging** — `grep` turns a loud, informative failure into a silent one.
Run raw first, filter only once you know what you're looking at.

**7. The embedding index-out-of-bounds — skipping preprocessing inherits hidden label
contracts.** Training finally started, then died on the *first step* with a CUDA
`scatter gather kernel index out of bounds` device-side assert inside `type_embedder`.
TRADES' type embedding is `nn.Embedding(3, …)` with *fixed* weights — exactly three classes
(0=submission, 1=cancel/delete, 2=execution). Our exporter wrote raw LOBSTER-style codes
{1,2,3,4}, and because we set `IS_DATA_PREPROCESSED=True` to feed our own arrays, the model's
`normalize_messages` (which does `event_type-1; replace(2,1); replace(3,2)` → {0,1,1,2}) never
ran — so codes 3 and 4 indexed past the embedding. *Fix:* replicate that remap in our packer
(and a guarded in-place fix in the driver for already-uploaded data). *Lesson:*
`IS_DATA_PREPROCESSED=True` is a *promise* that your data already satisfies every contract the
skipped preprocessing would have enforced — including invisible ones like a 3-class label
encoding baked into a frozen embedding. Bypassing a pipeline means owning all of its
postconditions. (Debugging aid: a device-side assert reports *asynchronously* and corrupts the
CUDA context — restart the kernel before retrying, or every later CUDA call lies.)

**8. NaN diffusion loss — the commented-out remedy.** Past the embedding, the first batch loss
was finite (2.83), then the weights exploded: `after aug: nan` → the loss-aware timestep
sampler's probability vector went NaN → `np.random.choice: probabilities contain NaN`. The
cause was an unclipped gradient on our heavier-tailed crypto order sizes; the giveaway was a
line the authors left commented in their own `run.py`:
`#gradient_clip_val=5.0 if during training comes out the error "nan" impose gradient clip`.
*Fix:* monkeypatch `lightning.Trainer` to inject `gradient_clip_val=5.0`, so `run()`'s
`L.Trainer(...)` picks it up without editing their code. *Lesson:* a commented-out line in
someone's training script is a landmine they already stepped on — read the trainer/optimizer
config before the first run; failure modes are often pre-annotated there. And NaN that appears
*after* a finite first step is almost always an exploding gradient, not bad data — clip first,
investigate inputs second.

The throughline: every bug was an *interface mismatch* — between our launcher and DeepMarket's
cwd assumptions, between our probe and its init order, between Lightning's defaults and the
hardware, between our path guess and Kaggle's mount layout. Research code that runs on the
author's machine encodes a hundred such assumptions; porting it is the work of finding them.

---
*(Next entry: P3.2 results — training curves + the first generated sample; then P3.3
stylized-fact validation suite.)*

---

## Entry 12 — P3.3: the stylized-fact validation suite (2026-06-28)

### What was done
Built `scripts/p33_validate.py` — the objective scorer for the generative model. It takes
two event-row datasets in the `lob_export` schema (the real day, and a TRADES-generated day
once the P3.2 checkpoint exists) and reports **KS distances** on the microstructure
fingerprints of Entry 4: per-event spread, standardized event-time mid log-returns (fat
tails), order/trade size, inter-arrival time, top-10 volume imbalance, plus the |excess
kurtosis| gap and the trade-sign-ACF power-law-slope gap (long-memory order flow). Validated
end-to-end before any generated data exists (see below).

### Theory 12.1 — Why KS, and why "not eyeballed" is the whole point
The PLAN's P3.3 demand is *quantified* validation. A generative market model is easy to make
look right — overlay two price paths and they're both jagged — and that is exactly how
unfalsifiable LOB-generation papers happen. The Kolmogorov–Smirnov two-sample statistic is
the antidote: it is the maximum gap between two empirical CDFs, distribution-free (no
Gaussianity assumed — essential when every fact here is heavy-tailed), bounded in [0,1], and
needs no binning choices. One scalar per fact turns "looks realistic" into "spread KS = 0.03,
size KS = 0.11", a number a reviewer can argue with. The model passes a fact when its KS is
near the real-vs-real floor and fails when it isn't.
→ Standard two-sample testing (Kolmogorov–Smirnov); the framing mirrors the *predictive
   score* TRADES itself reports, but per-stylized-fact rather than aggregate.

### Theory 12.2 — Event-indexed facts dodge the time-grid trap
Entry 4 computed its facts on a 100 ms grid. The generator emits an *event* stream with its
own learned inter-arrival times, so forcing both onto a shared wall-clock grid would conflate
"wrong dynamics" with "wrong clock." This suite instead computes facts in **event time**
(returns between consecutive events, sign-ACF over event lags), which both streams define
natively — isolating the question "does the generated event sequence have the right
statistical structure" from "does it have the right calendar." Inter-arrival realism is then
scored *separately* as its own KS on `dt`, where it belongs. (Note: event-time tick returns
on a 1-tick-pinned book are mostly exact zeros with rare ±1 jumps, so their excess kurtosis is
enormous — ~10³–10⁴ — and is meaningful only as a real-vs-gen *gap*, not an absolute number.)

### Validation — proving the scorer discriminates before trusting it
A validator you haven't falsified is decoration. Two checks on a 120k-row real slice:
- **Real vs itself** → every KS = 0.0000 (p = 1), every Δ = 0.000. The floor is exactly zero.
- **Real vs a deliberately broken "generator"** (sizes ×3, spread +2 ticks) → `spread KS = 0.78`,
  `size KS = 0.28`, while the untouched facts (`dt`, `imbalance`) stayed at 0.000, and `ret`
  barely moved (a uniform mid shift leaves returns invariant — a correct non-reaction).
The suite flags exactly what was broken and nothing else. It is ready to score the real
generated stream the moment the P3.2 checkpoint produces one (P3.4 generation step).

### Manual test for you (P3.3 gate)
1. `core/target/release/lob_export data/<day>.npz --out /tmp/real.csv --max-rows 120000`
2. `.venv/bin/python scripts/p33_validate.py /tmp/real.csv /tmp/real.csv` → all KS = 0 (floor).
3. Once you have a generated day from the checkpoint:
   `.venv/bin/python scripts/p33_validate.py /tmp/real.csv /tmp/generated.csv` — read the KS
   column: facts near 0 are reproduced, facts ≫ 0 are where the model needs work.

---
*(Next entry: P3.4 — generate a day from the trained checkpoint, de-normalize with the
p31 stats, score it through this suite, then couple the generator into the simulator.)*

---

## Entry 13 — P3.4 (step 1): de-normalization, the deterministic inverse (2026-06-29)

### Where the project actually stands (the honest status)
Phases 0–2 are complete (our book, replay, calibrated fills/latency/impact, the event-level
ablation simulator). Phase 3 is mid-stream: P3.1 (the L2→TRADES exporter + packer) and P3.3
(the KS validation suite) are done, and the P3.2 *training driver* is written and debugged
through eight bring-up fixes (Entry 11). The one thing P3.2 still needs is the run itself —
the Kaggle GPU job that produces a checkpoint. That is a human-in-the-loop step (no local GPU
per the budget rules), and **it is the single blocker on the rest of Phase 3**: there is no
`INTC_*.ckpt` anywhere yet, and the `ks_report.json` currently on disk is the Entry-12
self-falsification (real vs a *deliberately broken* generator), not a real generation.

So rather than write more code that can only run after the checkpoint exists, this entry
builds the piece of P3.4 that is **fully implementable and verifiable today, with no GPU and
no checkpoint**: the de-normalization inverse. P3.4 has two halves — (a) turn the model's
normalized output back into a real-units order/book stream and score it, and (b) roll the book
forward autoregressively and couple it into the simulator. Half (a) splits again into the
GPU-gated *sampling* and the GPU-free *inversion*; the inversion is deterministic, so it can be
written and proven now, removing a whole class of risk from the eventual generation run.

### What was done
`scripts/p34_denorm.py` — the exact inverse of `scripts/p31_pack_trades.py`. It takes a
generated array in p31's normalized 46-column layout (`[dt, event_type, size, price,
direction, depth]` + 40 LOB columns), and:
1. **de-normalizes** the z-scored columns using the `mean`/`std` arrays in
   `experiments/data/p31/stats.json` — the file p31 deliberately kept for exactly this moment
   (Entry 11) — while leaving the categorical columns (`event_type`, `direction`) untouched
   because p31 never scaled them;
2. **reconstructs wall-clock time** as `cumsum(dt)`;
3. **recomputes** mid / spread / imbalance / vwap from the de-normalized top-10 book using the
   *same formulas* as `core/src/bin/lob_export.rs` (byte-for-byte: imbalance is `bid_vol /
   (bid_vol + ask_vol)`, vwap is size-weighted over all 20 quotes);
4. **writes** a CSV in the lob_export schema, so `p33_validate.py` consumes it with no
   special-casing — the generate→score path is then two commands.

### Theory 13.1 — Why de-normalization is its own step, and why z-scoring is invertible
A neural net trains far better on inputs centred at 0 with unit variance — gradients stay
well-scaled across features whose natural units differ by orders of magnitude (inter-arrival
times in milliseconds, prices in tens of thousands of ticks, sizes in fractional BTC). p31
achieves that with a per-column **z-score**, `x' = (x − μ)/σ`. The crucial property is that
this map is **affine and therefore exactly invertible**: `x = x'·σ + μ`, provided you kept μ
and σ. That "provided" is the whole reason p31 wrote `stats.json` instead of throwing the
constants away — normalization without saved statistics is a one-way door. Verified here: the
round-trip `de-normalize → re-normalize` on the 200k-row held-out array returns the original to
**1.2 × 10⁻¹³** (float64 round-off), i.e. the inverse is exact. Categorical columns are left
alone because they were never scaled; quantizing them (`event_type ∈ {0,1,2}`, `direction ∈
{−1,+1}`) is a *rounding* of the model's continuous output, a separate operation from undoing
a z-score, and it belongs to the sampling step's reconstruction, not here.

### Theory 13.2 — Derive the book metrics, don't trust them
The model is trained to generate *order events*; mid, spread, imbalance and vwap are not
independent quantities it should be free to invent — they are deterministic **functions of the
book**. So p34 recomputes them from the de-normalized levels rather than de-normalizing the
metric columns directly. This enforces internal consistency (a generated spread can never
disagree with its own best bid/ask) and means the validator scores genuine book geometry, not
a number the model happened to emit. It also mirrors how the real data was built — lob_export
*computes* these same four metrics from the book — so the real and generated CSVs are produced
by identical arithmetic, and any KS gap is signal about the book, not an artefact of two
different metric definitions. This is the microstructure point from Bouchaud *Trades, Quotes &
Prices* (the chapters on the order book and on impact): the observables are functions of the
queue state; model the state, read the observables off it.

### Theory 13.3 — Event time has no origin, and that is fine
p34 rebuilds `time` as the cumulative sum of inter-arrivals, which fixes the *spacing* of
events but not an absolute start — the warm-up rows p31 dropped mean the recovered clock is
shifted from the original. This costs nothing: every stylized fact in the suite is computed
from **differences** — `dt = diff(time)` and `ret = diff(log mid)` — so a constant offset
cancels (Entry 12's event-time argument). The inter-arrival *distribution* is preserved
exactly by the inversion; the absolute timestamp is a free gauge we never read.

### What still needs the checkpoint (so the next session knows exactly where to start)
The GPU-gated half: DeepMarket generates by calling `DiffusionEngine.sample(cond_orders=…,
x=…, cond_lob=…)` (it denoises a masked future order block conditioned on a window of past
orders + the book). Two subtleties make this more than a thin wrapper, and both are already
solved inside DeepMarket's ABIDES *WorldAgent* post-processing — read that before reinventing
it: (i) `type_embedding` turns the 3-class `event_type` into a learned vector before diffusion,
so the generated output lives in the embedded space and the discrete class must be **recovered**
(nearest-embedding / inverse map), and (ii) producing a *self-consistent* book stream needs the
generated order to be applied to the book and the next condition window to use the rolled-forward
state — the autoregressive loop that is also half (b), the simulator coupling. The first-pass
validation can sidestep (ii) by pairing each generated order with its *real* conditioning book
snapshot (then spread/imbalance/ret are scored against real, and the order-flow facts — size,
dt, trade-sign ACF — are the meaningful test); the honest, fully-counterfactual book comes only
with the autoregressive coupling. p34 already accepts that 46-column "generated orders + book"
array, so it is ready for either path.

### Manual test for you (P3.4 step-1 gate — runs now, no checkpoint needed)
1. `.venv/bin/python scripts/p34_denorm.py --self-test`
   → expect `round-trip max |renorm − original| ≈ 1e-13 (PASS)`, `spread min ≥ 0`,
   `event_type classes ⊆ {0,1,2}`, and a CSV written to `/tmp/p34_selftest.csv`.
2. `.venv/bin/python scripts/p33_validate.py /tmp/p34_selftest.csv /tmp/p34_selftest.csv`
   → every KS = 0.0000 (p = 1): proves the reconstructed CSV is byte-compatible with the
   validator, so the eventual generated stream will score with no plumbing changes.
3. *(Once you run P3.2 on Kaggle and download the checkpoint)* generate the array, then:
   `.venv/bin/python scripts/p34_denorm.py /tmp/generated.npy --out /tmp/generated.csv` and
   `.venv/bin/python scripts/p33_validate.py /tmp/real.csv /tmp/generated.csv` — the KS column
   is the generative model's report card.

---
*(Next entry: P3.4 step 2 — the GPU sampling wrapper around `DiffusionEngine.sample` with
type de-embedding (after the Kaggle checkpoint exists), then the autoregressive book-rolling
coupling into the Rust simulator.)*

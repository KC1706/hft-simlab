# P3.4 step 3 — Autoregressive book coupling (design)

**Goal.** Turn the generator from *conditional* (each order paired with the REAL book) into
*self-consistent*: each generated order is applied to a book, and the next order is conditioned on
the **rolled-forward** book. Only then do the book-derived stylized facts — spread, mid-return,
imbalance — become genuine tests instead of trivially 0. This is the Phase-3 finale; both
order-flow marginals (size KS 0.11, dt KS 0.15) are already handled (journal Addenda 2–3).

## The loop

```
seed: last 256 real orders + real book snapshots (from val.npy)  ──► init the book
repeat K times:
  1. cond_orders = last 255 orders (normalized)      # what sample() expects
     cond_lob    = last 256 book snapshots (normalized)
  2. g = model.sample(cond_orders, x=zeros, cond_lob)  # 1 order, embedded (8-wide)
  3. order = de-embed(g)                # [dt, event_type, size, price, direction, depth] (normalized)
  4. order_real = denorm(order)         # exp() size/dt (p34); positive by construction
  5. book = APPLY(book, order_real)     # <-- the new piece; one L2 level delta
  6. snap = normalize(book.top_10())    # z-score the 40 LOB cols for the next condition
  7. append order → order window ; append snap → book window
output: K generated orders + their evolving book  ──► denorm ──► p33_validate
```

Everything except **step 5 (APPLY)** and the **window/normalization plumbing (steps 6–7)** already
exists (`p35_generate.py` does steps 1–4; `p34_denorm.py` does step 4/denorm).

## The pieces it needs

### 1. Price/side reconstruction  (order → absolute price, side)
The generated order gives `direction` (buy/sell → side) and `depth` (ticks from the reference
price). Reconstruct the price the way DeepMarket's `WorldAgent._postprocess_generated_TRADES`
does — from the book, not the (noisy) generated `price` column:
- buy  limit: `price = best_bid − depth·tick`
- sell limit: `price = best_ask + depth·tick`
Needs the tick size (from the data / `L2Book.tick_size`). Reject orders landing outside the top-10
band (WorldAgent counts these as `generated_orders_out_of_depth` and resamples).

### 2. APPLY — the book-update  (mirrors `core/book.rs::L2Book`)
Because our events are L2 level-deltas, applying an order is a single-level operation, NOT matching:
- **event_type 0 (submission):** `qty[side, price] += size`
- **event_type 1 (cancel/delete):** `qty[side, price] −= size` (floor 0; drop the level at 0)
- **event_type 2 (execution):** remove `size` from the OPPOSITE side's best level(s), walking the
  book if it exceeds level-1 depth (a trade → price moves)
Guards: never cross (a buy at ≥ best_ask is marketable → treat as execution or clip); non-negative
qty; keep levels sorted. Then read the top-10 each side → the 40-col snapshot (interleaved
`ask_p1,ask_s1,bid_p1,bid_s1,…`, matching `stats.json.lob_columns`).
- **Build option A (recommended first):** a ~60-line Python `L2Book` (dict `tick→qty` per side)
  mirroring `book.rs` semantics — fast to write, keeps the rollout loop in one Python file.
- **Build option B (later):** call the real Rust `L2Book` via PyO3 for speed + exact parity.

### 3. Rollout loop  (`experiments/kaggle/p36_autoregressive.py`, new)
Extends p35: seed from a real window, run steps 1–7, collect the generated orders + evolving book
into the 46-col array. Runs on Kaggle GPU (each step = one diffusion sample, ~0.15 s → K=200 ×
50 rollouts ≈ 10k samples ≈ ~30–45 min on a T4).

### 4. Normalization plumbing
The model consumes NORMALIZED conditioning. The generated order is already normalized (append
directly). The book snapshot from APPLY is in real units → z-score the 40 LOB cols with p31 stats
before appending to `cond_lob`. (LOB cols are linear-z-scored — no log there.) Round-trip must be
exact; reuse `p31`'s mean/std from `stats.json`.

### 5. Stability guards + rejection sampling  (the hard part)
Errors **compound** over a rollout — the central risk. If the book drifts outside the normalized
range the model trained on, generations degrade → book degrades (feedback loop). Mitigations:
- **Short rollouts, many of them:** e.g. 50 independent rollouts × ~200 steps, not one long run —
  bounds drift and gives a distribution.
- **Rejection sampling** (as WorldAgent): resample orders that produce crossed/invalid/out-of-depth
  books; cap retries.
- **Health telemetry:** log spread, mid, level counts each step; abort a rollout if the book
  collapses (empty side) or explodes (spread ≫ real).

### 6. Evaluation
Denorm the fully-autoregressive stream → `p33_validate` against real. Now spread / ret / imbalance
are REAL tests (the book is generated); size / dt should hold near their current 0.11 / 0.15. This
is the true generative-market scorecard — [FIG-7] in the paper; write journal Entry + paper §3.8.

## Key decisions (flagged for the owner)
| decision | recommendation | why |
|---|---|---|
| own loop vs reuse ABIDES WorldAgent | **build our own** | avoid pulling in ABIDES; reuse our L2Book; fits the project |
| Python vs Rust book | **Python first, Rust later** | validate the approach fast; port for speed/parity |
| price source | **depth-reconstruction from book** | proven in WorldAgent; generated price col is noisy |
| rollout shape | **many short rollouts** | compounding drift is the main failure mode |

## Effort — ~3 focused sessions
1. **APPLY + price reconstruction** (Python L2Book), unit-tested against real order→book transitions
   from the export (apply the real event, check the book matches the next real snapshot).
2. **Rollout loop + normalization plumbing + rejection sampling**; smoke-test a 50-step rollout locally
   on CPU (tiny, no GPU) to prove the mechanics before Kaggle.
3. **Run on Kaggle, evaluate, tune stability**; journal + paper §3.8 + [FIG-7].

## Definition of done
A one-command autoregressive generation that rolls the book forward for K steps without collapsing,
whose KS scorecard reports spread/ret/imbalance as genuine (non-trivial) numbers — and, downstream
(Phase 4), whose backtest PnL gap vs. real replay is the real acceptance test (PLAN.md).

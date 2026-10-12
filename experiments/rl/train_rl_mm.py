"""P4.1 — Train the tiny RL market-maker (the 3rd ablation subject).

WHY THIS EXISTS (read docs/JOURNAL.md Entry 18): the ablation's finding — *which realism
component matters depends on the strategy* — is argued with two hand-written robots (fixed-offset
MM, OFI taker). A LEARNED strategy is a third, harder-to-dismiss probe: it shows the finding holds
even for a self-taught policy, not just toy rules.

THE DESIGN (the honest shape the user approved — "tiny policy, train in Python, run in Rust"):
  * The realism stack (calibrated fill / latency / impact) lives in Rust. To keep the RL subject
    apples-to-apples with MM/OFI it must be EVALUATED through that same Rust stack. So this script
    only TRAINS the brain; `core/src/bin/backtest.rs` (Strategy::RlMm) RUNS it.
  * The policy is deliberately tiny: a linear softmax over 5 standardized features -> a small
    discrete action grid (half-spread x skew). Few dozen weights, CPU-trainable in seconds, and
    cheap to forward-pass inside the Rust hot loop (no NN runtime needed in Rust).
  * Training env = a faithful NAIVE-fill replay of Day A (same event stream, grid, and fill rule as
    the Rust `naive` config). Training in the simplest config and EVALUATING frozen across all
    configs {naive,+fill,+latency,+impact} is a legitimate sim-to-sim transfer — documented as such
    in the paper's §4. Latency/impact are intentionally NOT in the training env: the policy must not
    be tuned to a particular realism toggle, or the ablation would be circular.

HONESTY RAIL: trains on Day A ONLY. Day B (held-out, Tardis) is never touched here — the Rust
evaluator loads the frozen weights and runs them on B. Enforced by this script taking a single
--npz and writing weights with a `trained_on` provenance stamp.

  python experiments/rl/train_rl_mm.py --npz data/btcusdt_20260501_0000_0656.npz

Output: experiments/rl/rl_mm.weights.json  (W matrix + feature norm + action table + provenance).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

# ---- event-stream constants: must mirror core/src/events.rs exactly -------------------------
DEPTH_EVENT = 1
TRADE_EVENT = 2
DEPTH_CLEAR_EVENT = 3
DEPTH_SNAPSHOT_EVENT = 4
BUY_EVENT = 1 << 29
SELL_EVENT = 1 << 28
LOCAL_EVENT = 1 << 30

# ---- market/strategy constants: must mirror core/src/bin/backtest.rs ------------------------
TICK_SIZE = 0.1
ORDER_QTY = 0.001        # BTC
MAX_POSITION = 0.005     # BTC
MAKER_FEE = 0.0002       # 2 bps, post-only
SKEW_TICKS_BASE = 10.0   # the MM's inventory-skew scale (RL picks a multiplier of it)
GRID_NS = 100_000_000    # 100 ms decision grid (matches the default --grid-ms)
OFI_LEVELS = 5           # top-N for the imbalance feature (matches the OFI taker)

# ---- the action grid: (half_spread ticks, skew multiplier) ----------------------------------
HALF_SPREADS = [20.0, 40.0, 60.0, 80.0]
SKEW_MULTS = [0.0, 1.0, 2.0]
ACTIONS = [(hs, sk) for hs in HALF_SPREADS for sk in SKEW_MULTS]   # 12 actions
N_ACTIONS = len(ACTIONS)
N_FEATURES = 5  # [bias, imbalance-0.5, position/MAX, (spread_ticks-mu)/sd, return*scale]


# =============================================================================================
# 1. Build a grid-sampled sequence from the npz: one row per 100 ms grid step, carrying the book
#    features AT the grid instant and the trades that occurred in the interval that follows it.
#    This reduces a ~26 M-event day to a few hundred-k lightweight steps the trainer can replay
#    many times — the faithful, fast training surface.
# =============================================================================================
def build_grid_sequence(npz_path, cache_path, max_events=0):
    if Path(cache_path).exists():
        z = np.load(cache_path, allow_pickle=False)
        return (z["mid"], z["imb"], z["spread_ticks"], z["trade_px"], z["trade_isbuy"],
                z["trade_step"], {"spread_mu": float(z["spread_mu"]),
                                  "spread_sd": float(z["spread_sd"]),
                                  "ret_scale": float(z["ret_scale"])})

    # Memory-light load (8 GB machine): decompress the member once, pull out ONLY the local-event
    # columns we need as compact numpy arrays, then free the ~1.6 GB structured array before the
    # loop — otherwise it stays resident and the pure-Python sweep thrashes in swap.
    t0 = time.time()
    with np.load(npz_path) as zf:
        data = zf["data"]
        ev_all = data["ev"].astype(np.uint64)
        local = (ev_all & LOCAL_EVENT) != 0
        ev = np.ascontiguousarray(ev_all[local])
        ts = np.ascontiguousarray(data["local_ts"][local])
        px = np.ascontiguousarray(data["px"][local])
        qty = np.ascontiguousarray(data["qty"][local])
        del data, ev_all, local
    if max_events and len(ev) > max_events:
        ev, ts, px, qty = ev[:max_events], ts[:max_events], px[:max_events], qty[:max_events]
    kind = (ev & 0xFF).astype(np.int64)          # low byte = event kind (events.rs::kind)
    is_buy = (ev & BUY_EVENT) != 0
    n = len(ev)
    print(f"[grid] loaded {n:,} local events in {time.time()-t0:.1f}s; sweeping...", flush=True)

    bids, asks = {}, {}  # price-level -> absolute qty
    grid_mid, grid_imb, grid_spread = [], [], []
    trade_px, trade_isbuy, trade_step = [], [], []

    def top_n_vol(side_map, best_fn, n_):
        if not side_map:
            return 0.0
        prices = sorted(side_map.keys(), reverse=best_fn)[:n_]
        return sum(side_map[p] for p in prices)

    next_grid = None
    step = -1
    for i in range(n):
        k, t = kind[i], int(ts[i])
        if next_grid is None:
            next_grid = (t // GRID_NS) * GRID_NS + GRID_NS

        # advance the grid: snapshot book features at each boundary we cross
        while t > next_grid:
            step += 1
            if bids and asks:
                bb, ba = max(bids), min(asks)
                mid = 0.5 * (bb + ba)
                bvol = top_n_vol(bids, True, OFI_LEVELS)
                avol = top_n_vol(asks, False, OFI_LEVELS)
                imb = bvol / (bvol + avol) if (bvol + avol) > 0 else 0.5
                grid_mid.append(mid); grid_imb.append(imb)
                grid_spread.append((ba - bb) / TICK_SIZE)
            else:
                grid_mid.append(np.nan); grid_imb.append(0.5); grid_spread.append(np.nan)
            next_grid += GRID_NS

        if k == DEPTH_EVENT or k == DEPTH_SNAPSHOT_EVENT:
            side = bids if is_buy[i] else asks
            p = px[i]
            if qty[i] <= 0:
                side.pop(p, None)
            else:
                side[p] = qty[i]
        elif k == DEPTH_CLEAR_EVENT:
            (bids if is_buy[i] else asks).clear()
        elif k == TRADE_EVENT and step >= 0:
            trade_px.append(px[i])
            trade_isbuy.append(1 if is_buy[i] else 0)
            trade_step.append(step)

        if (i & 0x1FFFFF) == 0 and i:                 # progress every ~2.1M events
            print(f"[grid]   {i:,}/{n:,} events ({100*i/n:.0f}%), {step+1:,} grid steps, "
                  f"{time.time()-t0:.0f}s", flush=True)

    mid = np.asarray(grid_mid); imb = np.asarray(grid_imb); spread = np.asarray(grid_spread)
    # normalization constants (baked into the weights so Rust computes identical features)
    fin_spread = spread[np.isfinite(spread)]
    spread_mu = float(np.nanmean(fin_spread)) if fin_spread.size else 0.0
    spread_sd = float(np.nanstd(fin_spread)) or 1.0
    fin_mid = mid[np.isfinite(mid)]
    ret_scale = 1.0 / (float(np.nanstd(np.diff(fin_mid) / fin_mid[:-1])) or 1e-6) if fin_mid.size > 2 else 1.0

    norm = {"spread_mu": spread_mu, "spread_sd": spread_sd, "ret_scale": ret_scale}
    np.savez(cache_path, mid=mid, imb=imb, spread_ticks=spread,
             trade_px=np.asarray(trade_px), trade_isbuy=np.asarray(trade_isbuy, dtype=np.int8),
             trade_step=np.asarray(trade_step, dtype=np.int64),
             spread_mu=spread_mu, spread_sd=spread_sd, ret_scale=ret_scale)
    print(f"[grid] built {len(mid)} grid steps, {len(trade_px)} trades in {time.time()-t0:.1f}s "
          f"-> cached {cache_path}")
    return mid, imb, spread, np.asarray(trade_px), np.asarray(trade_isbuy), np.asarray(trade_step), norm


# =============================================================================================
# 2. The policy: linear softmax over standardized features. Tiny on purpose.
# =============================================================================================
def features(mid, imb, spread_ticks, prev_mid, position, norm):
    ret = 0.0 if (not np.isfinite(prev_mid) or prev_mid == 0) else (mid - prev_mid) / prev_mid
    sp = 0.0 if not np.isfinite(spread_ticks) else (spread_ticks - norm["spread_mu"]) / norm["spread_sd"]
    return np.array([1.0, imb - 0.5, position / MAX_POSITION, sp, ret * norm["ret_scale"]])


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# =============================================================================================
# 3. One episode over a window of the grid sequence: place quotes, settle naive fills, accumulate
#    reward = equity increment - inventory penalty. Returns (total_reward, grads-ingredients).
# =============================================================================================
def run_episode(W, mid, imb, spread, trades_by_step, norm, start, length, rng, inv_pen, explore=True):
    position, cash, prev_mid = 0.0, 0.0, np.nan
    logp_terms = []  # (feature_vec, action_idx, probs) per step for REINFORCE
    rewards = []
    bid = ask = None  # resting quote prices for the interval
    for s in range(start, min(start + length, len(mid))):
        m = mid[s]
        if not np.isfinite(m):
            continue
        f = features(m, imb[s], spread[s], prev_mid, position, norm)
        probs = softmax(W @ f)
        a = rng.choice(N_ACTIONS, p=probs) if explore else int(np.argmax(probs))
        half_ticks, skew_mult = ACTIONS[a]
        half = half_ticks * TICK_SIZE
        skew = (position / MAX_POSITION) * skew_mult * SKEW_TICKS_BASE * TICK_SIZE
        bid = np.floor((m - half - skew) / TICK_SIZE) * TICK_SIZE if position < MAX_POSITION else None
        ask = np.ceil((m + half - skew) / TICK_SIZE) * TICK_SIZE if position > -MAX_POSITION else None

        equity0 = cash + position * m
        for (tpx, tbuy) in trades_by_step.get(s, ()):            # naive fills in this interval
            if (not tbuy) and bid is not None and tpx <= bid and position < MAX_POSITION:
                position += ORDER_QTY; cash -= bid * ORDER_QTY + MAKER_FEE * bid * ORDER_QTY; bid = None
            elif tbuy and ask is not None and tpx >= ask and position > -MAX_POSITION:
                position -= ORDER_QTY; cash += ask * ORDER_QTY - MAKER_FEE * ask * ORDER_QTY; ask = None
        equity1 = cash + position * m
        reward = (equity1 - equity0) - inv_pen * (position / MAX_POSITION) ** 2
        rewards.append(reward)
        logp_terms.append((f, a, probs))
        prev_mid = m
    # mark-to-market flatten at episode end (no free inventory at the finish)
    if logp_terms and np.isfinite(mid[min(start + length, len(mid)) - 1]):
        rewards[-1] += -abs(position) * 0.0  # inventory already penalized each step; keep simple
    return rewards, logp_terms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Day A (calibration day) — the ONLY data RL sees")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--windows", type=int, default=8, help="FIXED evaluation windows per epoch")
    ap.add_argument("--rollouts", type=int, default=4, help="stochastic rollouts per window (per-window baseline)")
    ap.add_argument("--window", type=int, default=2000, help="grid steps per episode (~3 min at 100ms)")
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--inv-pen", type=float, default=0.02, help="inventory penalty weight in reward")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-events", type=int, default=6_000_000,
                    help="cap local events for the training grid (8 GB RAM); 0 = all. A few hours "
                         "of Day A is ample for a ~60-weight policy and avoids swap thrash.")
    ap.add_argument("--out", default="experiments/rl/rl_mm.weights.json")
    args = ap.parse_args()

    cache = str(Path(args.npz).with_suffix("")) + ".rlgrid.npz"
    mid, imb, spread, tpx, tbuy, tstep, norm = build_grid_sequence(args.npz, cache, args.max_events)

    # index trades by grid step for O(1) interval lookup
    trades_by_step = {}
    for p, b, s in zip(tpx, tbuy, tstep):
        trades_by_step.setdefault(int(s), []).append((float(p), int(b)))

    rng = np.random.default_rng(args.seed)
    W = np.zeros((N_ACTIONS, N_FEATURES))        # start at uniform policy
    n_steps = len(mid)

    # FIXED evaluation windows, chosen once. Training each epoch on the SAME windows (not fresh
    # random ones) is the key variance fix: the gradient then reflects how the sampled ACTIONS did,
    # not which market segment happened to be drawn. Each window gets its OWN baseline (the mean
    # return of its rollouts this epoch), so advantage = "did these actions beat the average actions
    # on THIS window" — a proper per-state baseline, the standard REINFORCE variance reduction.
    fixed_starts = sorted(int(x) for x in
                          rng.integers(0, max(1, n_steps - args.window), size=args.windows))
    print(f"[train] {args.windows} fixed windows x {args.rollouts} rollouts, "
          f"window={args.window} steps, lr={args.lr}")

    hist = []
    t0 = time.time()
    for epoch in range(args.epochs):
        gradW = np.zeros_like(W)
        epoch_returns = []
        n_terms = 0
        for start in fixed_starts:
            roll = [run_episode(W, mid, imb, spread, trades_by_step, norm,
                                start, args.window, rng, args.inv_pen, explore=True)
                    for _ in range(args.rollouts)]
            roll = [(float(np.sum(r)), t) for r, t in roll if t]
            if not roll:
                continue
            w_baseline = float(np.mean([G for G, _ in roll]))   # per-window baseline
            for G, terms in roll:
                epoch_returns.append(G)
                advantage = G - w_baseline
                for (f, a, probs) in terms:                     # REINFORCE gradient
                    onehot = np.zeros(N_ACTIONS); onehot[a] = 1.0
                    gradW += advantage * np.outer(onehot - probs, f)
                    n_terms += 1
        if epoch_returns:
            W += args.lr * gradW / max(1, n_terms)              # normalize by #steps for a stable step
            hist.append(float(np.mean(epoch_returns)))
        if epoch % 25 == 0 or epoch == args.epochs - 1:
            print(f"[train] epoch {epoch:4d}  mean return {np.mean(epoch_returns):+.5f}", flush=True)

    out = {
        "schema": "rl_mm.v1",
        "trained_on": Path(args.npz).name,
        "note": "linear softmax policy; trained NAIVE-fill on Day A only; eval in Rust across all configs",
        "n_features": N_FEATURES,
        "feature_names": ["bias", "imbalance_minus_half", "position_over_max", "spread_z", "ret_scaled"],
        "norm": norm,
        "actions": [{"half_spread_ticks": hs, "skew_mult": sk} for (hs, sk) in ACTIONS],
        "W": W.tolist(),
        "train": {"epochs": args.epochs, "windows": args.windows, "rollouts": args.rollouts,
                  "window": args.window, "lr": args.lr, "inv_pen": args.inv_pen, "seed": args.seed},
        "final_mean_return": hist[-1] if hist else None,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    write_txt(out, str(Path(args.out).with_suffix(".txt")))
    print(f"[train] done in {time.time()-t0:.1f}s -> {args.out}  (final mean return {out['final_mean_return']})")


def write_txt(out, path):
    """Plain whitespace/newline weights for the Rust loader (core keeps zero deps beyond flate2).
    Schema (must match Policy::load in core/src/bin/backtest.rs):
      line1: rl_mm_v1
      line2: n_features n_actions
      line3: spread_mu spread_sd ret_scale
      then n_actions lines: half_spread_ticks skew_mult
      then n_actions lines: W row (n_features values)."""
    n_f, n_a = out["n_features"], len(out["actions"])
    lines = ["rl_mm_v1", f"{n_f} {n_a}",
             f'{out["norm"]["spread_mu"]:.10g} {out["norm"]["spread_sd"]:.10g} {out["norm"]["ret_scale"]:.10g}']
    lines += [f'{a["half_spread_ticks"]:.10g} {a["skew_mult"]:.10g}' for a in out["actions"]]
    lines += [" ".join(f"{v:.10g}" for v in row) for row in out["W"]]
    Path(path).write_text("\n".join(lines) + "\n")
    print(f"[train] wrote Rust-loadable weights -> {path}")


if __name__ == "__main__":
    main()

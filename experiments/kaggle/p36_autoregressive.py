"""P3.4 step 3 — autoregressive book coupling (rollout loop).

Rolls the trained model forward as a self-consistent market: each generated order is APPLIED to a
book (autoreg_book.L2Book) and the next order is conditioned on the ROLLED-FORWARD book — so the
book-derived facts (spread/ret/imbalance) become genuine tests instead of trivially 0. See
docs/P34_AUTOREGRESSIVE_DESIGN.md.

The model call is pluggable (`sample_fn`) so the loop mechanics — windowing, normalization
round-trip, APPLY integration, rejection sampling, output assembly — are smoke-tested locally on
CPU with a mock that replays REAL orders (the autoreg book should then track the real book), with
no torch/GPU. On Kaggle, `sample_fn` wraps DiffusionEngine.sample.

  python experiments/kaggle/p36_autoregressive.py --smoke        # local, no GPU
  python experiments/kaggle/p36_autoregressive.py --deepmarket ... --ckpt ... --data ...  # Kaggle
"""
import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autoreg_book import L2Book, kind_from_model, reconstruct_price_tick  # noqa: E402

# frozen 3x3 type-embedder weights (DeepMarket diffusion_engine.py:57) — used to (de)embed the class
TYPE_W = np.array([[0.4438, -0.2984, 0.2888], [0.8249, 0.5847, 0.1448], [1.5600, -1.2847, 1.0294]])
P31 = Path(__file__).resolve().parents[2] / "experiments" / "data" / "p31"


class Norm:
    """p31 normalization helpers driven by stats.json (log-then-zscore forward; unzscore-then-exp inverse)."""

    def __init__(self, stats):
        self.of = stats["order_features"]
        self.mean = np.asarray(stats["mean"], float)
        self.std = np.asarray(stats["std"], float)
        self.zcols = set(stats["z_scored_cols"])
        self.logcols = {self.of.index(n) for n in stats.get("log_cols", [])}
        self.n_ord = len(self.of)
        self.n_lob = len(stats["lob_columns"])

    def denorm_order(self, o):
        """normalized 6-vec [dt,type,size,price,dir,depth] -> real units (exp on log cols)."""
        out = np.array(o, float)
        for c in range(self.n_ord):
            if c in self.zcols:
                out[c] = out[c] * self.std[c] + self.mean[c]
            if c in self.logcols:
                out[c] = np.exp(out[c])
        return out

    def norm_lob(self, snap_real):
        """40 real ticks -> z-scored (LOB cols are never log-transformed)."""
        out = np.array(snap_real, float)
        for j in range(self.n_lob):
            c = self.n_ord + j
            out[j] = (out[j] - self.mean[c]) / self.std[c]
        return out

    def denorm_lob(self, snap_norm):
        out = np.array(snap_norm, float)
        for j in range(self.n_lob):
            c = self.n_ord + j
            out[j] = out[j] * self.std[c] + self.mean[c]
        return out


def de_embed(g8, ste=3):
    """8-wide model output -> normalized 6-vec [dt, class, size, price, dir, depth]."""
    dt = g8[0]
    cls = int(np.argmin(np.abs(TYPE_W - g8[1:1 + ste]).sum(axis=1)))
    size, price = g8[1 + ste], g8[2 + ste]
    direction = 1.0 if g8[3 + ste] >= 0 else -1.0
    depth = g8[4 + ste]
    return np.array([dt, cls, size, price, direction, depth], float)


def run_rollout(sample_fn, val, stats, K, seed_index=0, reject=True, max_retries=5, log_every=0):
    """Roll the book forward K steps. Returns (out_rows[K,46] normalized, health dict)."""
    nrm = Norm(stats)
    orders = val[:, :nrm.n_ord]
    lob_raw = val[:, nrm.n_ord:nrm.n_ord + nrm.n_lob]
    lob_rolled = np.roll(lob_raw, 1, axis=0); lob_rolled[0] = 0.0   # book BEFORE each order (LOBDataset)

    SEQ, COND = 256, 255
    i = seed_index
    order_win = deque(orders[i:i + COND].tolist(), maxlen=COND)      # 255 cond orders (normalized)
    lob_win = deque(lob_rolled[i:i + SEQ].tolist(), maxlen=SEQ)      # 256 cond books (normalized, before)
    book = L2Book.from_snapshot(nrm.denorm_lob(lob_rolled[i + COND]))  # current book = before the target

    out = np.zeros((K, nrm.n_ord + nrm.n_lob), float)
    n_reject = crossed = empty = 0
    for k in range(K):
        cond_orders = np.asarray(order_win)                          # (255,6)
        cond_lob = np.asarray(lob_win)                               # (256,40)
        order6 = None
        for _retry in range(max_retries + 1):
            g8 = sample_fn(cond_orders, cond_lob, step=k)
            o = de_embed(g8)
            real = nrm.denorm_order(o)
            kind = kind_from_model(real[1])
            side = int(real[4])
            ptick = reconstruct_price_tick(real[5], side, book)      # from depth+side+book
            trial = L2Book.from_snapshot(book.snapshot())            # cheap copy to test validity
            trial.apply(kind, real[2], ptick, side)
            bb, ba = trial.best_bid_tick(), trial.best_ask_tick()
            ok = bb is not None and ba is not None and bb < ba
            if ok or not reject:
                order6 = o
                book.apply(kind, real[2], ptick, side)
                break
            n_reject += 1
        if order6 is None:                                           # all retries invalid: keep book, skip apply
            order6 = o
        snap_real = book.snapshot()
        bb, ba = book.best_bid_tick(), book.best_ask_tick()
        if bb is None or ba is None:
            empty += 1
        elif bb >= ba:
            crossed += 1
        snap_norm = nrm.norm_lob(snap_real)
        out[k, :nrm.n_ord] = order6
        out[k, nrm.n_ord:] = snap_norm
        order_win.append(order6.tolist())
        lob_win.append(snap_norm.tolist())
        if log_every and (k + 1) % log_every == 0:
            sp = (ba - bb) if (bb is not None and ba is not None) else -1
            print(f"[rollout] {k+1}/{K}  spread_ticks={sp}  rejects={n_reject}", flush=True)

    health = {"steps": K, "rejects": n_reject, "crossed": crossed, "empty_side": empty}
    return out, health


# ---------------------------------------------------------------------------------------------
def smoke_test(n_steps=300, seed_index=1000):
    """No GPU: mock sample_fn REPLAYS real orders (re-embedded). The autoreg book should then track
    the real after-books. Verifies windowing shapes, norm round-trip, APPLY, and book health."""
    stats = json.loads((P31 / "stats.json").read_text())
    val = np.load(P31 / "val.npy").astype(np.float64)
    nrm = Norm(stats)

    # mock: return the real order at (seed_index+255+step), re-embedded to 8-wide
    def sample_fn(cond_orders, cond_lob, step):
        o = val[seed_index + 255 + step, :nrm.n_ord]                 # real normalized 6-vec
        cls = int(round(o[1]))
        return np.concatenate([[o[0]], TYPE_W[cls], [o[2], o[3], o[4], o[5]]])

    # sanity: norm round-trip on a book snapshot
    snap = val[seed_index, nrm.n_ord:]
    rt = nrm.norm_lob(nrm.denorm_lob(snap))
    print(f"[smoke] lob norm round-trip max err: {np.abs(rt - snap).max():.2e}")

    out, health = run_rollout(sample_fn, val, stats, K=n_steps, seed_index=seed_index,
                              reject=True, log_every=100)
    print(f"[smoke] health: {health}")

    # how well did the autoreg book track the REAL after-books? (top-of-book, robust to horizon)
    real_after = val[seed_index + 255: seed_index + 255 + n_steps, nrm.n_ord:]
    gen = out[:, nrm.n_ord:]
    # compare denormalized top-of-book (best ask/bid price+size)
    gd = np.array([nrm.denorm_lob(r) for r in gen])[:, :4]
    rd = np.array([nrm.denorm_lob(r) for r in real_after])[:, :4]
    l1_match = np.mean(np.all(np.isclose(gd, rd, atol=0.5), axis=1))
    print(f"[smoke] autoreg-vs-real top-of-book match over {n_steps} replayed steps: {100*l1_match:.1f}%")
    print(f"[smoke] {'PASS' if health['crossed']==0 and health['empty_side']==0 and l1_match>0.8 else 'CHECK'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="local CPU plumbing test (no GPU)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--deepmarket"); ap.add_argument("--data"); ap.add_argument("--ckpt")
    ap.add_argument("--augment-dim", type=int, default=64); ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--n-rollouts", type=int, default=50); ap.add_argument("--rollout-len", type=int, default=200)
    ap.add_argument("--out", default="/kaggle/working/autoreg.npy")
    args = ap.parse_args()

    if args.smoke:
        smoke_test(args.steps)
        return

    # --- Kaggle path: wrap the real model as sample_fn (built in piece 3b / next step) ---
    raise SystemExit("GPU rollout wiring (real DiffusionEngine sample_fn) lands in the next step; "
                     "run --smoke to validate the loop mechanics first.")


if __name__ == "__main__":
    main()

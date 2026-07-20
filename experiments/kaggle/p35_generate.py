"""P3.4 step 2 — generate an event stream from the trained TRADES checkpoint.

FIRST-PASS (non-autoregressive) generation, per docs/JOURNAL.md Entry 13: slide real
256-order windows through val.npy and call DiffusionEngine.sample() once per window — each
generated order is conditioned on the REAL past 255 orders + the REAL book. We then de-embed
the categorical type, keep the model's normalized numeric columns, pair each generated order
with its REAL book snapshot at the same timestamp, and write a 46-column array in p31's exact
layout so scripts/p34_denorm.py inverts it and scripts/p33_validate.py scores it. This isolates
the ORDER-FLOW question (does the model produce realistic dt / size / type / direction?) from
the harder autoregressive-book question, which is the next step.

Reference (copied, with attribution): DeepMarket ABIDES/agent/WorldAgent.py
  * _generate_order (l.274): builds cond_orders / x=zeros / cond_lob, calls
        self.model.sample(cond_orders=..., x=..., cond_lob=...)
  * _postprocess_generated_TRADES (l.493): type recovery via nearest row (L1) of the frozen
        type_embedder weight; direction = sign; size at [ste+1], depth at [-1], dt at [0].
Layout after type_embedding (LEN_ORDER 6 -> 5+ste wide, ste=size_type_emb=3, so 8):
    [dt, e0..e_{ste-1}, size, price, direction, depth]

Runs on Kaggle GPU (needs the DeepMarket repo + the checkpoint). Not testable in the dev env
(no torch/GPU); the kernel that drives it downloads the checkpoint as a Kaggle dataset.

Usage:
  python experiments/kaggle/p35_generate.py --deepmarket /kaggle/working/DeepMarket \
      --data /kaggle/input/hft-p31new --ckpt /kaggle/input/hft-trades-ckpt/<file>.ckpt \
      --n 3000 --out /kaggle/working/generated.npy --real-out /kaggle/working/real.npy
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deepmarket", required=True, help="path to the cloned DeepMarket repo")
    ap.add_argument("--data", required=True, help="dir containing val.npy (p31 layout)")
    ap.add_argument("--ckpt", required=True, help="trained TRADES .ckpt")
    ap.add_argument("--n", type=int, default=3000, help="number of orders to generate")
    ap.add_argument("--augment-dim", type=int, default=64, help="must match the trained model")
    ap.add_argument("--depth", type=int, default=8, help="must match the trained model")
    ap.add_argument("--stock-slot", default="INTC")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/kaggle/working/generated.npy")
    ap.add_argument("--real-out", default="/kaggle/working/real.npy")
    args = ap.parse_args()

    dm = Path(args.deepmarket).resolve()
    if not (dm / "run.py").exists():
        sys.exit(f"DeepMarket not found at {dm} (expected run.py)")
    sys.path.insert(0, str(dm))
    os.chdir(dm)  # DeepMarket uses relative imports/paths

    import numpy as np
    import torch
    import constants as cst
    import configuration
    from run import HP_DICT_MODEL
    from preprocessing.LOBDataset import LOBDataset
    from models.diffusers.diffusion_engine import DiffusionEngine

    LHP = cst.LearningHyperParameter

    # Rebuild the SAME config the model was trained with, so the architecture matches the ckpt.
    fixed = HP_DICT_MODEL[cst.Models.TRADES].fixed
    fixed[LHP.AUGMENT_DIM.value] = args.augment_dim
    fixed[LHP.CDT_DEPTH.value] = args.depth

    config = configuration.Configuration()
    config.CHOSEN_MODEL = cst.Models.TRADES
    config.CHOSEN_STOCK = [cst.Stocks[args.stock_slot]]
    config.IS_DATA_PREPROCESSED = True
    config.IS_WANDB = False
    config.IS_SWEEP = False
    config.IS_TRAINING = False
    config.IS_DEBUG = False
    for p in cst.LearningHyperParameter:
        if p.value in fixed:
            config.HYPER_PARAMETERS[p] = fixed[p.value]
    config.FILENAME_CKPT = "GEN"

    # PyTorch >=2.6 defaults torch.load(weights_only=True), which cannot unpickle the custom
    # `config` object Lightning stored in the checkpoint hparams. We trust our own checkpoint,
    # so force full unpickling for every torch.load (incl. Lightning's internal call).
    _orig_torch_load = torch.load
    def _torch_load(*a, **k):
        k["weights_only"] = False
        return _orig_torch_load(*a, **k)
    torch.load = _torch_load

    device = cst.DEVICE
    seq_size = config.HYPER_PARAMETERS[LHP.SEQ_SIZE]           # 256
    gen_seq_size = config.HYPER_PARAMETERS[LHP.MASKED_SEQ_SIZE]  # 1
    print(f"[cfg] device={device} seq_size={seq_size} gen_seq_size={gen_seq_size} "
          f"augment_dim={args.augment_dim} depth={args.depth}", flush=True)

    # Load the trained (EMA) weights.
    model = DiffusionEngine.load_from_checkpoint(args.ckpt, config=config, map_location=device)
    model.eval().to(device)
    model.training = False
    ste = model.size_type_emb                                   # 3
    type_w = model.type_embedder.weight.data.detach().cpu().numpy()  # (3, ste)
    print(f"[model] loaded ckpt; size_type_emb={ste} type_embedder_weight_shape={type_w.shape}", flush=True)

    # Conditioning comes from LOBDataset (identical windowing to training). is_val=False loads
    # the whole array; __getitem__(i) -> (cond[255,6], x0[1,6], lob[256,40]).
    val_path = os.path.join(args.data, "val.npy")
    ds = LOBDataset([val_path], seq_size=seq_size, gen_seq_size=gen_seq_size,
                    chosen_model=cst.Models.TRADES, is_val=False)
    raw = np.load(val_path)  # (T, 46) normalized — source of the REAL book + REAL order per index
    n_windows = len(ds)
    N = min(args.n, n_windows)
    print(f"[data] val rows={raw.shape[0]} windows={n_windows} generating N={N}", flush=True)

    rng = np.random.default_rng(args.seed)
    idxs = np.sort(rng.choice(n_windows, size=N, replace=False))

    gen_rows = np.zeros((N, raw.shape[1]), dtype=np.float32)
    real_rows = np.zeros((N, raw.shape[1]), dtype=np.float32)
    n_neg_size = 0

    with torch.no_grad():
        for k, index in enumerate(idxs):
            cond, x0, lob = ds[int(index)]
            cond = cond.unsqueeze(0).to(device)
            lob = lob.unsqueeze(0).to(device)
            x = torch.zeros(1, gen_seq_size, cst.LEN_ORDER, device=device, dtype=torch.float32)
            g = model.sample(cond_orders=cond, x=x, cond_lob=lob)[0, 0].detach().cpu().numpy()  # (5+ste,)

            # --- de-embed (WorldAgent._postprocess_generated_TRADES) ---
            dt = g[0]
            etype = int(np.argmin(np.sum(np.abs(type_w - g[1:1 + ste]), axis=1)))  # 0/1/2 (no +1: p34 wants 0-based)
            size = g[1 + ste]
            price = g[2 + ste]
            direction = 1.0 if g[3 + ste] >= 0 else -1.0
            depth = g[4 + ste]
            if size < 0:
                n_neg_size += 1  # kept anyway; p34/p33 see the raw model output (honest)

            # target order sits at absolute row index_x = cond_seq_size + index + gen_seq_size
            index_x = (seq_size - gen_seq_size) + int(index) + gen_seq_size
            real_row = raw[index_x]                        # the REAL 46-col row at this position
            # generated order columns + REAL book (first-pass: book is real, order is generated)
            gen_rows[k, 0] = dt
            gen_rows[k, 1] = etype
            gen_rows[k, 2] = size
            gen_rows[k, 3] = price
            gen_rows[k, 4] = direction
            gen_rows[k, 5] = depth
            gen_rows[k, 6:] = real_row[6:]                 # real LOB (40 cols)
            real_rows[k] = real_row

            if (k + 1) % 500 == 0:
                print(f"[gen] {k + 1}/{N}", flush=True)

    np.save(args.out, gen_rows)
    np.save(args.real_out, real_rows)
    print(f"[done] wrote {args.out} {gen_rows.shape} and {args.real_out} {real_rows.shape}; "
          f"neg_size={n_neg_size} ({100 * n_neg_size / N:.1f}%)", flush=True)


if __name__ == "__main__":
    main()

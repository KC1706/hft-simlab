"""P3.2 — Kaggle training driver for TRADES on our crypto data.

Runs on a Kaggle free-tier GPU (T4). It wires our packed arrays
(experiments/data/p31/{train,val}.npy) into DeepMarket's training loop, widens the
conditional diffusion transformer toward the ~10M-parameter budget (PLAN P3.1), prints
the *actual* parameter count, and launches PyTorch-Lightning training.

WHY THIS WIRES CLEANLY: DeepMarket's LOBDataset loads a single array and slices
`[:, :LEN_ORDER]` (6) as the order features and the remaining 40 columns as the LOB
condition — exactly the layout scripts/p31_pack_trades.py emits. So with
IS_DATA_PREPROCESSED=True the LOBSTER builder is skipped and our arrays train directly.

NOT TESTED LOCALLY (no GPU/torch in the dev env). Run the --smoke path first on Kaggle
to validate wiring (tiny data, depth-1, 1 epoch) before the full run. See README.md.

Usage on Kaggle (GPU on):
    python experiments/kaggle/p32_train_trades.py --deepmarket /kaggle/working/DeepMarket \
        --data experiments/data/p31 [--smoke] [--augment-dim 128] [--depth 10] [--epochs 50]
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

# Force a single GPU: Kaggle's T4 x2 makes Lightning default to distributed (DDP), whose
# subprocess re-launcher breaks under our chdir. One GPU is plenty for a ~10M model.
# Must be set before torch is imported (DeepMarket imports it transitively).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deepmarket", required=True, help="path to the cloned DeepMarket repo")
    ap.add_argument("--data", default="experiments/data/p31", help="dir with train.npy/val.npy")
    ap.add_argument("--stock-slot", default="INTC", help="DeepMarket stock dir to reuse")
    ap.add_argument("--augment-dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--smoke", action="store_true", help="tiny wiring test (depth 1, 1 epoch)")
    ap.add_argument("--count-only", action="store_true", help="print param count for the given dims and exit")
    args = ap.parse_args()

    dm = Path(args.deepmarket).resolve()
    if not (dm / "run.py").exists():
        sys.exit(f"DeepMarket not found at {dm} (expected run.py). Clone it first — see README.md")
    sys.path.insert(0, str(dm))

    # 1) Place our pre-packed, pre-normalized arrays where LOBDataset expects them.
    data_src = Path(args.data).resolve()
    dest = dm / "data" / args.stock_slot
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("train.npy", "val.npy"):
        src = data_src / name
        if not src.exists():
            sys.exit(f"missing {src} — run lob_export + scripts/p31_pack_trades.py first")
        shutil.copy(src, dest / name)
    print(f"[data] copied train/val.npy -> {dest}")

    # Guarded fix for data packed before the event_type remap: TRADES' type embedding has
    # 3 slots (0=submission, 1=cancel/delete, 2=execution); raw codes {1,2,3,4} index out of
    # bounds. Remap in place only if raw codes are present (max > 2), so it's idempotent.
    import numpy as np
    for name in ("train.npy", "val.npy"):
        p = dest / name
        a = np.load(p)
        if a[:, 1].max() > 2:  # column 1 = event_type
            et = a[:, 1].astype(int)
            a[:, 1] = np.where(et == 1, 0, np.where(et == 4, 2, 1)).astype(a.dtype)
            np.save(p, a)
            print(f"[fix] remapped raw event_type {{1,2,3,4}}->{{0,1,1,2}} in {name}")

    # DeepMarket uses relative paths ("data/INTC/train.npy", "data/checkpoints/...") so we
    # must run from its repo root regardless of where this driver was launched.
    os.chdir(dm)

    # 2) Import DeepMarket and configure for our data + a from-scratch TRADES run.
    import constants as cst
    import configuration
    from run import HP_DICT_MODEL, run

    LHP = cst.LearningHyperParameter
    # Widen the CDT toward the ~10M-param budget by patching the model's fixed hparams
    # (run() copies these over config defaults, so this is the authoritative knob).
    fixed = HP_DICT_MODEL[cst.Models.TRADES].fixed
    fixed[LHP.AUGMENT_DIM.value] = args.augment_dim
    fixed[LHP.CDT_DEPTH.value] = 1 if args.smoke else args.depth
    fixed[LHP.BATCH_SIZE.value] = 16 if args.smoke else args.batch_size
    fixed[LHP.EPOCHS.value] = 1 if args.smoke else args.epochs

    config = configuration.Configuration()
    config.CHOSEN_MODEL = cst.Models.TRADES
    config.CHOSEN_STOCK = [cst.Stocks[args.stock_slot]]
    config.IS_DATA_PREPROCESSED = True   # use our arrays, skip the LOBSTER builder
    config.IS_WANDB = False
    config.IS_SWEEP = False
    config.IS_TRAINING = True
    config.IS_DEBUG = bool(args.smoke)

    # 3) Report the real parameter count before committing to a full run.
    try:
        # apply the fixed hparams the way run() will, then instantiate to count.
        for p in cst.LearningHyperParameter:
            if p.value in fixed:
                config.HYPER_PARAMETERS[p] = fixed[p.value]
        config.FILENAME_CKPT = "PROBE"  # run() sets this; the probe runs before run()
        from models.diffusers.diffusion_engine import DiffusionEngine
        model = DiffusionEngine(config)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[model] TRADES CDT — augment_dim={args.augment_dim} depth={fixed[LHP.CDT_DEPTH.value]} "
              f"-> {n_params/1e6:.2f}M parameters")
        if not args.smoke and not (5e6 <= n_params <= 15e6):
            print(f"[warn] {n_params/1e6:.1f}M is outside the 5-15M target — adjust --augment-dim/--depth")
    except Exception as e:  # counting is best-effort; training still proceeds
        print(f"[model] param-count probe skipped ({type(e).__name__}: {e})")

    if args.count_only:
        print("[count-only] exiting before training.")
        return

    # 4) Train. run() builds the Lightning Trainer (EarlyStopping on val_ema_loss) and fits.
    accelerator = "gpu" if cst.DEVICE.startswith("cuda") else "cpu"
    print(f"[train] accelerator={accelerator} epochs={fixed[LHP.EPOCHS.value]} "
          f"batch={fixed[LHP.BATCH_SIZE.value]} smoke={args.smoke}")
    run(config, accelerator)
    print(f"[done] checkpoint written under {dm/'data'/'checkpoints'} "
          f"(filename starts with {args.stock_slot}_). Download it from Kaggle output.")


if __name__ == "__main__":
    main()

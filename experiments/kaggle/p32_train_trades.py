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
    ap.add_argument("--grad-clip", type=float, default=5.0,
                    help="gradient-norm clip (DeepMarket's own remedy for NaN diffusion loss; 0=off)")
    # --- Throughput / early-stop controls (added after the 2026-07-19 run: each epoch cost
    #     ~2.5h because validation = 100 diffusion-sampling batches x2/epoch (~35min each),
    #     and EarlyStopping(patience=6, checked twice/epoch) killed it at epoch 0. See
    #     docs/JOURNAL.md P3.2 debugging log. These knobs make validation cheap and give the
    #     model room to actually learn.) ---
    ap.add_argument("--val-batches", type=int, default=20,
                    help="cap validation batches per check (Trainer limit_val_batches); val is the wall-clock bottleneck")
    ap.add_argument("--val-every", type=float, default=1.0,
                    help="validation frequency (Trainer val_check_interval); 1.0 = once per epoch (run.py default is 0.5 = twice)")
    ap.add_argument("--patience", type=int, default=15,
                    help="EarlyStopping patience override (run.py hardcodes 6, far too tight for noisy diffusion val)")
    ap.add_argument("--limit-train-batches", type=int, default=0,
                    help="cap training steps per epoch (0 = full ~3124); used by --probe to make epochs fast")
    ap.add_argument("--probe", action="store_true",
                    help="fast diagnostic run: 6 epochs, capped train/val batches, early-stop OFF — confirms loss actually decreases before committing a full ~9h session")
    # --- Learning-rate overrides (added after the first probe showed the model NOT learning:
    #     val_loss_simple pinned at 2.74, train loss rising 2.77->3.6. DeepMarket's base LR is
    #     0.001 and its type-embedder param-group is hardcoded to lr=0.01 (diffusion_engine.py
    #     line 248) which the scheduler never decays — both too hot for our 10M model.) ---
    ap.add_argument("--lr", type=float, default=0.0,
                    help="override base learning rate (0 = DeepMarket default 0.001)")
    ap.add_argument("--embed-lr", type=float, default=0.0,
                    help="override the type-embedder param-group LR (0 = DeepMarket default 0.01, hardcoded and never decayed)")
    ap.add_argument("--resume", default="",
                    help="path to a .ckpt to continue training from (Lightning ckpt_path resume: "
                         "restores weights + optimizer + epoch counter). --epochs is the ABSOLUTE "
                         "target, so to add N epochs to a ckpt at epoch=E set --epochs E+1+N.")
    args = ap.parse_args()

    # --probe presets (only fill values the user did not override on the CLI).
    if args.probe:
        if args.epochs == 50:
            args.epochs = 6
        if args.limit_train_batches == 0:
            args.limit_train_batches = 600      # ~3.5 min/epoch at ~2.8 it/s vs ~90 min full
        if args.val_batches == 20:
            args.val_batches = 10               # ~4 min/val vs ~35 min at 100
        args.patience = 10 ** 6                 # never early-stop during a probe; we read the trend

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
    if args.lr:                                  # base LR (diffuser param group)
        fixed[LHP.LEARNING_RATE.value] = args.lr

    # Override the hardcoded type-embedder LR (diffusion_engine.py:248 sets it to 0.01 in a
    # separate Adam param group the scheduler never touches). We can't edit refs/, so patch
    # configure_optimizers at runtime to reset that group's lr after the optimizer is built.
    if args.embed_lr:
        from models.diffusers.diffusion_engine import DiffusionEngine as _DE
        _orig_cfg = _DE.configure_optimizers
        def _cfg_with_embed_lr(self):
            opt = _orig_cfg(self)
            # Adam path builds [diffuser, type_embedder]; the embedder group is the one
            # carrying an explicit per-group 'lr'. Reset every non-base group to be safe.
            for g in self.optimizer.param_groups[1:]:
                g["lr"] = args.embed_lr
            return opt
        _DE.configure_optimizers = _cfg_with_embed_lr

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

    # Patch lightning.Trainer so run()'s `L.Trainer(...)` call (attribute lookup at call time)
    # picks up our overrides. run() builds the Trainer with values we must change:
    #   * gradient_clip_val — DeepMarket's own remedy for NaN diffusion loss (absent in run.py)
    #   * val_check_interval=0.5 and 100 val batches — the wall-clock killer (~35min x2/epoch)
    #   * EarlyStopping(patience=6) — too tight; killed the 2026-07-19 run at epoch 0
    # setdefault can't override keys run.py already passes, so for those we assign directly.
    if not args.count_only:
        import lightning as L
        from lightning.pytorch.callbacks.early_stopping import EarlyStopping
        _OrigTrainer = L.Trainer

        if args.resume:  # PyTorch>=2.6 weights_only=True can't unpickle the config in the ckpt
            import torch as _torch
            _orig_load = _torch.load
            def _full_load(*la, **lk):
                lk["weights_only"] = False
                return _orig_load(*la, **lk)
            _torch.load = _full_load

        def _PatchedTrainer(*a, **k):
            if args.grad_clip:
                k.setdefault("gradient_clip_val", args.grad_clip)
            k["val_check_interval"] = args.val_every          # override run.py's 0.5
            k["limit_val_batches"] = args.val_batches         # cap the expensive sampling val
            if args.limit_train_batches:
                k["limit_train_batches"] = args.limit_train_batches
            for cb in (k.get("callbacks") or []):             # relax EarlyStopping in place
                if isinstance(cb, EarlyStopping):
                    cb.patience = args.patience
            trainer = _OrigTrainer(*a, **k)
            if args.resume:                                   # inject ckpt_path into run()'s fit()
                _orig_fit = trainer.fit
                def _fit_resume(*fa, **fk):
                    fk.setdefault("ckpt_path", args.resume)
                    print(f"[train] resuming from {args.resume}", flush=True)
                    return _orig_fit(*fa, **fk)
                trainer.fit = _fit_resume
            return trainer

        L.Trainer = _PatchedTrainer
        print(f"[train] overrides: grad_clip={args.grad_clip} val_every={args.val_every} "
              f"val_batches={args.val_batches} patience={args.patience} "
              f"limit_train_batches={args.limit_train_batches or 'full'} "
              f"lr={args.lr or 'default(0.001)'} embed_lr={args.embed_lr or 'default(0.01)'} "
              f"resume={args.resume or 'no'}")

    # 4) Train. run() builds the Lightning Trainer (EarlyStopping on val_ema_loss) and fits.
    accelerator = "gpu" if cst.DEVICE.startswith("cuda") else "cpu"
    print(f"[train] accelerator={accelerator} epochs={fixed[LHP.EPOCHS.value]} "
          f"batch={fixed[LHP.BATCH_SIZE.value]} smoke={args.smoke}")
    run(config, accelerator)
    print(f"[done] checkpoint written under {dm/'data'/'checkpoints'} "
          f"(filename starts with {args.stock_slot}_). Download it from Kaggle output.")


if __name__ == "__main__":
    main()

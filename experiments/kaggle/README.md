# P3.2 — Training TRADES on Kaggle (free T4 GPU)

The generative market model (TRADES) trains here because the dev machine has no GPU.
Everything else in the project runs locally; only this step needs the free Kaggle GPU.

## 0. One-time: prepare the training arrays (local)

Export a sizeable contiguous window and pack it (more rows than the 50k smoke default):

```bash
cd hft-simlab
core/target/release/lob_export data/btcusdt_20260501_0000_0656.npz \
    --out /tmp/rows.csv --max-rows 2000000
.venv/bin/python scripts/p31_pack_trades.py /tmp/rows.csv
# -> experiments/data/p31/{train.npy, val.npy, stats.json}
```

Keep `stats.json` — it holds the means/stds needed to **invert the normalization** when
the trained model generates events (P3.4). The model trains on normalized values; to turn
a generated event back into prices/sizes you de-normalize with these.

## 1. On Kaggle

1. New Notebook → **Settings → Accelerator → GPU T4 ×2** (or P100), Internet **On**.
2. In the first cell, clone the model repo and this project, install deps:
   ```python
   !git clone --depth 1 https://github.com/LeonardoBerti00/DeepMarket /kaggle/working/DeepMarket
   !git clone --depth 1 https://github.com/KC1706/hft-simlab /kaggle/working/hft-simlab
   !pip install -q -r /kaggle/working/DeepMarket/requirements.txt
   ```
3. Upload `experiments/data/p31/{train,val}.npy` as a **Kaggle Dataset** (or regenerate on
   Kaggle from the repo + data), and note its path, e.g. `/kaggle/input/hft-p31`.
4. **Smoke test the wiring first** (tiny data, depth-1, 1 epoch — finishes in ~1 min):
   ```python
   !cd /kaggle/working/hft-simlab && python experiments/kaggle/p32_train_trades.py \
       --deepmarket /kaggle/working/DeepMarket --data /kaggle/input/hft-p31 --smoke
   ```
   This must print `[model] ... -> N.NNM parameters` and complete one epoch without error.
5. **Full run** (prints the real param count, trains to EarlyStopping):
   ```python
   !cd /kaggle/working/hft-simlab && python experiments/kaggle/p32_train_trades.py \
       --deepmarket /kaggle/working/DeepMarket --data /kaggle/input/hft-p31 \
       --augment-dim 128 --depth 10 --epochs 50
   ```
6. Download the checkpoint from `DeepMarket/data/checkpoints/INTC_*.ckpt` (Kaggle output).

## How it wires (why it works)

- DeepMarket's `LOBDataset` loads one array and slices `[:, :6]` as order features and the
  remaining 40 columns as the LOB condition — **exactly** what `p31_pack_trades.py` emits,
  so `IS_DATA_PREPROCESSED=True` lets our arrays train directly (the LOBSTER builder is
  skipped). We reuse the `INTC` stock slot as the on-disk location.
- Model size is set by patching `HP_DICT_MODEL[TRADES].fixed` (which `run()` applies over
  the config defaults). The reference CDT is ~1–2 M params at `augment_dim=64, depth=8`;
  `--augment-dim 128 --depth 10` lifts it toward the 5–15 M budget. The driver **prints the
  measured count** — tune the two knobs if it lands outside the band.

## Caveats (also in docs/JOURNAL.md Entry 11)

- The driver is **untested in the dev env** (no GPU/torch). The `--smoke` path is the
  wiring check; run it before the full job.
- Our L2→L3 event stream is synthesized (cancellation rate is an upper bound), so the
  realism of generations is validated empirically in **P3.3** (stylized-fact suite), not
  assumed.
- Exact parity with TRADES' own `normalize_messages` is approximated; if generations look
  off, the first suspect is the order-feature normalization in `p31_pack_trades.py`.

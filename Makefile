# HFT-SimLab — reproducible entry points.
PY := .venv/bin/python
NPZ ?= data/btcusdt_20260501_0000_0656.npz

.PHONY: ablation backtest-bin

# P4: the sim-to-real ablation harness (strategy x realism-config x seed -> PnL waterfall).
ablation: backtest-bin
	$(PY) experiments/ablation/run_ablation.py --npz $(NPZ)

backtest-bin:
	cd core && cargo build --release --bin backtest

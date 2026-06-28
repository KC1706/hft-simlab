# Reference repositories

These five repos are **read-only references** — we copy patterns/code *out* of them
(with attribution comments in our source), never edit them. They are **not** committed
to this repository (each carries its own git history); reproduce them with:

```bash
cd refs
git clone --depth 1 https://github.com/nkaz001/hftbacktest          hftbacktest
git clone --depth 1 https://github.com/LeonardoBerti00/DeepMarket    DeepMarket
git clone --depth 1 https://github.com/jpmorganchase/abides-jpmc-public abides-jpmc-public
git clone --depth 1 https://github.com/matteoprata/LOBCAST           LOBCAST
git clone --depth 1 https://github.com/objectcomputing/liquibook     liquibook
```

| Repo | What we take from it |
|------|----------------------|
| `hftbacktest` | **Base.** Rust backtesting core: L2/L3 book reconstruction, queue/latency models, exchange processors, Tardis/Binance data tooling |
| `DeepMarket` | Official TRADES code: diffusion-transformer LOB generator (Phase 3) |
| `abides-jpmc-public` | ABIDES agent-based exchange + ABIDES-Gym RL wrapper (Phases 3–4) |
| `LOBCAST` | Deep-learning LOB prediction baselines + preprocessing patterns |
| `liquibook` | Clean C++ matching-engine design reference (Phase 1 warm-up) |

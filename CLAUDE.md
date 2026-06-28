# HFT-SimLab — Session Protocol

Read `PLAN.md` for the full roadmap and current phase. The user is learning quant dev in
depth through this project — teaching is a first-class deliverable, equal to the code.

## Non-negotiable workflow
1. Implement one step from `PLAN.md` at a time.
2. After every **major change**, append an entry to `docs/JOURNAL.md` BEFORE moving on:
   - What was done (files, design decisions).
   - The theory/concepts behind it, explained from first principles (the user owns Bouchaud
     *Trades, Quotes & Prices*, Hasbrouck, Cartea *Algorithmic & HFT*, Harris *Trading &
     Exchanges* — cite specific chapters they can read).
   - Which research paper was used, with arXiv ID, and **how exactly** it informed the code.
3. Give the user the manual test for the step (defined in PLAN.md) and a plain-language
   explanation in chat. **Wait for the user to run it and confirm before the next step.**
4. Journal entries are written in parallel with the work, never batched at the end.
5. **Paper track:** in the same session as each journal entry, draft/extend the matching
   section of `paper/DRAFT.md` (mapping is defined inside it). Journal = teaching voice;
   paper = academic voice. Any paper used in code also gets 1–2 sentences in Related Work.
6. **Figure track:** `paper/FIGURES.md` specifies every planned figure (content, axes,
   caption, and the raw data each phase must LOG for it). When implementing a phase step,
   check FIGURES.md first and make the code emit that figure's data (parquet) as part of the
   step — never re-run experiments later just for plotting. New figure ideas during work get
   a full FIG-spec entry immediately; data figures regenerate from scripts in
   `experiments/figures/`, schematics live in `paper/figs-src/`.

## Engineering rules
- `refs/` is read-only reference material (5 shallow clones). Copy patterns/code out with
  attribution comments pointing to the source file; never modify refs.
- Disk budget: keep `data/` under ~5 GB (zstd-compress, delete raw after conversion).
  Machine: M2, 8 GB RAM — no heavy local GPU training (use Kaggle free tier, Phase 3).
- Budget: $0. Never suggest paid data/services without flagging cost explicitly.
- Rust for the hot path (book, replay, sim), Python (uv-managed) for analysis/calibration.

//! # lob-core — HFT-SimLab Phase 1
//!
//! A minimal L2 (market-by-price) limit order book written from scratch.
//!
//! Scope (PLAN.md P1.1): apply incremental depth updates, maintain best bid/ask,
//! expose depth arrays, and detect locked/crossed books. No matching engine, no
//! order management — an L2 feed gives aggregate quantity per price level, so the
//! book is a *mirror* of the exchange's published state, not a simulation of it.
//!
//! Design notes (full discussion in docs/JOURNAL.md):
//! - Prices are stored as integer ticks (`round(px / tick_size)`), never as f64
//!   map keys. Quantities stay f64 but emptiness is decided in integer lots.
//! - Sides are `BTreeMap<i64, f64>` so depth arrays fall out of ordered iteration.
//!   hftbacktest uses a HashMap + cached best pointers instead; we replicate its
//!   *pointer semantics* exactly (stale-level retention on crossings) so that our
//!   reconstruction matches theirs level-for-level in the P1.2 comparison test.
//!   See refs/hftbacktest/hftbacktest/src/depth/hashmapmarketdepth.rs.

pub mod book;
pub mod events;
pub mod npz;

pub use book::{BookState, L2Book, Level, Side};
pub use npz::NpzEventReader;

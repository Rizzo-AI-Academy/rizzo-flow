//! rizzo-flow-rs — Rizzo Flow in Rust.
//!
//! Port of the [Rizzo Flow](https://github.com/Rizzo-AI-Academy/rizzo-flow) decision pattern
//! (state in → typed decisions out, zero generated tokens), with a llama.cpp/GGUF backend.
//! The original project is © Rizzo AI Academy (Apache-2.0); the Jev pattern is © TypeSafe.
//! This is an independent Rust port — same contract, different engine.
//!
//! F1 scope: the pure decision core (`schema` + `decisions`), ported with the original tests.
//! Backend (`llama` feature), calibration and HTTP serve follow in F1/F2.

pub mod calibration;
pub mod compat;
pub mod decisions;
pub mod evaluate;
pub mod gguf;
pub mod jsonio;
pub mod pinned;
pub mod prompts;
pub mod schema;

#[cfg(feature = "llama")]
pub mod backend;

pub use decisions::{candidates, decode, softmax, Candidate, ABOVE, BELOW, UNKNOWN};
pub use prompts::{compile_request, Compiled, Tokenizer};
pub use schema::{Policy, Question, Request, ValidationError, MAX_SLOTS};

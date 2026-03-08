//! Persistent memory system for agent facts across sessions.

pub mod store;
pub mod types;

pub use store::{format_for_prompt, MemoryStore};
pub use types::{MemoryCategory, MemoryItem};

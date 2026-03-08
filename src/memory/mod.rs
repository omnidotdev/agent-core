//! Persistent memory system for agent facts across sessions.

pub mod store;
pub mod types;

pub use store::{MemoryStore, format_for_prompt};
pub use types::{MemoryCategory, MemoryItem};

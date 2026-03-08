//! Reusable AI agent library for Omni.

pub mod conversation;
pub mod error;
pub mod knowledge;
pub mod permission;
pub mod plan;
pub mod provider;
pub mod providers;
pub mod registry;
pub mod types;

#[cfg(feature = "memory")]
pub mod memory;

#[cfg(feature = "mcp")]
pub mod mcp;

#[cfg(feature = "skills")]
pub mod skills;

#[cfg(feature = "tools")]
pub mod tools;

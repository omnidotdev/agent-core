//! Reusable agent tool implementations.

#[cfg(feature = "browser")]
pub mod browser;
pub mod loop_detection;
pub mod policy;
pub mod shell;
#[cfg(feature = "web")]
pub mod web;

/// Classification for tool execution strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolKind {
    /// Safe to run fully in parallel (read-only operations)
    Read,
    /// Serialized among mutating tools, parallel with reads
    Mutate,
    /// Requires user response before batch continues
    Interactive,
}

impl Default for ToolKind {
    fn default() -> Self {
        Self::Mutate
    }
}

/// Provider of executable tools
///
/// Implement this trait to define a set of tools that can be
/// discovered and executed by a tool executor.
#[async_trait::async_trait]
pub trait ToolProvider: Send + Sync {
    /// Return all tool definitions this provider offers
    fn definitions(&self) -> Vec<crate::types::Tool>;

    /// Execute a named tool with JSON-string arguments
    ///
    /// # Errors
    ///
    /// Returns error if the tool name is unknown or execution fails.
    async fn execute(&self, name: &str, arguments: &str) -> anyhow::Result<String>;

    /// Classify a tool for execution strategy
    fn kind(&self, _name: &str) -> ToolKind {
        ToolKind::Mutate
    }
}

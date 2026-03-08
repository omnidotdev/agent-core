//! Memory store trait for pluggable backends.

use async_trait::async_trait;

use super::types::{MemoryCategory, MemoryItem};

/// Pluggable memory storage backend.
#[async_trait]
pub trait MemoryStore: Send + Sync {
    /// Add a memory item, returning its ID.
    async fn add(&self, item: MemoryItem) -> anyhow::Result<String>;

    /// Get a memory item by ID (updates access tracking).
    async fn get(&self, id: &str) -> anyhow::Result<Option<MemoryItem>>;

    /// List items, optionally filtered by category.
    async fn list(&self, category: Option<MemoryCategory>) -> anyhow::Result<Vec<MemoryItem>>;

    /// Search by text query.
    async fn search(&self, query: &str, limit: Option<usize>) -> anyhow::Result<Vec<MemoryItem>>;

    /// Delete a memory item.
    async fn delete(&self, id: &str) -> anyhow::Result<bool>;

    /// Update content and/or pinned status.
    async fn update(
        &self,
        id: &str,
        content: Option<String>,
        pinned: Option<bool>,
    ) -> anyhow::Result<()>;

    /// Get items for context injection (pinned + recent, up to `max_items`).
    async fn get_context(&self, max_items: usize) -> anyhow::Result<Vec<MemoryItem>>;
}

/// Format memories for system prompt injection.
#[must_use]
pub fn format_for_prompt(items: &[MemoryItem]) -> String {
    if items.is_empty() {
        return String::new();
    }

    use std::fmt::Write;
    let mut output = String::from("<memory>\n");
    output.push_str("The following are facts learned about this project and user:\n\n");

    for item in items {
        let _ = writeln!(output, "- [{}] {}", item.category, item.content);
    }

    output.push_str("</memory>\n");
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_empty() {
        assert!(format_for_prompt(&[]).is_empty());
    }

    #[test]
    fn format_items() {
        let items = vec![
            MemoryItem::new("prefers vim".to_string(), MemoryCategory::Preference),
            MemoryItem::new("uses tokio".to_string(), MemoryCategory::Fact),
        ];
        let output = format_for_prompt(&items);
        assert!(output.contains("<memory>"));
        assert!(output.contains("[preference] prefers vim"));
        assert!(output.contains("[fact] uses tokio"));
        assert!(output.contains("</memory>"));
    }
}

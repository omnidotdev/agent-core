//! Memory item types shared across agent products.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Memory item categories.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryCategory {
    /// User preferences and coding style.
    Preference,
    /// Project/domain-specific facts.
    Fact,
    /// Corrections from user feedback.
    Correction,
    /// General learned information.
    General,
}

impl std::fmt::Display for MemoryCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Preference => write!(f, "preference"),
            Self::Fact => write!(f, "fact"),
            Self::Correction => write!(f, "correction"),
            Self::General => write!(f, "general"),
        }
    }
}

/// A single memory item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryItem {
    /// Unique identifier.
    pub id: String,
    /// The memory content.
    pub content: String,
    /// Category of memory.
    pub category: MemoryCategory,
    /// When this was created.
    pub created_at: DateTime<Utc>,
    /// When this was last accessed.
    pub accessed_at: DateTime<Utc>,
    /// Number of times retrieved.
    pub access_count: u32,
    /// Optional tags for filtering.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Whether this memory is pinned (always included in context).
    #[serde(default)]
    pub pinned: bool,
    /// Optional embedding vector for semantic search.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
    /// Optional owner ID (for multi-user scenarios).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
}

impl MemoryItem {
    /// Create a new memory item.
    #[must_use]
    pub fn new(content: String, category: MemoryCategory) -> Self {
        let now = Utc::now();
        Self {
            id: format!("mem_{}", ulid::Ulid::new()),
            content,
            category,
            created_at: now,
            accessed_at: now,
            access_count: 0,
            tags: Vec::new(),
            pinned: false,
            embedding: None,
            user_id: None,
        }
    }

    /// Add a tag.
    #[must_use]
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Set pinned.
    #[must_use]
    pub const fn pinned(mut self) -> Self {
        self.pinned = true;
        self
    }

    /// Set user ID.
    #[must_use]
    pub fn with_user_id(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    /// Set embedding.
    #[must_use]
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_item_creation() {
        let item = MemoryItem::new("test content".to_string(), MemoryCategory::General);
        assert!(item.id.starts_with("mem_"));
        assert_eq!(item.category, MemoryCategory::General);
        assert!(!item.pinned);
        assert_eq!(item.access_count, 0);
        assert!(item.embedding.is_none());
        assert!(item.user_id.is_none());
    }

    #[test]
    fn memory_item_with_builders() {
        let item = MemoryItem::new("pref".to_string(), MemoryCategory::Preference)
            .with_tag("rust")
            .with_tag("http")
            .pinned()
            .with_user_id("user_123")
            .with_embedding(vec![0.1, 0.2, 0.3]);

        assert_eq!(item.tags, vec!["rust", "http"]);
        assert!(item.pinned);
        assert_eq!(item.user_id.as_deref(), Some("user_123"));
        assert_eq!(item.embedding.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn category_display() {
        assert_eq!(MemoryCategory::Preference.to_string(), "preference");
        assert_eq!(MemoryCategory::Fact.to_string(), "fact");
        assert_eq!(MemoryCategory::Correction.to_string(), "correction");
        assert_eq!(MemoryCategory::General.to_string(), "general");
    }

    #[test]
    fn category_serialization() {
        let json = serde_json::to_string(&MemoryCategory::Preference).unwrap();
        assert_eq!(json, "\"preference\"");

        let deserialized: MemoryCategory = serde_json::from_str("\"fact\"").unwrap();
        assert_eq!(deserialized, MemoryCategory::Fact);
    }
}

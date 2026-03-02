//! Shared knowledge types used by CLI and Beacon

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// When to inject a knowledge chunk
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KnowledgePriority {
    /// Inject every turn (core identity facts)
    Always,
    /// Inject when tags match user message
    #[default]
    Relevant,
}

/// A single knowledge chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeChunk {
    /// Human-readable topic label
    #[serde(default)]
    pub topic: Option<String>,

    /// Machine-readable tags for selection
    #[serde(default)]
    pub tags: Vec<String>,

    /// Freeform knowledge content (markdown)
    pub content: String,

    /// Behavioral rules injected alongside this chunk
    #[serde(default)]
    pub rules: Vec<String>,

    /// Injection priority
    #[serde(default)]
    pub priority: KnowledgePriority,

    /// Optional pre-computed embedding vector
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

/// Knowledge configuration for a persona
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KnowledgeConfig {
    /// Inline knowledge chunks owned by this persona
    #[serde(default)]
    pub inline: Vec<KnowledgeChunk>,

    /// References to external knowledge packs on Manifold
    #[serde(default)]
    pub packs: Vec<KnowledgePackRef>,
}

/// Reference to an external knowledge pack on Manifold
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgePackRef {
    /// Manifold artifact path: `@{namespace}/knowledge/{artifact}`
    #[serde(rename = "ref")]
    pub pack_ref: String,

    /// Semver version constraint
    pub version: Option<String>,

    /// Override priority for all chunks in this pack
    pub priority: Option<KnowledgePriority>,
}

/// Pre-computed embedding vectors for knowledge pack chunks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackEmbeddings {
    /// Embedding model used to generate vectors
    pub model: String,

    /// Dimensionality of each vector
    pub dimensions: usize,

    /// Map from chunk index (as string) to embedding vector
    pub vectors: HashMap<String, Vec<f32>>,
}

/// A knowledge pack published to Manifold
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgePack {
    /// Schema URL
    #[serde(rename = "$schema", default, skip_serializing_if = "Option::is_none")]
    pub schema: Option<String>,

    /// Semver version
    pub version: String,

    /// Display name
    pub name: String,

    /// Description
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Pack-level tags (for marketplace search)
    #[serde(default)]
    pub tags: Vec<String>,

    /// Knowledge chunks
    pub chunks: Vec<KnowledgeChunk>,

    /// Pre-computed embeddings for chunks
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embeddings: Option<PackEmbeddings>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knowledge_priority_defaults_to_relevant() {
        let priority = KnowledgePriority::default();
        assert_eq!(priority, KnowledgePriority::Relevant);
    }

    #[test]
    fn knowledge_config_defaults_empty() {
        let config = KnowledgeConfig::default();
        assert!(config.inline.is_empty());
        assert!(config.packs.is_empty());
    }

    #[test]
    fn chunk_deserializes_from_toml_snake_case() {
        let toml_str = r#"
            topic = "Omni Basics"
            tags = ["omni"]
            content = "Omni is an ecosystem"
            priority = "always"
        "#;

        let chunk: KnowledgeChunk = toml::from_str(toml_str).unwrap();
        assert_eq!(chunk.topic.as_deref(), Some("Omni Basics"));
        assert_eq!(chunk.priority, KnowledgePriority::Always);
        assert!(chunk.embedding.is_none());
    }

    #[test]
    fn chunk_deserializes_from_json() {
        let json_str = r#"{
            "topic": "Crypto Basics",
            "tags": ["crypto", "token"],
            "content": "MCG is on Solana",
            "rules": ["Always cite mint address"],
            "priority": "relevant",
            "embedding": [1.0, 0.0, 0.5]
        }"#;

        let chunk: KnowledgeChunk = serde_json::from_str(json_str).unwrap();
        assert_eq!(chunk.topic.as_deref(), Some("Crypto Basics"));
        assert_eq!(chunk.priority, KnowledgePriority::Relevant);
        assert_eq!(chunk.embedding.unwrap(), vec![1.0, 0.0, 0.5]);
    }

    #[test]
    fn pack_ref_uses_ref_key() {
        let json_str = r#"{
            "ref": "@omni/knowledge/crypto-basics",
            "version": "1.0.0"
        }"#;

        let pack_ref: KnowledgePackRef = serde_json::from_str(json_str).unwrap();
        assert_eq!(pack_ref.pack_ref, "@omni/knowledge/crypto-basics");
        assert_eq!(pack_ref.version.as_deref(), Some("1.0.0"));
    }

    #[test]
    fn pack_with_embeddings_deserializes() {
        let json_str = r#"{
            "version": "1.0.0",
            "name": "test-pack",
            "tags": [],
            "chunks": [
                { "content": "Hello" }
            ],
            "embeddings": {
                "model": "text-embedding-3-small",
                "dimensions": 3,
                "vectors": { "0": [1.0, 0.0, 0.0] }
            }
        }"#;

        let pack: KnowledgePack = serde_json::from_str(json_str).unwrap();
        assert!(pack.embeddings.is_some());
        let emb = pack.embeddings.unwrap();
        assert_eq!(emb.model, "text-embedding-3-small");
        assert_eq!(emb.vectors.get("0").unwrap(), &vec![1.0, 0.0, 0.0]);
    }
}

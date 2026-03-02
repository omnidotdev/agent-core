//! Knowledge management for persona context
//!
//! Provides shared types, Manifold pack resolution, and per-turn
//! chunk selection used by both the CLI and Beacon gateway

mod models;
mod resolver;
mod selection;

pub use models::{
    KnowledgeChunk, KnowledgeConfig, KnowledgePack, KnowledgePackRef, KnowledgePriority,
    PackEmbeddings,
};
pub use resolver::{KnowledgePackResolver, ResolverError, hydrate_embeddings, resolve_and_merge};
pub use selection::{
    build_knowledge_context, cosine_similarity, format_knowledge, select_knowledge,
    select_knowledge_with_embeddings,
};

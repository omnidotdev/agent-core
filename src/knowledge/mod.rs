//! Knowledge management for persona context
//!
//! Provides shared types, Manifold pack resolution, per-turn
//! chunk selection, and text embedding used by both the CLI and
//! Beacon gateway

pub mod bm25;
pub mod condenser;
pub mod embedder;
mod models;
pub mod reranker;
mod resolver;
mod selection;

pub use condenser::{CondenseError, LlmCondenser, QueryCondenser, build_retrieval_query_condensed};
pub use embedder::{EMBEDDING_DIM, Embedder, EmbedderError, contextual_text};
pub use models::{
    KnowledgeChunk, KnowledgeConfig, KnowledgePack, KnowledgePackRef, KnowledgePriority,
    PackEmbeddings,
};
pub use bm25::Bm25Scorer;
pub use reranker::{ApiReranker, Reranker, RerankerError, select_knowledge_reranked};
pub use resolver::{KnowledgePackResolver, ResolverError, hydrate_embeddings, resolve_and_merge};
pub use selection::{
    build_knowledge_context, build_retrieval_query, cosine_similarity, format_knowledge,
    select_knowledge, select_knowledge_with_embeddings,
};

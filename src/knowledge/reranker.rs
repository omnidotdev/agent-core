//! Cross-encoder reranking for knowledge retrieval
//!
//! Reranks candidate chunks using a cross-encoder model that jointly
//! encodes query + document pairs for higher accuracy than bi-encoder
//! (embedding) retrieval alone

use async_trait::async_trait;
use thiserror::Error;

/// Over-fetch multiplier: retrieve 3x candidates then rerank to the actual budget
const OVERFETCH_MULTIPLIER: usize = 3;

/// Errors from the reranking API
#[derive(Debug, Error)]
pub enum RerankerError {
    /// Reranking request failed
    #[error("reranking request failed: {0}")]
    Request(String),
}

/// Rerank candidate chunks for a given query
#[async_trait]
pub trait Reranker: Send + Sync {
    /// Score query-document pairs, returning relevance scores in [0, 1]
    ///
    /// The returned vector has the same length as `documents`, with each
    /// entry being the relevance score for the corresponding document
    async fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>, RerankerError>;
}

/// API-backed cross-encoder reranker
///
/// Calls an external reranking endpoint (Cohere, Jina, or a
/// Synapse-routed service) with query + document pairs
pub struct ApiReranker {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
}

impl ApiReranker {
    /// Create a reranker using the Cohere rerank API
    ///
    /// # Errors
    ///
    /// Returns error if API key is empty
    pub fn cohere(api_key: String) -> Result<Self, RerankerError> {
        if api_key.is_empty() {
            return Err(RerankerError::Request(
                "API key required for reranking".to_string(),
            ));
        }

        Ok(Self {
            client: reqwest::Client::new(),
            api_key,
            model: "rerank-v3.5".to_string(),
            base_url: "https://api.cohere.com/v2/rerank".to_string(),
        })
    }

    /// Create a reranker with a custom model and endpoint
    pub fn with_config(
        api_key: String,
        model: String,
        base_url: String,
    ) -> Result<Self, RerankerError> {
        if api_key.is_empty() {
            return Err(RerankerError::Request(
                "API key required for reranking".to_string(),
            ));
        }

        Ok(Self {
            client: reqwest::Client::new(),
            api_key,
            model,
            base_url,
        })
    }

    /// Return the over-fetch multiplier used during reranking
    #[must_use]
    pub const fn overfetch_multiplier() -> usize {
        OVERFETCH_MULTIPLIER
    }
}

#[async_trait]
impl Reranker for ApiReranker {
    async fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>, RerankerError> {
        #[derive(serde::Serialize)]
        struct RerankRequest<'a> {
            model: &'a str,
            query: &'a str,
            documents: &'a [&'a str],
            top_n: usize,
        }

        #[derive(serde::Deserialize)]
        struct RerankResponse {
            results: Vec<RerankResult>,
        }

        #[derive(serde::Deserialize)]
        struct RerankResult {
            index: usize,
            relevance_score: f32,
        }

        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let request = RerankRequest {
            model: &self.model,
            query,
            documents,
            top_n: documents.len(),
        };

        let response = self
            .client
            .post(&self.base_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await
            .map_err(|e| RerankerError::Request(e.to_string()))?;

        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(RerankerError::Request(format!("API error: {body}")));
        }

        let result: RerankResponse = response
            .json()
            .await
            .map_err(|e| RerankerError::Request(e.to_string()))?;

        // Build scores array indexed by original document position
        let mut scores = vec![0.0_f32; documents.len()];
        for r in result.results {
            if r.index < scores.len() {
                scores[r.index] = r.relevance_score;
            }
        }

        Ok(scores)
    }
}

/// Select knowledge with optional cross-encoder reranking
///
/// Over-fetches 3x candidates from the fast path (BM25 + embedding),
/// then reranks the top candidates with a cross-encoder for higher accuracy
pub async fn select_knowledge_reranked<'a>(
    chunks: &'a [super::models::KnowledgeChunk],
    user_message: &str,
    user_embedding: Option<&[f32]>,
    reranker: &dyn Reranker,
    max_tokens: usize,
) -> Vec<&'a super::models::KnowledgeChunk> {
    use super::models::KnowledgePriority;

    // Over-fetch candidates
    let overfetch_budget = max_tokens * OVERFETCH_MULTIPLIER;
    let candidates = super::selection::select_knowledge_with_embeddings(
        chunks,
        user_message,
        user_embedding,
        overfetch_budget,
    );

    if candidates.is_empty() {
        return candidates;
    }

    // Separate always-priority (don't rerank those) from relevant
    let mut always_chunks: Vec<&super::models::KnowledgeChunk> = Vec::new();
    let mut rerank_candidates: Vec<&super::models::KnowledgeChunk> = Vec::new();

    for chunk in &candidates {
        if chunk.priority == KnowledgePriority::Always {
            always_chunks.push(chunk);
        } else {
            rerank_candidates.push(chunk);
        }
    }

    if rerank_candidates.is_empty() {
        return always_chunks;
    }

    // Build document strings for the reranker
    let docs: Vec<&str> = rerank_candidates
        .iter()
        .map(|c| c.content.as_str())
        .collect();

    match reranker.rerank(user_message, &docs).await {
        Ok(relevance) => {
            // Pair with scores and sort descending
            let mut ranked: Vec<(usize, f32)> = relevance.into_iter().enumerate().collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Rebuild result: always chunks first, then reranked
            let mut result = always_chunks;
            for (idx, _score) in &ranked {
                result.push(rerank_candidates[*idx]);
            }

            // Trim to actual budget
            super::selection::trim_to_budget_pub(&mut result, max_tokens);
            result
        }
        Err(e) => {
            tracing::warn!(error = %e, "reranking failed, using fast-path results");
            // Fall back: trim the over-fetched candidates to the real budget
            let mut result = candidates;
            super::selection::trim_to_budget_pub(&mut result, max_tokens);
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockReranker {
        scores: Vec<f32>,
    }

    #[async_trait]
    impl Reranker for MockReranker {
        async fn rerank(
            &self,
            _query: &str,
            _documents: &[&str],
        ) -> Result<Vec<f32>, RerankerError> {
            Ok(self.scores.clone())
        }
    }

    struct FailingReranker;

    #[async_trait]
    impl Reranker for FailingReranker {
        async fn rerank(
            &self,
            _query: &str,
            _documents: &[&str],
        ) -> Result<Vec<f32>, RerankerError> {
            Err(RerankerError::Request("mock failure".to_string()))
        }
    }

    #[tokio::test]
    async fn reranked_selection_reorders_by_score() {
        use crate::knowledge::models::{KnowledgeChunk, KnowledgePriority};

        let chunks = vec![
            KnowledgeChunk {
                topic: Some("Low BM25".to_string()),
                tags: vec![],
                content: "MCG token".to_string(),
                rules: vec![],
                priority: KnowledgePriority::Relevant,
                embedding: None,
            },
            KnowledgeChunk {
                topic: Some("High Rerank".to_string()),
                tags: vec![],
                content: "MCG Solana blockchain token".to_string(),
                rules: vec![],
                priority: KnowledgePriority::Relevant,
                embedding: None,
            },
        ];

        // Reranker scores: second doc higher
        let reranker = MockReranker {
            scores: vec![0.3, 0.9],
        };

        let selected =
            select_knowledge_reranked(&chunks, "MCG token", None, &reranker, 10000).await;

        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].topic.as_deref(), Some("High Rerank"));
    }

    #[tokio::test]
    async fn reranked_selection_preserves_always_chunks() {
        use crate::knowledge::models::{KnowledgeChunk, KnowledgePriority};

        let chunks = vec![
            KnowledgeChunk {
                topic: Some("Core".to_string()),
                tags: vec![],
                content: "Core identity".to_string(),
                rules: vec![],
                priority: KnowledgePriority::Always,
                embedding: None,
            },
            KnowledgeChunk {
                topic: Some("Info".to_string()),
                tags: vec![],
                content: "MCG token info".to_string(),
                rules: vec![],
                priority: KnowledgePriority::Relevant,
                embedding: None,
            },
        ];

        let reranker = MockReranker { scores: vec![0.5] };

        let selected =
            select_knowledge_reranked(&chunks, "MCG token", None, &reranker, 10000).await;

        assert_eq!(selected[0].topic.as_deref(), Some("Core"));
    }

    #[tokio::test]
    async fn reranked_selection_falls_back_on_error() {
        use crate::knowledge::models::{KnowledgeChunk, KnowledgePriority};

        let chunks = vec![KnowledgeChunk {
            topic: Some("Token".to_string()),
            tags: vec![],
            content: "MCG token details".to_string(),
            rules: vec![],
            priority: KnowledgePriority::Relevant,
            embedding: None,
        }];

        let reranker = FailingReranker;

        let selected =
            select_knowledge_reranked(&chunks, "MCG token", None, &reranker, 10000).await;

        // Should still return results from the fast path
        assert!(!selected.is_empty());
    }

    #[test]
    fn empty_api_key_rejected() {
        let result = ApiReranker::cohere(String::new());
        assert!(result.is_err());
    }
}

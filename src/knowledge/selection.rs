//! Knowledge selection and injection for persona context

use std::fmt::Write;

use super::bm25::Bm25Scorer;
use super::models::{KnowledgeChunk, KnowledgePriority};

/// Default knowledge token budget (rough estimate: 4 chars per token)
const DEFAULT_KNOWLEDGE_TOKEN_BUDGET: usize = 4000;

/// Default number of recent turns to include in retrieval queries
const DEFAULT_MAX_RETRIEVAL_TURNS: usize = 3;

/// Minimum similarity threshold for embedding-based selection
const MIN_SIMILARITY: f32 = 0.2;

/// RRF smoothing constant (standard value from the original paper)
const RRF_K: f32 = 60.0;

/// Build a retrieval query from recent user messages
///
/// Concatenates the last `max_turns` user messages (current message last)
/// to provide conversational context for knowledge selection. Falls back to
/// `current_message` alone when no history is available.
///
/// # Examples
///
/// ```
/// use agent_core::knowledge::build_retrieval_query;
///
/// let query = build_retrieval_query("tell me more", &["what is MCG?"], 3);
/// assert!(query.contains("what is MCG?"));
/// assert!(query.contains("tell me more"));
/// ```
#[must_use]
pub fn build_retrieval_query(current_message: &str, history: &[&str], max_turns: usize) -> String {
    let max_turns = if max_turns == 0 {
        DEFAULT_MAX_RETRIEVAL_TURNS
    } else {
        max_turns
    };

    if history.is_empty() {
        return current_message.to_string();
    }

    // Take up to max_turns - 1 from history (newest first in input),
    // reverse so oldest context comes first, then append current message
    let prior: Vec<&str> = history.iter().take(max_turns - 1).rev().copied().collect();

    let mut parts = prior;
    parts.push(current_message);
    parts.join("\n")
}

/// Select relevant knowledge chunks based on user message
///
/// Delegates to `select_knowledge_with_embeddings` with no user embedding,
/// preserving tag-matching behavior
#[must_use]
pub fn select_knowledge<'a>(
    chunks: &'a [KnowledgeChunk],
    user_message: &str,
    max_tokens: usize,
) -> Vec<&'a KnowledgeChunk> {
    select_knowledge_with_embeddings(chunks, user_message, None, max_tokens)
}

/// Select relevant knowledge chunks using BM25 + embedding hybrid search
///
/// Selection strategy:
/// 1. All chunks with priority "always" are included unconditionally
/// 2. For "relevant" chunks, uses reciprocal rank fusion (RRF) to combine:
///    - BM25 term-frequency scores (always available)
///    - Cosine similarity scores (when embeddings are available)
/// 3. Trim to token budget
#[must_use]
pub fn select_knowledge_with_embeddings<'a>(
    chunks: &'a [KnowledgeChunk],
    user_message: &str,
    user_embedding: Option<&[f32]>,
    max_tokens: usize,
) -> Vec<&'a KnowledgeChunk> {
    let mut selected: Vec<&KnowledgeChunk> = Vec::new();

    // Always-priority chunks first
    let mut relevant_chunks: Vec<(usize, &KnowledgeChunk)> = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        if chunk.priority == KnowledgePriority::Always {
            selected.push(chunk);
        } else if chunk.priority == KnowledgePriority::Relevant {
            relevant_chunks.push((i, chunk));
        }
    }

    if relevant_chunks.is_empty() || user_message.is_empty() {
        trim_to_budget(&mut selected, max_tokens);
        return selected;
    }

    // Build BM25 scorer over relevant chunks only
    let relevant_only: Vec<&KnowledgeChunk> = relevant_chunks.iter().map(|(_, c)| *c).collect();
    let relevant_chunk_data: Vec<KnowledgeChunk> =
        relevant_only.iter().map(|c| (*c).clone()).collect();
    let scorer = Bm25Scorer::new(&relevant_chunk_data);
    let bm25_scores = scorer.score(user_message);

    // Build BM25 rank map (index in relevant_chunks → rank)
    let mut bm25_rank: Vec<(usize, usize)> = Vec::new();
    for (rank, (idx, _score)) in bm25_scores.iter().enumerate() {
        bm25_rank.push((*idx, rank + 1));
    }
    let bm25_rank_map: std::collections::HashMap<usize, usize> = bm25_rank.into_iter().collect();

    // Build embedding rank map if user embedding is available
    let emb_rank_map: std::collections::HashMap<usize, usize> =
        if let Some(user_emb) = user_embedding {
            let mut emb_ranked: Vec<(usize, f32)> = relevant_chunks
                .iter()
                .enumerate()
                .filter_map(|(local_idx, (_, chunk))| {
                    chunk
                        .embedding
                        .as_ref()
                        .map(|emb| (local_idx, cosine_similarity(emb, user_emb)))
                })
                .filter(|(_, score)| *score >= MIN_SIMILARITY)
                .collect();
            emb_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            emb_ranked
                .into_iter()
                .enumerate()
                .map(|(rank, (idx, _))| (idx, rank + 1))
                .collect()
        } else {
            std::collections::HashMap::new()
        };

    // Compute RRF fusion scores
    #[allow(clippy::cast_precision_loss)]
    let mut fused: Vec<(usize, f32)> = relevant_chunks
        .iter()
        .enumerate()
        .filter_map(|(local_idx, _)| {
            let bm25_component = bm25_rank_map
                .get(&local_idx)
                .map(|&r| 1.0 / (RRF_K + r as f32));
            let emb_component = emb_rank_map
                .get(&local_idx)
                .map(|&r| 1.0 / (RRF_K + r as f32));

            // Must appear in at least one ranking
            match (bm25_component, emb_component) {
                (Some(b), Some(e)) => Some((local_idx, b + e)),
                (Some(b), None) => Some((local_idx, b)),
                (None, Some(e)) => Some((local_idx, e)),
                (None, None) => None,
            }
        })
        .collect();

    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Append fused results after always chunks
    for (local_idx, _score) in &fused {
        selected.push(relevant_chunks[*local_idx].1);
    }

    // Trim to token budget
    trim_to_budget(&mut selected, max_tokens);

    selected
}

/// Compute cosine similarity between two vectors
///
/// Returns 0.0 if either vector has zero magnitude or the lengths differ
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        return 0.0;
    }

    dot / denom
}

/// Format selected knowledge chunks as markdown for prompt injection
#[must_use]
pub fn format_knowledge(chunks: &[&KnowledgeChunk]) -> String {
    if chunks.is_empty() {
        return String::new();
    }

    let sections: Vec<String> = chunks
        .iter()
        .map(|chunk| {
            let topic = chunk.topic.as_deref().unwrap_or("Knowledge");
            let mut section = format!("## {topic}\n{}", chunk.content);
            if !chunk.rules.is_empty() {
                section.push_str("\n\nRules:");
                for rule in &chunk.rules {
                    let _ = write!(section, "\n- {rule}");
                }
            }
            section
        })
        .collect();

    sections.join("\n\n")
}

/// Build a knowledge context block for system prompt injection
///
/// Selects relevant chunks based on the user message and wraps the
/// formatted result in `<knowledge>...</knowledge>` XML tags.
/// Returns an empty string if no chunks match
#[must_use]
pub fn build_knowledge_context(chunks: &[KnowledgeChunk], user_message: &str) -> String {
    let selected = select_knowledge(chunks, user_message, DEFAULT_KNOWLEDGE_TOKEN_BUDGET);
    let formatted = format_knowledge(&selected);

    if formatted.is_empty() {
        return String::new();
    }

    format!("<knowledge>\n{formatted}\n</knowledge>")
}

/// Rough token estimation (4 chars per token)
const fn estimate_tokens(text: &str) -> usize {
    text.len() / 4
}

/// Trim chunks to fit within a token budget (public for cross-module use)
pub(super) fn trim_to_budget_pub(chunks: &mut Vec<&KnowledgeChunk>, max_tokens: usize) {
    trim_to_budget(chunks, max_tokens);
}

/// Trim chunks to fit within a token budget
fn trim_to_budget(chunks: &mut Vec<&KnowledgeChunk>, max_tokens: usize) {
    let mut total_tokens = 0;
    let mut keep = 0;

    for chunk in chunks.iter() {
        let topic_str = chunk.topic.as_deref().unwrap_or("");
        let chunk_tokens = estimate_tokens(&chunk.content) + estimate_tokens(topic_str);
        for rule in &chunk.rules {
            total_tokens += estimate_tokens(rule);
        }
        total_tokens += chunk_tokens;

        if total_tokens > max_tokens && keep > 0 {
            break;
        }
        keep += 1;
    }

    chunks.truncate(keep);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chunk(topic: &str, tags: &[&str], priority: KnowledgePriority) -> KnowledgeChunk {
        KnowledgeChunk {
            topic: Some(topic.to_string()),
            tags: tags.iter().map(|t| t.to_string()).collect(),
            content: format!("Content about {topic}"),
            rules: vec![],
            priority,
            embedding: None,
        }
    }

    fn make_embedded_chunk(
        topic: &str,
        embedding: Vec<f32>,
        priority: KnowledgePriority,
    ) -> KnowledgeChunk {
        KnowledgeChunk {
            topic: Some(topic.to_string()),
            tags: vec![],
            content: format!("Content about {topic}"),
            rules: vec![],
            priority,
            embedding: Some(embedding),
        }
    }

    // Tag-based / BM25 selection tests

    #[test]
    fn always_chunks_included() {
        let chunks = vec![
            make_chunk("Token Info", &["token"], KnowledgePriority::Always),
            make_chunk("Platform", &["platform"], KnowledgePriority::Relevant),
        ];

        let selected = select_knowledge(&chunks, "random question", 10000);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].topic.as_deref(), Some("Token Info"));
    }

    #[test]
    fn bm25_selects_relevant_by_content() {
        let chunks = vec![
            KnowledgeChunk {
                topic: Some("Token Info".to_string()),
                tags: vec!["token".to_string(), "mcg".to_string()],
                content: "MCG is a token on Solana".to_string(),
                rules: vec![],
                priority: KnowledgePriority::Relevant,
                embedding: None,
            },
            KnowledgeChunk {
                topic: Some("Platform".to_string()),
                tags: vec!["platform".to_string()],
                content: "Omni is a developer platform".to_string(),
                rules: vec![],
                priority: KnowledgePriority::Relevant,
                embedding: None,
            },
        ];

        let selected = select_knowledge(&chunks, "tell me about the token MCG", 10000);
        assert!(!selected.is_empty());
        assert_eq!(selected[0].topic.as_deref(), Some("Token Info"));
    }

    #[test]
    fn multiple_bm25_matches() {
        let chunks = vec![
            KnowledgeChunk {
                topic: Some("Token".to_string()),
                tags: vec!["token".to_string()],
                content: "MCG token details".to_string(),
                rules: vec![],
                priority: KnowledgePriority::Relevant,
                embedding: None,
            },
            KnowledgeChunk {
                topic: Some("Platform".to_string()),
                tags: vec!["platform".to_string()],
                content: "Omni platform overview".to_string(),
                rules: vec![],
                priority: KnowledgePriority::Relevant,
                embedding: None,
            },
        ];

        let selected = select_knowledge(&chunks, "MCG token Omni platform", 10000);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn always_plus_relevant() {
        let chunks = vec![
            make_chunk("Core", &[], KnowledgePriority::Always),
            KnowledgeChunk {
                topic: Some("Token".to_string()),
                tags: vec!["token".to_string()],
                content: "MCG token information".to_string(),
                rules: vec![],
                priority: KnowledgePriority::Relevant,
                embedding: None,
            },
            make_chunk("Other", &["other"], KnowledgePriority::Relevant),
        ];

        let selected = select_knowledge(&chunks, "what is the MCG token?", 10000);
        assert!(selected.len() >= 2);
        assert_eq!(selected[0].topic.as_deref(), Some("Core"));
        // Token should appear because BM25 matches on "token" and "MCG"
        assert!(selected.iter().any(|c| c.topic.as_deref() == Some("Token")));
    }

    #[test]
    fn no_matches_returns_only_always() {
        let chunks = vec![make_chunk("Token", &["token"], KnowledgePriority::Relevant)];

        // "xyz" doesn't appear in any chunk content or tags
        let selected = select_knowledge(&chunks, "xyz", 10000);
        assert!(selected.is_empty());
    }

    #[test]
    fn token_budget_trimming() {
        let chunks = vec![
            KnowledgeChunk {
                topic: Some("A".to_string()),
                tags: vec!["a".to_string()],
                content: "Short".to_string(),
                rules: vec![],
                priority: KnowledgePriority::Always,
                embedding: None,
            },
            KnowledgeChunk {
                topic: Some("B".to_string()),
                tags: vec!["b".to_string()],
                content: "This is a much longer content string that should push us over the token budget when combined with the first chunk".to_string(),
                rules: vec![],
                priority: KnowledgePriority::Always,
                embedding: None,
            },
        ];

        // Very tight budget - should keep at least the first chunk
        let selected = select_knowledge(&chunks, "", 5);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].topic.as_deref(), Some("A"));
    }

    // Formatting tests

    #[test]
    fn format_knowledge_empty() {
        let formatted = format_knowledge(&[]);
        assert!(formatted.is_empty());
    }

    #[test]
    fn format_knowledge_with_rules() {
        let chunk = KnowledgeChunk {
            topic: Some("Token".to_string()),
            tags: vec![],
            content: "MCG is on Solana".to_string(),
            rules: vec!["Always cite mint address".to_string()],
            priority: KnowledgePriority::Always,
            embedding: None,
        };

        let formatted = format_knowledge(&[&chunk]);
        assert!(formatted.contains("## Token"));
        assert!(formatted.contains("MCG is on Solana"));
        assert!(formatted.contains("Rules:"));
        assert!(formatted.contains("- Always cite mint address"));
    }

    #[test]
    fn build_knowledge_context_wraps_in_xml() {
        let chunks = vec![make_chunk("Core", &[], KnowledgePriority::Always)];

        let context = build_knowledge_context(&chunks, "anything");
        assert!(context.starts_with("<knowledge>"));
        assert!(context.ends_with("</knowledge>"));
        assert!(context.contains("## Core"));
    }

    #[test]
    fn build_knowledge_context_empty_when_no_matches() {
        let chunks = vec![KnowledgeChunk {
            topic: Some("Token".to_string()),
            tags: vec!["token".to_string()],
            content: "Unique content xyz".to_string(),
            rules: vec![],
            priority: KnowledgePriority::Relevant,
            embedding: None,
        }];

        let context = build_knowledge_context(&chunks, "unrelated message");
        assert!(context.is_empty());
    }

    // Cosine similarity tests

    #[test]
    fn cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < f32::EPSILON);
    }

    #[test]
    fn cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn cosine_similarity_mismatched_lengths() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < f32::EPSILON);
    }

    #[test]
    fn cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < f32::EPSILON);
    }

    // Embedding-aware selection tests

    #[test]
    fn embedding_selection_ranks_by_similarity() {
        let chunks = vec![
            make_embedded_chunk("Far", vec![0.0, 1.0, 0.0], KnowledgePriority::Relevant),
            make_embedded_chunk("Close", vec![1.0, 0.1, 0.0], KnowledgePriority::Relevant),
        ];

        let user_emb = vec![1.0, 0.0, 0.0];
        let selected =
            select_knowledge_with_embeddings(&chunks, "anything", Some(&user_emb), 10000);

        // "Close" should rank higher via embedding similarity
        assert!(!selected.is_empty());
        assert_eq!(selected[0].topic.as_deref(), Some("Close"));
    }

    #[test]
    fn embedding_selection_includes_always_chunks() {
        let chunks = vec![
            make_embedded_chunk("Core", vec![0.0, 0.0, 1.0], KnowledgePriority::Always),
            make_embedded_chunk("Far", vec![0.0, 1.0, 0.0], KnowledgePriority::Relevant),
        ];

        let user_emb = vec![1.0, 0.0, 0.0];
        let selected =
            select_knowledge_with_embeddings(&chunks, "anything", Some(&user_emb), 10000);

        // Always chunk included even though embedding is distant
        assert!(!selected.is_empty());
        assert_eq!(selected[0].topic.as_deref(), Some("Core"));
    }

    #[test]
    fn embedding_selection_falls_back_to_bm25() {
        // Chunk without embedding still gets picked up by BM25
        let chunks = vec![KnowledgeChunk {
            topic: Some("Token".to_string()),
            tags: vec!["token".to_string()],
            content: "MCG token on Solana".to_string(),
            rules: vec![],
            priority: KnowledgePriority::Relevant,
            embedding: None,
        }];

        let user_emb = vec![1.0, 0.0, 0.0];
        let selected = select_knowledge_with_embeddings(
            &chunks,
            "tell me about the MCG token",
            Some(&user_emb),
            10000,
        );

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].topic.as_deref(), Some("Token"));
    }

    #[test]
    fn no_user_embedding_uses_bm25_only() {
        let chunks = vec![
            make_embedded_chunk("Crypto", vec![1.0, 0.0, 0.0], KnowledgePriority::Relevant),
            KnowledgeChunk {
                topic: Some("Token".to_string()),
                tags: vec!["token".to_string()],
                content: "MCG token details".to_string(),
                rules: vec![],
                priority: KnowledgePriority::Relevant,
                embedding: None,
            },
        ];

        // No user embedding: BM25 alone drives selection
        let selected =
            select_knowledge_with_embeddings(&chunks, "tell me about the MCG token", None, 10000);

        assert!(!selected.is_empty());
        assert_eq!(selected[0].topic.as_deref(), Some("Token"));
    }

    // Retrieval query tests

    #[test]
    fn retrieval_query_empty_history() {
        let query = build_retrieval_query("what is MCG?", &[], 3);
        assert_eq!(query, "what is MCG?");
    }

    #[test]
    fn retrieval_query_single_prior() {
        let query = build_retrieval_query("tell me more", &["what is MCG?"], 3);
        assert_eq!(query, "what is MCG?\ntell me more");
    }

    #[test]
    fn retrieval_query_multi_turn_truncation() {
        // history is newest-first: ["c", "b", "a"]
        // max_turns = 3 means take 2 from history + current
        let query = build_retrieval_query("d", &["c", "b", "a"], 3);
        // Should include b, c (reversed to chronological), then d
        assert_eq!(query, "b\nc\nd");
    }

    #[test]
    fn retrieval_query_fewer_turns_than_max() {
        let query = build_retrieval_query("current", &["prev"], 5);
        assert_eq!(query, "prev\ncurrent");
    }

    #[test]
    fn hybrid_rrf_combines_signals() {
        // Chunk that matches both BM25 and embedding should rank highest
        let chunks = vec![
            KnowledgeChunk {
                topic: Some("BM25 Only".to_string()),
                tags: vec![],
                content: "MCG token Solana blockchain".to_string(),
                rules: vec![],
                priority: KnowledgePriority::Relevant,
                embedding: None, // No embedding
            },
            KnowledgeChunk {
                topic: Some("Both Signals".to_string()),
                tags: vec![],
                content: "MCG token details and pricing".to_string(),
                rules: vec![],
                priority: KnowledgePriority::Relevant,
                embedding: Some(vec![1.0, 0.0, 0.0]), // Close to user embedding
            },
            KnowledgeChunk {
                topic: Some("Embedding Only".to_string()),
                tags: vec![],
                content: "Unrelated text about nothing".to_string(),
                rules: vec![],
                priority: KnowledgePriority::Relevant,
                embedding: Some(vec![0.9, 0.1, 0.0]), // Close to user embedding
            },
        ];

        let user_emb = vec![1.0, 0.0, 0.0];
        let selected =
            select_knowledge_with_embeddings(&chunks, "MCG token", Some(&user_emb), 10000);

        // "Both Signals" should rank first because it scores in both BM25 and embedding
        assert!(!selected.is_empty());
        assert_eq!(selected[0].topic.as_deref(), Some("Both Signals"));
    }
}

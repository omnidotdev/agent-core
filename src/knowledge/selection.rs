//! Knowledge selection and injection for persona context

use std::fmt::Write;

use super::models::{KnowledgeChunk, KnowledgePriority};

/// Default knowledge token budget (rough estimate: 4 chars per token)
const DEFAULT_KNOWLEDGE_TOKEN_BUDGET: usize = 4000;

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

/// Select relevant knowledge chunks with optional embedding-based ranking
///
/// Selection strategy:
/// 1. All chunks with priority "always" are included unconditionally
/// 2. For "relevant" chunks:
///    - If a chunk has an embedding AND `user_embedding` is provided,
///      rank by cosine similarity (highest first)
///    - Otherwise fall back to tag matching against the user message
/// 3. Trim to token budget
#[must_use]
pub fn select_knowledge_with_embeddings<'a>(
    chunks: &'a [KnowledgeChunk],
    user_message: &str,
    user_embedding: Option<&[f32]>,
    max_tokens: usize,
) -> Vec<&'a KnowledgeChunk> {
    const MIN_SIMILARITY: f32 = 0.2;

    let mut selected: Vec<&KnowledgeChunk> = Vec::new();

    // Always-priority chunks first
    for chunk in chunks {
        if chunk.priority == KnowledgePriority::Always {
            selected.push(chunk);
        }
    }

    // Collect relevant chunks that can be scored by embedding similarity
    let mut scored: Vec<(&KnowledgeChunk, f32)> = Vec::new();
    let mut unscored_relevant: Vec<&KnowledgeChunk> = Vec::new();

    // Strip punctuation and split into clean tokens for tag fallback
    let message_lower = user_message.to_lowercase();
    let tokens: Vec<String> = message_lower
        .split_whitespace()
        .map(|t| t.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|t| !t.is_empty())
        .collect();

    for chunk in chunks {
        if chunk.priority != KnowledgePriority::Relevant {
            continue;
        }

        if let (Some(chunk_emb), Some(user_emb)) = (&chunk.embedding, user_embedding) {
            // Both embeddings available: compute similarity score
            let score = cosine_similarity(chunk_emb, user_emb);
            scored.push((chunk, score));
        } else {
            // Fall back to tag matching
            let matched = chunk.tags.iter().any(|tag| {
                let tag_lower = tag.to_lowercase();
                tokens.contains(&tag_lower)
            });
            if matched {
                unscored_relevant.push(chunk);
            }
        }
    }

    // Sort scored chunks by similarity (descending)
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Filter scored chunks above minimum relevance threshold
    for (chunk, score) in &scored {
        if *score >= MIN_SIMILARITY {
            selected.push(chunk);
        }
    }

    // Append tag-matched chunks after embedding-ranked ones
    selected.extend(unscored_relevant);

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

    // Tag-based selection tests

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
    fn tag_matching_selects_relevant() {
        let chunks = vec![
            make_chunk("Token Info", &["token", "mcg"], KnowledgePriority::Relevant),
            make_chunk("Platform", &["platform"], KnowledgePriority::Relevant),
        ];

        let selected = select_knowledge(&chunks, "tell me about the token", 10000);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].topic.as_deref(), Some("Token Info"));
    }

    #[test]
    fn multiple_tag_matches() {
        let chunks = vec![
            make_chunk("Token", &["token"], KnowledgePriority::Relevant),
            make_chunk("Platform", &["platform"], KnowledgePriority::Relevant),
        ];

        let selected = select_knowledge(&chunks, "token and platform", 10000);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn always_plus_relevant() {
        let chunks = vec![
            make_chunk("Core", &[], KnowledgePriority::Always),
            make_chunk("Token", &["token"], KnowledgePriority::Relevant),
            make_chunk("Other", &["other"], KnowledgePriority::Relevant),
        ];

        let selected = select_knowledge(&chunks, "what is the token?", 10000);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].topic.as_deref(), Some("Core"));
        assert_eq!(selected[1].topic.as_deref(), Some("Token"));
    }

    #[test]
    fn no_matches_returns_empty() {
        let chunks = vec![make_chunk("Token", &["token"], KnowledgePriority::Relevant)];

        let selected = select_knowledge(&chunks, "hello world", 10000);
        assert!(selected.is_empty());
    }

    #[test]
    fn tag_matching_strips_punctuation() {
        let chunks = vec![make_chunk("Token", &["mcg"], KnowledgePriority::Relevant)];

        let selected = select_knowledge(&chunks, "what is $mcg?", 10000);
        assert_eq!(selected.len(), 1);
    }

    #[test]
    fn case_insensitive_matching() {
        let chunks = vec![make_chunk("Token", &["MCG"], KnowledgePriority::Relevant)];

        let selected = select_knowledge(&chunks, "what is mcg?", 10000);
        assert_eq!(selected.len(), 1);
    }

    #[test]
    fn short_tags_no_false_positives() {
        let chunks = vec![make_chunk(
            "AR Platform",
            &["ar"],
            KnowledgePriority::Relevant,
        )];

        // "are" should NOT match tag "ar"
        let selected = select_knowledge(&chunks, "what are you?", 10000);
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
        let chunks = vec![make_chunk("Token", &["token"], KnowledgePriority::Relevant)];

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
        let selected = select_knowledge_with_embeddings(&chunks, "", Some(&user_emb), 10000);

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].topic.as_deref(), Some("Close"));
    }

    #[test]
    fn embedding_selection_includes_always_chunks() {
        let chunks = vec![
            make_embedded_chunk("Core", vec![0.0, 0.0, 1.0], KnowledgePriority::Always),
            make_embedded_chunk("Far", vec![0.0, 1.0, 0.0], KnowledgePriority::Relevant),
        ];

        let user_emb = vec![1.0, 0.0, 0.0];
        let selected = select_knowledge_with_embeddings(&chunks, "", Some(&user_emb), 10000);

        // Always chunk included even though embedding is distant
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].topic.as_deref(), Some("Core"));
    }

    #[test]
    fn embedding_selection_falls_back_to_tags() {
        // Chunk without embedding falls back to tag matching
        let chunks = vec![make_chunk("Token", &["token"], KnowledgePriority::Relevant)];

        let user_emb = vec![1.0, 0.0, 0.0];
        let selected = select_knowledge_with_embeddings(
            &chunks,
            "tell me about the token",
            Some(&user_emb),
            10000,
        );

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].topic.as_deref(), Some("Token"));
    }

    #[test]
    fn embedding_selection_no_user_embedding_uses_tags() {
        let chunks = vec![
            make_embedded_chunk("Crypto", vec![1.0, 0.0, 0.0], KnowledgePriority::Relevant),
            make_chunk("Token", &["token"], KnowledgePriority::Relevant),
        ];

        // No user embedding: embedded chunk treated as unscored, falls back to tags
        let selected =
            select_knowledge_with_embeddings(&chunks, "tell me about the token", None, 10000);

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].topic.as_deref(), Some("Token"));
    }

    #[test]
    fn embedding_selection_filters_low_similarity() {
        let chunks = vec![make_embedded_chunk(
            "Unrelated",
            vec![0.0, 1.0, 0.0],
            KnowledgePriority::Relevant,
        )];

        // Orthogonal vectors should be filtered out (similarity = 0 < 0.2 threshold)
        let user_emb = vec![1.0, 0.0, 0.0];
        let selected = select_knowledge_with_embeddings(&chunks, "", Some(&user_emb), 10000);

        assert!(selected.is_empty());
    }
}

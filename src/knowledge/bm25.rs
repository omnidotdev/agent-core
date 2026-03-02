//! BM25 relevance scorer for knowledge chunks
//!
//! Implements Okapi BM25 (k1=1.2, b=0.75) for term-frequency scoring
//! over a small in-memory corpus of knowledge chunks

use std::collections::HashMap;

use super::models::KnowledgeChunk;

/// BM25 tuning parameter: term frequency saturation
const K1: f32 = 1.2;

/// BM25 tuning parameter: document length normalization
const B: f32 = 0.75;

/// BM25 relevance scorer for knowledge chunks
pub struct Bm25Scorer {
    /// IDF values for each term in the corpus
    idf: HashMap<String, f32>,
    /// Per-document term frequencies and lengths
    docs: Vec<DocStats>,
    /// Average document length across the corpus
    avg_dl: f32,
}

/// Pre-computed statistics for a single document
struct DocStats {
    term_freq: HashMap<String, u32>,
    len: u32,
}

impl Bm25Scorer {
    /// Build a scorer from a set of knowledge chunks
    ///
    /// Tokenizes each chunk's content, topic, and tags to build the
    /// term-frequency index and IDF table
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn new(chunks: &[KnowledgeChunk]) -> Self {
        let n = chunks.len() as f32;
        let mut doc_freq: HashMap<String, u32> = HashMap::new();
        let mut docs = Vec::with_capacity(chunks.len());
        let mut total_len: u32 = 0;

        for chunk in chunks {
            let tokens = tokenize_chunk(chunk);
            let len = u32::try_from(tokens.len()).unwrap_or(u32::MAX);
            total_len = total_len.saturating_add(len);

            let mut term_freq: HashMap<String, u32> = HashMap::new();
            let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

            for token in &tokens {
                *term_freq.entry(token.clone()).or_insert(0) += 1;
                if seen.insert(token.clone()) {
                    *doc_freq.entry(token.clone()).or_insert(0) += 1;
                }
            }

            docs.push(DocStats { term_freq, len });
        }

        let avg_dl = if docs.is_empty() {
            1.0
        } else {
            #[allow(clippy::cast_possible_truncation)]
            {
                f64::from(total_len) as f32 / docs.len() as f32
            }
        };

        // Compute IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        let mut idf = HashMap::new();
        for (term, df) in &doc_freq {
            #[allow(clippy::cast_precision_loss)]
            let df_f = *df as f32;
            let val = ((n - df_f + 0.5) / (df_f + 0.5)).ln_1p();
            idf.insert(term.clone(), val);
        }

        Self { idf, docs, avg_dl }
    }

    /// Score a query against all chunks
    ///
    /// Returns `(chunk_index, score)` pairs sorted by score descending.
    /// Only returns entries with a positive score.
    #[must_use]
    pub fn score(&self, query: &str) -> Vec<(usize, f32)> {
        let query_tokens = tokenize(query);

        if query_tokens.is_empty() {
            return Vec::new();
        }

        let mut scores: Vec<(usize, f32)> = self
            .docs
            .iter()
            .enumerate()
            .filter_map(|(i, doc)| {
                let score = self.score_doc(doc, &query_tokens);
                if score > 0.0 { Some((i, score)) } else { None }
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
    }

    /// Compute BM25 score for a single document against query tokens
    #[allow(clippy::cast_precision_loss)]
    fn score_doc(&self, doc: &DocStats, query_tokens: &[String]) -> f32 {
        let mut score = 0.0_f32;

        for token in query_tokens {
            let Some(&idf) = self.idf.get(token) else {
                continue;
            };

            let tf = *doc.term_freq.get(token).unwrap_or(&0) as f32;
            let dl = doc.len as f32;

            // BM25 formula: IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl/avgdl))
            let numerator = tf * (K1 + 1.0);
            let denominator = K1.mul_add(1.0 - B + B * dl / self.avg_dl, tf);
            score += idf * numerator / denominator;
        }

        score
    }
}

/// Tokenize a knowledge chunk for indexing
fn tokenize_chunk(chunk: &KnowledgeChunk) -> Vec<String> {
    let mut tokens = tokenize(&chunk.content);

    if let Some(ref topic) = chunk.topic {
        tokens.extend(tokenize(topic));
    }

    for tag in &chunk.tags {
        tokens.extend(tokenize(tag));
    }

    tokens
}

/// Tokenize text: lowercase, strip non-alphanumeric, filter empty
fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|t| {
            t.chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
                .to_lowercase()
        })
        .filter(|t| !t.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::models::KnowledgePriority;

    fn make_chunk(topic: &str, tags: &[&str], content: &str) -> KnowledgeChunk {
        KnowledgeChunk {
            topic: Some(topic.to_string()),
            tags: tags.iter().map(|t| t.to_string()).collect(),
            content: content.to_string(),
            rules: vec![],
            priority: KnowledgePriority::Relevant,
            embedding: None,
        }
    }

    #[test]
    fn single_term_scoring() {
        let chunks = vec![
            make_chunk("Solana", &[], "MCG is a token on Solana"),
            make_chunk("Ethereum", &[], "ETH is the native currency of Ethereum"),
        ];

        let scorer = Bm25Scorer::new(&chunks);
        let results = scorer.score("Solana");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
        assert!(results[0].1 > 0.0);
    }

    #[test]
    fn multi_term_scoring() {
        let chunks = vec![
            make_chunk("Token", &["mcg"], "MCG is a Solana token"),
            make_chunk("Platform", &[], "Omni is a platform for developers"),
            make_chunk("Both", &[], "MCG token on the Omni platform"),
        ];

        let scorer = Bm25Scorer::new(&chunks);
        let results = scorer.score("MCG Omni");

        // "Both" chunk should score highest (has both terms)
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 2);
    }

    #[test]
    fn empty_query_returns_empty() {
        let chunks = vec![make_chunk("Token", &[], "MCG is a token")];
        let scorer = Bm25Scorer::new(&chunks);
        let results = scorer.score("");
        assert!(results.is_empty());
    }

    #[test]
    fn idf_weights_rare_terms_higher() {
        let chunks = vec![
            make_chunk("A", &[], "the common word appears here"),
            make_chunk("B", &[], "the common word appears here too"),
            make_chunk("C", &[], "rare unique special MCG token"),
        ];

        let scorer = Bm25Scorer::new(&chunks);

        // "the" appears in 2 docs, "mcg" in 1 — mcg should have higher IDF
        let the_idf = scorer.idf.get("the").copied().unwrap_or(0.0);
        let mcg_idf = scorer.idf.get("mcg").copied().unwrap_or(0.0);
        assert!(mcg_idf > the_idf);
    }

    #[test]
    fn tags_contribute_to_scoring() {
        let chunks = vec![
            make_chunk("Info", &["solana", "token"], "General information"),
            make_chunk("Other", &[], "Unrelated content about nothing"),
        ];

        let scorer = Bm25Scorer::new(&chunks);
        let results = scorer.score("solana");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn empty_corpus() {
        let scorer = Bm25Scorer::new(&[]);
        let results = scorer.score("anything");
        assert!(results.is_empty());
    }
}

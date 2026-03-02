//! Text embedding for semantic knowledge and memory retrieval

use std::time::Duration;

use mini_moka::sync::Cache;
use thiserror::Error;

/// Embedding dimension for text-embedding-3-small
pub const EMBEDDING_DIM: usize = 1536;

/// Maximum number of cached embeddings
const CACHE_MAX_CAPACITY: u64 = 256;

/// Cache TTL — embeddings are deterministic for a given model+input
const CACHE_TTL_SECS: u64 = 3600;

/// Errors from the embedding API
#[derive(Debug, Error)]
pub enum EmbedderError {
    /// Empty API key
    #[error("OpenAI API key required for embeddings")]
    MissingApiKey,

    /// HTTP request failed
    #[error("embedding request failed: {0}")]
    Request(#[from] reqwest::Error),

    /// API returned a non-success status
    #[error("embedding API error {status}: {body}")]
    Api {
        status: reqwest::StatusCode,
        body: String,
    },

    /// Empty response from API
    #[error("empty embedding response")]
    EmptyResponse,
}

/// Text embedder using `OpenAI`'s embedding API
#[derive(Debug, Clone)]
pub struct Embedder {
    client: reqwest::Client,
    api_key: String,
    model: String,
    cache: Cache<String, Vec<f32>>,
}

impl Embedder {
    /// Build the shared cache instance
    fn build_cache() -> Cache<String, Vec<f32>> {
        Cache::builder()
            .max_capacity(CACHE_MAX_CAPACITY)
            .time_to_live(Duration::from_secs(CACHE_TTL_SECS))
            .build()
    }

    /// Create a new embedder with `OpenAI` API key
    ///
    /// # Errors
    ///
    /// Returns error if API key is empty
    pub fn new(api_key: String) -> Result<Self, EmbedderError> {
        if api_key.is_empty() {
            return Err(EmbedderError::MissingApiKey);
        }

        Ok(Self {
            client: reqwest::Client::new(),
            api_key,
            model: "text-embedding-3-small".to_string(),
            cache: Self::build_cache(),
        })
    }

    /// Create an embedder with a custom model
    ///
    /// # Errors
    ///
    /// Returns error if API key is empty
    pub fn with_model(api_key: String, model: String) -> Result<Self, EmbedderError> {
        if api_key.is_empty() {
            return Err(EmbedderError::MissingApiKey);
        }

        Ok(Self {
            client: reqwest::Client::new(),
            api_key,
            model,
            cache: Self::build_cache(),
        })
    }

    /// Generate embedding for a single text
    ///
    /// Returns a cached result when available, otherwise calls the API
    ///
    /// # Errors
    ///
    /// Returns error if API call fails
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
        let key = text.to_string();
        if let Some(cached) = self.cache.get(&key) {
            return Ok(cached);
        }

        let embeddings = self.embed_batch(&[text]).await?;
        let result = embeddings
            .into_iter()
            .next()
            .ok_or(EmbedderError::EmptyResponse)?;

        self.cache.insert(key, result.clone());
        Ok(result)
    }

    /// Generate embeddings for multiple texts
    ///
    /// Checks the cache per item and only sends uncached texts to the API
    ///
    /// # Errors
    ///
    /// Returns error if API call fails
    pub async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        // Separate cached from uncached
        let mut results: Vec<Option<Vec<f32>>> = Vec::with_capacity(texts.len());
        let mut uncached_indices: Vec<usize> = Vec::new();
        let mut uncached_texts: Vec<&str> = Vec::new();

        for (i, text) in texts.iter().enumerate() {
            let key = (*text).to_string();
            if let Some(cached) = self.cache.get(&key) {
                results.push(Some(cached));
            } else {
                results.push(None);
                uncached_indices.push(i);
                uncached_texts.push(text);
            }
        }

        // All cached — skip API call
        if uncached_texts.is_empty() {
            return Ok(results.into_iter().map(Option::unwrap_or_default).collect());
        }

        let fetched = self.fetch_embeddings(&uncached_texts).await?;

        // Merge fetched results and populate cache
        for (slot_idx, embedding) in uncached_indices.into_iter().zip(fetched) {
            self.cache
                .insert(texts[slot_idx].to_string(), embedding.clone());
            results[slot_idx] = Some(embedding);
        }

        Ok(results.into_iter().map(Option::unwrap_or_default).collect())
    }

    /// Call the `OpenAI` embeddings API
    async fn fetch_embeddings(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        #[derive(serde::Serialize)]
        struct EmbeddingRequest<'a> {
            model: &'a str,
            input: &'a [&'a str],
        }

        #[derive(serde::Deserialize)]
        struct EmbeddingResponse {
            data: Vec<EmbeddingData>,
        }

        #[derive(serde::Deserialize)]
        struct EmbeddingData {
            embedding: Vec<f32>,
            index: usize,
        }

        let request = EmbeddingRequest {
            model: &self.model,
            input: texts,
        };

        let response = self
            .client
            .post("https://api.openai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(EmbedderError::Api { status, body });
        }

        let mut result: EmbeddingResponse = response.json().await?;

        // Sort by index to maintain input order
        result.data.sort_by_key(|d| d.index);

        Ok(result.data.into_iter().map(|d| d.embedding).collect())
    }

    /// Serialize embedding to bytes for storage
    #[must_use]
    pub fn to_bytes(embedding: &[f32]) -> Vec<u8> {
        embedding.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    /// Deserialize embedding from bytes
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| {
                let arr: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
                f32::from_le_bytes(arr)
            })
            .collect()
    }
}

/// Build contextual text for embedding by prepending pack/topic context
///
/// Follows Anthropic's contextual retrieval pattern: each chunk is
/// embedded with its parent document context for better semantic matching
#[must_use]
pub fn contextual_text(pack_name: &str, topic: Option<&str>, content: &str) -> String {
    if let Some(topic) = topic {
        format!("Document: {pack_name}\nSection: {topic}\n\n{content}")
    } else {
        format!("Document: {pack_name}\n\n{content}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_roundtrip() {
        let embedding = vec![1.0, 2.5, -3.14, 0.0, 100.0];
        let bytes = Embedder::to_bytes(&embedding);
        let restored = Embedder::from_bytes(&bytes);

        assert_eq!(embedding.len(), restored.len());
        for (a, b) in embedding.iter().zip(restored.iter()) {
            assert!((a - b).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn empty_api_key_rejected() {
        let result = Embedder::new(String::new());
        assert!(result.is_err());
    }

    #[test]
    fn valid_api_key_accepted() {
        let result = Embedder::new("sk-test123".to_string());
        assert!(result.is_ok());
    }

    #[test]
    fn cache_hit_returns_same_value() {
        let embedder = Embedder::new("sk-test".to_string()).unwrap();
        let vector = vec![1.0, 2.0, 3.0];

        // Pre-populate cache
        embedder
            .cache
            .insert("test query".to_string(), vector.clone());

        // Should return cached value
        let cached = embedder.cache.get(&"test query".to_string());
        assert!(cached.is_some());
        assert_eq!(cached.unwrap(), vector);
    }

    #[test]
    fn cache_miss_returns_none() {
        let embedder = Embedder::new("sk-test".to_string()).unwrap();

        let cached = embedder.cache.get(&"nonexistent".to_string());
        assert!(cached.is_none());
    }

    #[test]
    fn contextual_text_with_topic() {
        let text = contextual_text("crypto-basics", Some("Token Info"), "MCG is on Solana");
        assert_eq!(
            text,
            "Document: crypto-basics\nSection: Token Info\n\nMCG is on Solana"
        );
    }

    #[test]
    fn contextual_text_without_topic() {
        let text = contextual_text("crypto-basics", None, "MCG is on Solana");
        assert_eq!(text, "Document: crypto-basics\n\nMCG is on Solana");
    }
}

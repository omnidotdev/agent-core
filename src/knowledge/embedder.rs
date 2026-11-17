//! Text embedding for semantic knowledge and memory retrieval

use std::time::Duration;

use mini_moka::sync::Cache;
use thiserror::Error;

use super::endpoint::{OPENAI_BASE_URL, ensure_key_destination};

/// Embedding dimension for text-embedding-3-small
pub const EMBEDDING_DIM: usize = 1536;

/// Default embedding model when talking to `OpenAI` directly
pub const DEFAULT_EMBEDDING_MODEL: &str = "text-embedding-3-small";

/// Maximum number of cached embeddings
const CACHE_MAX_CAPACITY: u64 = 256;

/// Cache TTL — embeddings are deterministic for a given model+input
const CACHE_TTL_SECS: u64 = 3600;

/// Errors from the embedding API
#[derive(Debug, Error)]
pub enum EmbedderError {
    /// Empty API key
    #[error("API key required for embeddings")]
    MissingApiKey,

    /// A Synapse API key was bound for a third-party vendor host
    #[error("refusing to send a Synapse API key to third-party host {0}")]
    SynapseKeyToVendor(String),

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

/// Text embedder for any OpenAI-compatible `/embeddings` endpoint
///
/// Points at `OpenAI` by default; pass a Synapse gateway base URL (with a
/// provider-prefixed model id) to route embeddings through Synapse
#[derive(Debug, Clone)]
pub struct Embedder {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
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

    /// Create an embedder that calls `OpenAI` directly
    ///
    /// # Errors
    ///
    /// Returns error if the API key is empty or is a Synapse key
    pub fn new(api_key: String) -> Result<Self, EmbedderError> {
        Self::with_config(
            api_key,
            DEFAULT_EMBEDDING_MODEL.to_string(),
            OPENAI_BASE_URL.to_string(),
        )
    }

    /// Create an embedder that calls `OpenAI` directly with a custom model
    ///
    /// # Errors
    ///
    /// Returns error if the API key is empty or is a Synapse key
    pub fn with_model(api_key: String, model: String) -> Result<Self, EmbedderError> {
        Self::with_config(api_key, model, OPENAI_BASE_URL.to_string())
    }

    /// Create an embedder for an OpenAI-compatible endpoint
    ///
    /// `base_url` is the API root including its version segment (e.g.
    /// `https://api.openai.com/v1` or `https://gateway.synapse.omni.dev/v1`);
    /// requests go to `{base_url}/embeddings`
    ///
    /// # Errors
    ///
    /// Returns error if the API key is empty, or if it is a Synapse key and
    /// `base_url` points at a third-party vendor
    pub fn with_config(
        api_key: String,
        model: String,
        base_url: String,
    ) -> Result<Self, EmbedderError> {
        if api_key.is_empty() {
            return Err(EmbedderError::MissingApiKey);
        }
        let mut base_url = base_url;
        base_url.truncate(base_url.trim_end_matches('/').len());
        ensure_key_destination(&api_key, &base_url).map_err(EmbedderError::SynapseKeyToVendor)?;

        Ok(Self {
            client: reqwest::Client::new(),
            api_key,
            model,
            base_url,
            cache: Self::build_cache(),
        })
    }

    /// Embedding model id sent with each request
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    /// API root requests are sent under
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Full URL embedding requests are posted to
    #[must_use]
    pub fn endpoint_url(&self) -> String {
        format!("{}/embeddings", self.base_url)
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

    /// Call the configured OpenAI-compatible embeddings endpoint
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
            .post(self.endpoint_url())
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
        let embedding = vec![1.0, 2.5, -3.25, 0.0, 100.0];
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
    fn default_constructor_targets_openai() {
        let embedder = Embedder::new("sk-test".to_string()).unwrap();
        assert_eq!(
            embedder.endpoint_url(),
            "https://api.openai.com/v1/embeddings"
        );
        assert_eq!(embedder.model(), "text-embedding-3-small");
    }

    #[test]
    fn with_config_targets_the_given_base_url() {
        let embedder = Embedder::with_config(
            "synapse_abc".to_string(),
            "openai/text-embedding-3-small".to_string(),
            "https://gateway.synapse.omni.dev/v1/".to_string(),
        )
        .unwrap();
        assert_eq!(
            embedder.endpoint_url(),
            "https://gateway.synapse.omni.dev/v1/embeddings"
        );
        assert_eq!(embedder.model(), "openai/text-embedding-3-small");
    }

    #[test]
    fn synapse_key_is_never_sent_to_openai() {
        let err = Embedder::new("synapse_abc".to_string()).unwrap_err();
        assert!(matches!(err, EmbedderError::SynapseKeyToVendor(_)));

        let err = Embedder::with_model("synapse_abc".to_string(), "m".to_string()).unwrap_err();
        assert!(matches!(err, EmbedderError::SynapseKeyToVendor(_)));
    }

    #[test]
    fn synapse_key_is_rejected_for_any_vendor_host() {
        for base_url in [
            "https://api.openai.com/v1",
            "https://API.OpenAI.com:443/v1",
            "https://openrouter.ai/api/v1",
            "https://api.together.xyz/v1",
            "https://generativelanguage.googleapis.com/v1beta/openai",
        ] {
            let result = Embedder::with_config(
                "synapse_abc".to_string(),
                "openai/text-embedding-3-small".to_string(),
                base_url.to_string(),
            );
            assert!(
                matches!(result, Err(EmbedderError::SynapseKeyToVendor(_))),
                "synapse key accepted for {base_url}"
            );
        }
    }

    #[test]
    fn vendor_key_may_go_direct_to_vendor() {
        let embedder = Embedder::with_config(
            "sk-byok".to_string(),
            "text-embedding-3-small".to_string(),
            "https://api.openai.com/v1".to_string(),
        )
        .unwrap();
        assert_eq!(
            embedder.endpoint_url(),
            "https://api.openai.com/v1/embeddings"
        );
    }

    /// Serve one HTTP request, returning (request line, headers + body)
    async fn serve_once(listener: tokio::net::TcpListener, body: &'static str) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let (mut socket, _) = listener.accept().await.unwrap();
        let mut buf = Vec::new();
        let mut chunk = [0u8; 4096];
        loop {
            let n = socket.read(&mut chunk).await.unwrap();
            buf.extend_from_slice(&chunk[..n]);
            let text = String::from_utf8_lossy(&buf).to_string();
            if let Some(split) = text.find("\r\n\r\n") {
                let content_length = text[..split]
                    .lines()
                    .find_map(|l| {
                        let (k, v) = l.split_once(':')?;
                        k.eq_ignore_ascii_case("content-length")
                            .then(|| v.trim().parse::<usize>().ok())?
                    })
                    .unwrap_or(0);
                if buf.len() >= split + 4 + content_length {
                    break;
                }
            }
            if n == 0 {
                break;
            }
        }
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        );
        socket.write_all(response.as_bytes()).await.unwrap();
        String::from_utf8_lossy(&buf).to_string()
    }

    #[tokio::test]
    async fn embed_posts_to_the_configured_base_url() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(serve_once(
            listener,
            r#"{"data":[{"embedding":[0.5,0.25],"index":0}]}"#,
        ));

        let embedder = Embedder::with_config(
            "synapse_abc".to_string(),
            "openai/text-embedding-3-small".to_string(),
            format!("http://{addr}/v1"),
        )
        .unwrap();
        let embedding = embedder.embed("hello").await.unwrap();
        assert_eq!(embedding, vec![0.5, 0.25]);

        let request = server.await.unwrap();
        assert!(
            request.starts_with("POST /v1/embeddings HTTP/1.1"),
            "unexpected request: {request}"
        );
        assert!(
            request
                .to_ascii_lowercase()
                .contains("authorization: bearer synapse_abc"),
            "missing bearer key: {request}"
        );
        assert!(request.contains(r#""model":"openai/text-embedding-3-small""#));
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

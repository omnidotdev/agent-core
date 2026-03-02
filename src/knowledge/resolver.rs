//! Knowledge pack resolver for fetching and caching packs from Manifold

use std::path::PathBuf;
use std::time::{Duration, SystemTime};

use thiserror::Error;

use super::models::{KnowledgeChunk, KnowledgeConfig, KnowledgePack, KnowledgePackRef};

/// Default cache time-to-live (24 hours)
const DEFAULT_CACHE_TTL: Duration = Duration::from_secs(24 * 60 * 60);

/// Errors that can occur during knowledge pack resolution
#[derive(Debug, Error)]
pub enum ResolverError {
    /// Failed to fetch pack from Manifold
    #[error("manifold fetch failed: {0}")]
    Fetch(String),

    /// Failed to parse pack content
    #[error("invalid pack format: {0}")]
    Parse(String),

    /// Failed to read or write cache
    #[error("cache error: {0}")]
    Cache(String),

    /// Invalid pack reference format
    #[error("invalid pack ref: {0}")]
    InvalidRef(String),
}

/// Result type for resolver operations
type Result<T> = std::result::Result<T, ResolverError>;

/// Resolve knowledge pack references to full packs via Manifold
///
/// Fetches packs from the Manifold registry and caches them locally
/// to avoid repeated network requests
#[derive(Debug)]
pub struct KnowledgePackResolver {
    manifold_url: String,
    cache_dir: PathBuf,
    cache_ttl: Duration,
    client: reqwest::Client,
}

impl KnowledgePackResolver {
    /// Create a new resolver with an explicit cache directory
    ///
    /// Uses a 24-hour cache TTL by default
    #[must_use]
    pub fn new(manifold_url: &str, cache_dir: PathBuf) -> Self {
        Self {
            manifold_url: manifold_url.trim_end_matches('/').to_string(),
            cache_dir,
            cache_ttl: DEFAULT_CACHE_TTL,
            client: reqwest::Client::new(),
        }
    }

    /// Create a resolver using the platform default cache directory
    ///
    /// Uses `~/.cache/omni/knowledge/` (or platform equivalent)
    ///
    /// # Errors
    ///
    /// Returns an error if the cache directory cannot be determined
    pub fn with_default_cache(manifold_url: &str) -> Result<Self> {
        let cache_dir = default_cache_dir()
            .ok_or_else(|| ResolverError::Cache("cannot determine cache directory".to_string()))?;

        Ok(Self::new(manifold_url, cache_dir))
    }

    /// Resolve a single pack reference to a full knowledge pack
    ///
    /// Checks the local cache first; fetches from Manifold if the cache
    /// is missing or stale
    ///
    /// # Errors
    ///
    /// Returns an error if the pack cannot be fetched or parsed
    pub async fn resolve(&self, pack_ref: &KnowledgePackRef) -> Result<KnowledgePack> {
        // Check cache first
        if let Some(cached) = self.read_cache(pack_ref) {
            tracing::debug!(pack_ref = %pack_ref.pack_ref, "using cached knowledge pack");
            return Ok(cached);
        }

        // Fetch from Manifold
        let pack = self.fetch_from_manifold(pack_ref).await?;

        // Write to cache (log but don't fail on cache write errors)
        if let Err(e) = self.write_cache(pack_ref, &pack) {
            tracing::warn!(
                pack_ref = %pack_ref.pack_ref,
                error = %e,
                "failed to cache knowledge pack"
            );
        }

        Ok(pack)
    }

    /// Resolve all pack references concurrently
    ///
    /// Returns a vec of results in the same order as the input refs
    pub async fn resolve_all(&self, refs: &[KnowledgePackRef]) -> Vec<Result<KnowledgePack>> {
        let futures: Vec<_> = refs.iter().map(|r| self.resolve(r)).collect();
        futures::future::join_all(futures).await
    }

    /// Build the cache file path for a pack reference
    ///
    /// Layout: `{cache_dir}/{namespace}/{pack_name}/{version}.json`
    fn cache_path(&self, pack_ref: &KnowledgePackRef) -> Result<PathBuf> {
        let (namespace, pack_name) = parse_pack_ref(&pack_ref.pack_ref)?;
        let version = pack_ref.version.as_deref().unwrap_or("latest");

        Ok(self
            .cache_dir
            .join(namespace)
            .join(pack_name)
            .join(format!("{version}.json")))
    }

    /// Read a pack from the local cache if it exists and is fresh
    fn read_cache(&self, pack_ref: &KnowledgePackRef) -> Option<KnowledgePack> {
        let path = self.cache_path(pack_ref).ok()?;

        let metadata = std::fs::metadata(&path).ok()?;
        let modified = metadata.modified().ok()?;
        let age = SystemTime::now().duration_since(modified).ok()?;

        if age > self.cache_ttl {
            tracing::debug!(
                pack_ref = %pack_ref.pack_ref,
                age_secs = age.as_secs(),
                "cache entry is stale"
            );
            return None;
        }

        let contents = std::fs::read_to_string(&path).ok()?;

        match serde_json::from_str::<KnowledgePack>(&contents) {
            Ok(pack) => Some(pack),
            Err(e) => {
                tracing::warn!(
                    pack_ref = %pack_ref.pack_ref,
                    error = %e,
                    "corrupt cache entry"
                );
                None
            }
        }
    }

    /// Write a resolved pack to the local cache
    fn write_cache(&self, pack_ref: &KnowledgePackRef, pack: &KnowledgePack) -> Result<()> {
        let path = self.cache_path(pack_ref)?;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| ResolverError::Cache(format!("failed to create cache dir: {e}")))?;
        }

        let json = serde_json::to_string_pretty(pack)
            .map_err(|e| ResolverError::Cache(format!("failed to serialize pack: {e}")))?;

        std::fs::write(&path, json)
            .map_err(|e| ResolverError::Cache(format!("failed to write cache file: {e}")))?;

        tracing::debug!(
            pack_ref = %pack_ref.pack_ref,
            path = %path.display(),
            "cached knowledge pack"
        );

        Ok(())
    }

    /// Fetch a knowledge pack from the Manifold registry
    async fn fetch_from_manifold(&self, pack_ref: &KnowledgePackRef) -> Result<KnowledgePack> {
        let (namespace, pack_name) = parse_pack_ref(&pack_ref.pack_ref)?;

        let url = format!(
            "{}/@{}/knowledge/{}",
            self.manifold_url, namespace, pack_name
        );

        tracing::debug!(url = %url, "fetching knowledge pack from manifold");

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| ResolverError::Fetch(e.to_string()))?;

        if !response.status().is_success() {
            return Err(ResolverError::Fetch(format!(
                "pack not found: {} ({})",
                pack_ref.pack_ref,
                response.status()
            )));
        }

        let body = response
            .text()
            .await
            .map_err(|e| ResolverError::Fetch(e.to_string()))?;

        serde_json::from_str::<KnowledgePack>(&body)
            .map_err(|e| ResolverError::Parse(format!("{}: {e}", pack_ref.pack_ref)))
    }
}

/// Hydrate chunk embedding fields from a pack's embeddings section
///
/// If the pack contains a `PackEmbeddings` with vectors keyed by chunk
/// index, each chunk's `embedding` field is populated from that map.
/// Existing embeddings on chunks are preserved
#[must_use]
pub fn hydrate_embeddings(pack: &KnowledgePack) -> Vec<KnowledgeChunk> {
    let mut chunks = pack.chunks.clone();

    if let Some(ref embeddings) = pack.embeddings {
        for (i, chunk) in chunks.iter_mut().enumerate() {
            if chunk.embedding.is_none() {
                if let Some(vec) = embeddings.vectors.get(&i.to_string()) {
                    chunk.embedding = Some(vec.clone());
                }
            }
        }
    }

    chunks
}

/// Resolve all knowledge pack refs and merge chunks with inline knowledge
///
/// Pack chunks are appended after inline chunks. If a pack ref specifies
/// a priority override, all chunks from that pack inherit the override.
///
/// Resolution is best-effort: individual pack failures are logged but
/// do not prevent other packs from loading
///
/// # Errors
///
/// Returns an error if the resolver cannot be created
pub async fn resolve_and_merge(
    config: &KnowledgeConfig,
    manifold_url: &str,
) -> std::result::Result<Vec<KnowledgeChunk>, ResolverError> {
    let mut all_chunks = config.inline.clone();

    if config.packs.is_empty() {
        return Ok(all_chunks);
    }

    let resolver = KnowledgePackResolver::with_default_cache(manifold_url)?;
    let results = resolver.resolve_all(&config.packs).await;

    for (i, result) in results.into_iter().enumerate() {
        match result {
            Ok(pack) => {
                tracing::info!(
                    name = %pack.name,
                    chunks = pack.chunks.len(),
                    "loaded knowledge pack"
                );

                // Apply priority override from the pack ref if set
                let priority_override = config.packs.get(i).and_then(|r| r.priority);

                for mut chunk in pack.chunks {
                    if let Some(priority) = priority_override {
                        chunk.priority = priority;
                    }
                    all_chunks.push(chunk);
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "failed to resolve knowledge pack");
            }
        }
    }

    Ok(all_chunks)
}

/// Parse a pack ref string into (namespace, `pack_name`)
///
/// Expected format: `@{namespace}/knowledge/{pack_name}`
fn parse_pack_ref(pack_ref: &str) -> Result<(&str, &str)> {
    let trimmed = pack_ref.strip_prefix('@').unwrap_or(pack_ref);

    let parts: Vec<&str> = trimmed.splitn(3, '/').collect();

    match parts.as_slice() {
        [namespace, "knowledge", pack_name] => Ok((namespace, pack_name)),
        _ => Err(ResolverError::InvalidRef(format!(
            "expected @{{namespace}}/knowledge/{{pack_name}}, got: {pack_ref}"
        ))),
    }
}

/// Get the default knowledge cache directory (`~/.cache/omni/knowledge/`)
fn default_cache_dir() -> Option<PathBuf> {
    directories::BaseDirs::new().map(|base| base.cache_dir().join("omni").join("knowledge"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_valid_pack_ref() {
        let (ns, name) = parse_pack_ref("@omni/knowledge/crypto-basics").unwrap();
        assert_eq!(ns, "omni");
        assert_eq!(name, "crypto-basics");
    }

    #[test]
    fn parse_pack_ref_without_at() {
        let (ns, name) = parse_pack_ref("omni/knowledge/crypto-basics").unwrap();
        assert_eq!(ns, "omni");
        assert_eq!(name, "crypto-basics");
    }

    #[test]
    fn parse_invalid_pack_ref() {
        let result = parse_pack_ref("invalid-ref");
        assert!(result.is_err());
    }

    #[test]
    fn parse_wrong_artifact_type() {
        let result = parse_pack_ref("@omni/skills/some-skill");
        assert!(result.is_err());
    }

    #[test]
    fn resolver_trims_trailing_slash() {
        let resolver =
            KnowledgePackResolver::new("https://manifold.omni.dev/", PathBuf::from("/tmp/cache"));
        assert_eq!(resolver.manifold_url, "https://manifold.omni.dev");
    }

    #[test]
    fn cache_path_layout() {
        let resolver =
            KnowledgePackResolver::new("https://manifold.omni.dev", PathBuf::from("/tmp/cache"));

        let pack_ref = KnowledgePackRef {
            pack_ref: "@omni/knowledge/crypto-basics".to_string(),
            version: Some("1.0.0".to_string()),
            priority: None,
        };

        let path = resolver.cache_path(&pack_ref).unwrap();
        assert_eq!(
            path,
            PathBuf::from("/tmp/cache/omni/crypto-basics/1.0.0.json")
        );
    }

    #[test]
    fn cache_path_defaults_to_latest() {
        let resolver =
            KnowledgePackResolver::new("https://manifold.omni.dev", PathBuf::from("/tmp/cache"));

        let pack_ref = KnowledgePackRef {
            pack_ref: "@omni/knowledge/crypto-basics".to_string(),
            version: None,
            priority: None,
        };

        let path = resolver.cache_path(&pack_ref).unwrap();
        assert_eq!(
            path,
            PathBuf::from("/tmp/cache/omni/crypto-basics/latest.json")
        );
    }

    #[test]
    fn hydrate_embeddings_populates_chunks() {
        use super::super::models::{KnowledgePriority, PackEmbeddings};
        use std::collections::HashMap;

        let mut vectors = HashMap::new();
        vectors.insert("0".to_string(), vec![1.0, 0.0, 0.0]);
        vectors.insert("1".to_string(), vec![0.0, 1.0, 0.0]);

        let pack = KnowledgePack {
            schema: None,
            version: "1.0.0".to_string(),
            name: "test-pack".to_string(),
            description: None,
            tags: vec![],
            chunks: vec![
                KnowledgeChunk {
                    topic: Some("A".to_string()),
                    tags: vec![],
                    content: "Content A".to_string(),
                    rules: vec![],
                    priority: KnowledgePriority::Relevant,
                    embedding: None,
                },
                KnowledgeChunk {
                    topic: Some("B".to_string()),
                    tags: vec![],
                    content: "Content B".to_string(),
                    rules: vec![],
                    priority: KnowledgePriority::Relevant,
                    embedding: None,
                },
            ],
            embeddings: Some(PackEmbeddings {
                model: "text-embedding-3-small".to_string(),
                dimensions: 3,
                vectors,
            }),
        };

        let hydrated = hydrate_embeddings(&pack);
        assert_eq!(
            hydrated[0].embedding.as_ref().unwrap(),
            &vec![1.0, 0.0, 0.0]
        );
        assert_eq!(
            hydrated[1].embedding.as_ref().unwrap(),
            &vec![0.0, 1.0, 0.0]
        );
    }

    #[test]
    fn hydrate_embeddings_skips_without_embeddings() {
        use super::super::models::KnowledgePriority;

        let pack = KnowledgePack {
            schema: None,
            version: "1.0.0".to_string(),
            name: "test-pack".to_string(),
            description: None,
            tags: vec![],
            chunks: vec![KnowledgeChunk {
                topic: Some("A".to_string()),
                tags: vec![],
                content: "Content A".to_string(),
                rules: vec![],
                priority: KnowledgePriority::Relevant,
                embedding: None,
            }],
            embeddings: None,
        };

        let hydrated = hydrate_embeddings(&pack);
        assert!(hydrated[0].embedding.is_none());
    }

    #[test]
    fn hydrate_embeddings_preserves_existing() {
        use super::super::models::{KnowledgePriority, PackEmbeddings};
        use std::collections::HashMap;

        let mut vectors = HashMap::new();
        vectors.insert("0".to_string(), vec![0.0, 0.0, 1.0]);

        let pack = KnowledgePack {
            schema: None,
            version: "1.0.0".to_string(),
            name: "test-pack".to_string(),
            description: None,
            tags: vec![],
            chunks: vec![KnowledgeChunk {
                topic: Some("A".to_string()),
                tags: vec![],
                content: "Content A".to_string(),
                rules: vec![],
                priority: KnowledgePriority::Relevant,
                // Already has an embedding; should not be overwritten
                embedding: Some(vec![1.0, 0.0, 0.0]),
            }],
            embeddings: Some(PackEmbeddings {
                model: "text-embedding-3-small".to_string(),
                dimensions: 3,
                vectors,
            }),
        };

        let hydrated = hydrate_embeddings(&pack);
        // Existing embedding preserved
        assert_eq!(
            hydrated[0].embedding.as_ref().unwrap(),
            &vec![1.0, 0.0, 0.0]
        );
    }
}

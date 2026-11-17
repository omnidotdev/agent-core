//! Condense multi-turn conversation into a standalone retrieval query

use async_trait::async_trait;
use thiserror::Error;

use super::endpoint::{OPENAI_BASE_URL, ensure_key_destination};

/// System prompt that instructs the LLM to produce a search query
const CONDENSE_SYSTEM_PROMPT: &str = "\
You are a search-query rewriter. Given a conversation history and the user's \
latest message, produce a single standalone search query that captures the \
user's information need. The query should:\n\
- Resolve all pronouns and references (\"that\", \"it\", \"the one\") to their concrete referents\n\
- Be concise (under 30 words)\n\
- Be suitable for semantic similarity search over a knowledge base\n\
- Contain only the rewritten query, no explanation\n\
\n\
If the latest message is already a standalone query, return it unchanged.";

/// Errors from query condensation
#[derive(Debug, Error)]
pub enum CondenseError {
    /// LLM request failed
    #[error("LLM request failed: {0}")]
    Request(String),
}

/// Condense multi-turn conversation into a standalone retrieval query
#[async_trait]
pub trait QueryCondenser: Send + Sync {
    /// Rewrite the current user message as a standalone search query
    /// given prior conversation history
    async fn condense(&self, current: &str, history: &[&str]) -> Result<String, CondenseError>;
}

/// LLM-backed query condenser
///
/// Uses a fast/cheap model to rewrite the latest user message as a
/// standalone retrieval query by resolving anaphoric references
pub struct LlmCondenser {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
}

impl LlmCondenser {
    /// Create a condenser that calls `OpenAI` chat completions directly
    ///
    /// # Errors
    ///
    /// Returns error if the API key is empty or is a Synapse key
    pub fn new(api_key: String) -> Result<Self, CondenseError> {
        Self::with_config(
            api_key,
            "gpt-4o-mini".to_string(),
            OPENAI_BASE_URL.to_string(),
        )
    }

    /// Create a condenser with a custom model and OpenAI-compatible base URL
    ///
    /// # Errors
    ///
    /// Returns error if the API key is empty, or if it is a Synapse key and
    /// `base_url` points at a third-party vendor
    pub fn with_config(
        api_key: String,
        model: String,
        base_url: String,
    ) -> Result<Self, CondenseError> {
        if api_key.is_empty() {
            return Err(CondenseError::Request(
                "API key required for query condensation".to_string(),
            ));
        }
        let mut base_url = base_url;
        base_url.truncate(base_url.trim_end_matches('/').len());
        ensure_key_destination(&api_key, &base_url).map_err(|host| {
            CondenseError::Request(format!(
                "refusing to send a Synapse API key to third-party host {host}"
            ))
        })?;

        Ok(Self {
            client: reqwest::Client::new(),
            api_key,
            model,
            base_url,
        })
    }

    /// API root requests are sent under
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Chat model id used for condensing
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }
}

#[async_trait]
impl QueryCondenser for LlmCondenser {
    async fn condense(&self, current: &str, history: &[&str]) -> Result<String, CondenseError> {
        #[derive(serde::Serialize)]
        struct ChatMessage<'a> {
            role: &'a str,
            content: &'a str,
        }

        #[derive(serde::Serialize)]
        struct ChatRequest<'a> {
            model: &'a str,
            messages: Vec<ChatMessage<'a>>,
            max_tokens: u32,
            temperature: f32,
        }

        #[derive(serde::Deserialize)]
        struct ChatResponse {
            choices: Vec<ChatChoice>,
        }

        #[derive(serde::Deserialize)]
        struct ChatChoice {
            message: ChatChoiceMessage,
        }

        #[derive(serde::Deserialize)]
        struct ChatChoiceMessage {
            content: Option<String>,
        }

        // No history means no rewriting needed
        if history.is_empty() {
            return Ok(current.to_string());
        }

        // Build a user prompt with conversation context
        let mut user_prompt = String::from("Conversation history:\n");
        for msg in history {
            user_prompt.push_str("- ");
            user_prompt.push_str(msg);
            user_prompt.push('\n');
        }
        user_prompt.push_str("\nLatest message: ");
        user_prompt.push_str(current);

        let request = ChatRequest {
            model: &self.model,
            messages: vec![
                ChatMessage {
                    role: "system",
                    content: CONDENSE_SYSTEM_PROMPT,
                },
                ChatMessage {
                    role: "user",
                    content: &user_prompt,
                },
            ],
            max_tokens: 150,
            temperature: 0.0,
        };

        let url = format!("{}/chat/completions", self.base_url);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await
            .map_err(|e| CondenseError::Request(e.to_string()))?;

        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(CondenseError::Request(format!("API error: {body}")));
        }

        let result: ChatResponse = response
            .json()
            .await
            .map_err(|e| CondenseError::Request(e.to_string()))?;

        result
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| CondenseError::Request("empty response from condenser".to_string()))
    }
}

/// Build a retrieval query with optional LLM condensation
///
/// Tries the condenser first; falls back to the concatenation-based
/// `build_retrieval_query` on error
pub async fn build_retrieval_query_condensed(
    current: &str,
    history: &[&str],
    max_turns: usize,
    condenser: Option<&dyn QueryCondenser>,
) -> String {
    if let Some(condenser) = condenser {
        match condenser.condense(current, history).await {
            Ok(query) => {
                tracing::debug!(query = %query, "condensed retrieval query");
                return query;
            }
            Err(e) => {
                tracing::warn!(error = %e, "query condensation failed, using fallback");
            }
        }
    }

    super::selection::build_retrieval_query(current, history, max_turns)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synapse_key_is_never_sent_to_openai() {
        assert!(LlmCondenser::new("synapse_abc".to_string()).is_err());
        assert!(
            LlmCondenser::with_config(
                "synapse_abc".to_string(),
                "openai/gpt-4o-mini".to_string(),
                "https://api.openai.com/v1".to_string(),
            )
            .is_err()
        );
    }

    #[test]
    fn synapse_key_may_go_to_the_gateway() {
        let condenser = LlmCondenser::with_config(
            "synapse_abc".to_string(),
            "openai/gpt-4o-mini".to_string(),
            "https://gateway.synapse.omni.dev/v1/".to_string(),
        )
        .unwrap();
        assert_eq!(condenser.base_url(), "https://gateway.synapse.omni.dev/v1");
        assert_eq!(condenser.model(), "openai/gpt-4o-mini");
    }

    #[test]
    fn vendor_key_defaults_to_openai() {
        let condenser = LlmCondenser::new("sk-byok".to_string()).unwrap();
        assert_eq!(condenser.base_url(), "https://api.openai.com/v1");
    }

    struct MockCondenser {
        response: Result<String, CondenseError>,
    }

    #[async_trait]
    impl QueryCondenser for MockCondenser {
        async fn condense(
            &self,
            _current: &str,
            _history: &[&str],
        ) -> Result<String, CondenseError> {
            match &self.response {
                Ok(s) => Ok(s.clone()),
                Err(_) => Err(CondenseError::Request("mock error".to_string())),
            }
        }
    }

    #[tokio::test]
    async fn condensed_query_uses_condenser() {
        let condenser = MockCondenser {
            response: Ok("standalone query about MCG token".to_string()),
        };

        let result = build_retrieval_query_condensed(
            "tell me more about that",
            &["what is MCG?"],
            3,
            Some(&condenser),
        )
        .await;

        assert_eq!(result, "standalone query about MCG token");
    }

    #[tokio::test]
    async fn condensed_query_falls_back_on_error() {
        let condenser = MockCondenser {
            response: Err(CondenseError::Request("fail".to_string())),
        };

        let result =
            build_retrieval_query_condensed("tell me more", &["what is MCG?"], 3, Some(&condenser))
                .await;

        // Falls back to concatenation
        assert!(result.contains("MCG"));
        assert!(result.contains("tell me more"));
    }

    #[tokio::test]
    async fn condensed_query_no_condenser_uses_fallback() {
        let result =
            build_retrieval_query_condensed("tell me more", &["what is MCG?"], 3, None).await;

        assert!(result.contains("MCG"));
        assert!(result.contains("tell me more"));
    }

    #[test]
    fn empty_api_key_rejected() {
        let result = LlmCondenser::new(String::new());
        assert!(result.is_err());
    }
}

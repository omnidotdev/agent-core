//! Default provider and model definitions

use std::collections::HashMap;

use super::types::{ModelInfo, ProviderApiType, ProviderConfig};

/// Get the default provider configurations
#[must_use]
pub fn default_providers() -> HashMap<String, ProviderConfig> {
    let mut providers = HashMap::new();

    providers.insert(
        "anthropic".to_string(),
        ProviderConfig {
            api_type: ProviderApiType::Anthropic,
            base_url: None,
            api_key_env: Some("ANTHROPIC_API_KEY".to_string()),
            api_key: None,
        },
    );

    providers.insert(
        "openai".to_string(),
        ProviderConfig {
            api_type: ProviderApiType::OpenAi,
            base_url: None,
            api_key_env: Some("OPENAI_API_KEY".to_string()),
            api_key: None,
        },
    );

    providers.insert(
        "ollama".to_string(),
        ProviderConfig {
            api_type: ProviderApiType::OpenAi,
            base_url: Some("http://localhost:11434/v1".to_string()),
            api_key_env: None,
            api_key: None,
        },
    );

    providers.insert(
        "lmstudio".to_string(),
        ProviderConfig {
            api_type: ProviderApiType::OpenAi,
            base_url: Some("http://localhost:1234/v1".to_string()),
            api_key_env: None,
            api_key: None,
        },
    );

    providers.insert(
        "groq".to_string(),
        ProviderConfig {
            api_type: ProviderApiType::Groq,
            base_url: None,
            api_key_env: Some("GROQ_API_KEY".to_string()),
            api_key: None,
        },
    );

    providers.insert(
        "google".to_string(),
        ProviderConfig {
            api_type: ProviderApiType::Google,
            base_url: None,
            api_key_env: Some("GOOGLE_API_KEY".to_string()),
            api_key: None,
        },
    );

    providers.insert(
        "mistral".to_string(),
        ProviderConfig {
            api_type: ProviderApiType::Mistral,
            base_url: None,
            api_key_env: Some("MISTRAL_API_KEY".to_string()),
            api_key: None,
        },
    );

    providers.insert(
        "openrouter".to_string(),
        ProviderConfig {
            api_type: ProviderApiType::OpenAi,
            base_url: Some("https://openrouter.ai/api/v1".to_string()),
            api_key_env: Some("OPENROUTER_API_KEY".to_string()),
            api_key: None,
        },
    );

    providers.insert(
        "together".to_string(),
        ProviderConfig {
            api_type: ProviderApiType::OpenAi,
            base_url: Some("https://api.together.xyz/v1".to_string()),
            api_key_env: Some("TOGETHER_API_KEY".to_string()),
            api_key: None,
        },
    );

    providers.insert(
        "kimi".to_string(),
        ProviderConfig {
            api_type: ProviderApiType::OpenAi,
            base_url: Some("https://api.moonshot.cn/v1".to_string()),
            api_key_env: Some("MOONSHOT_API_KEY".to_string()),
            api_key: None,
        },
    );

    providers.insert(
        "synapse".to_string(),
        ProviderConfig {
            api_type: ProviderApiType::Synapse,
            base_url: Some("https://gateway.synapse.omni.dev".to_string()),
            api_key_env: Some("SYNAPSE_API_KEY".to_string()),
            api_key: None,
        },
    );

    providers
}

/// Get the default model definitions
#[must_use]
pub fn default_models() -> Vec<ModelInfo> {
    vec![
        // Anthropic
        ModelInfo {
            id: "claude-sonnet-4-20250514".to_string(),
            provider: "anthropic".to_string(),
        },
        ModelInfo {
            id: "claude-opus-4-20250514".to_string(),
            provider: "anthropic".to_string(),
        },
        ModelInfo {
            id: "claude-3-5-haiku-20241022".to_string(),
            provider: "anthropic".to_string(),
        },
        // OpenAI
        ModelInfo {
            id: "gpt-4o".to_string(),
            provider: "openai".to_string(),
        },
        ModelInfo {
            id: "gpt-4-turbo".to_string(),
            provider: "openai".to_string(),
        },
        ModelInfo {
            id: "gpt-3.5-turbo".to_string(),
            provider: "openai".to_string(),
        },
        ModelInfo {
            id: "o1".to_string(),
            provider: "openai".to_string(),
        },
        ModelInfo {
            id: "o1-mini".to_string(),
            provider: "openai".to_string(),
        },
        // Groq (fast inference)
        ModelInfo {
            id: "llama-3.3-70b-versatile".to_string(),
            provider: "groq".to_string(),
        },
        ModelInfo {
            id: "llama-3.1-8b-instant".to_string(),
            provider: "groq".to_string(),
        },
        ModelInfo {
            id: "mixtral-8x7b-32768".to_string(),
            provider: "groq".to_string(),
        },
        // Google
        ModelInfo {
            id: "gemini-2.0-flash".to_string(),
            provider: "google".to_string(),
        },
        ModelInfo {
            id: "gemini-1.5-pro".to_string(),
            provider: "google".to_string(),
        },
        // Mistral
        ModelInfo {
            id: "mistral-large-latest".to_string(),
            provider: "mistral".to_string(),
        },
        ModelInfo {
            id: "codestral-latest".to_string(),
            provider: "mistral".to_string(),
        },
        // Together
        ModelInfo {
            id: "meta-llama/Llama-3.3-70B-Instruct-Turbo".to_string(),
            provider: "together".to_string(),
        },
        ModelInfo {
            id: "Qwen/Qwen2.5-Coder-32B-Instruct".to_string(),
            provider: "together".to_string(),
        },
        // Kimi (Moonshot AI)
        ModelInfo {
            id: "kimi-k2.5".to_string(),
            provider: "kimi".to_string(),
        },
        ModelInfo {
            id: "moonshot-v1-128k".to_string(),
            provider: "kimi".to_string(),
        },
        ModelInfo {
            id: "moonshot-v1-32k".to_string(),
            provider: "kimi".to_string(),
        },
    ]
}

/// Detect provider by model ID prefix
///
/// Returns the provider name if the model ID matches a known prefix pattern
#[must_use]
pub fn detect_provider_by_prefix(model_id: &str) -> Option<&'static str> {
    let lower = model_id.to_lowercase();

    if lower.starts_with("claude") {
        Some("anthropic")
    } else if lower.starts_with("gpt") || lower.starts_with("o1") {
        Some("openai")
    } else if lower.starts_with("kimi") || lower.starts_with("moonshot") {
        Some("kimi")
    } else if lower.starts_with("gemini") {
        Some("google")
    } else if lower.starts_with("llama") || lower.starts_with("mixtral") {
        Some("groq")
    } else if lower.starts_with("mistral") || lower.starts_with("codestral") {
        Some("mistral")
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_providers_has_expected_count() {
        let providers = default_providers();
        assert_eq!(providers.len(), 11);
    }

    #[test]
    fn default_providers_has_all_entries() {
        let providers = default_providers();
        let expected = [
            "anthropic",
            "openai",
            "ollama",
            "lmstudio",
            "groq",
            "google",
            "mistral",
            "openrouter",
            "together",
            "kimi",
            "synapse",
        ];
        for name in &expected {
            assert!(providers.contains_key(*name), "missing provider: {name}");
        }
    }

    #[test]
    fn default_models_count() {
        let models = default_models();
        assert_eq!(models.len(), 20);
    }

    #[test]
    fn detect_provider_by_prefix_known() {
        assert_eq!(
            detect_provider_by_prefix("claude-sonnet-4"),
            Some("anthropic")
        );
        assert_eq!(detect_provider_by_prefix("gpt-4o"), Some("openai"));
        assert_eq!(detect_provider_by_prefix("o1-mini"), Some("openai"));
        assert_eq!(detect_provider_by_prefix("kimi-k2.5"), Some("kimi"));
        assert_eq!(detect_provider_by_prefix("moonshot-v1"), Some("kimi"));
        assert_eq!(
            detect_provider_by_prefix("gemini-2.0-flash"),
            Some("google")
        );
        assert_eq!(detect_provider_by_prefix("llama-3.3-70b"), Some("groq"));
        assert_eq!(detect_provider_by_prefix("mixtral-8x7b"), Some("groq"));
        assert_eq!(detect_provider_by_prefix("mistral-large"), Some("mistral"));
        assert_eq!(
            detect_provider_by_prefix("codestral-latest"),
            Some("mistral")
        );
    }

    #[test]
    fn detect_provider_by_prefix_case_insensitive() {
        assert_eq!(
            detect_provider_by_prefix("CLAUDE-SONNET"),
            Some("anthropic")
        );
        assert_eq!(detect_provider_by_prefix("GPT-4o"), Some("openai"));
        assert_eq!(detect_provider_by_prefix("KIMI-K2.5"), Some("kimi"));
    }

    #[test]
    fn detect_provider_by_prefix_unknown() {
        assert_eq!(detect_provider_by_prefix("unknown-model"), None);
        assert_eq!(detect_provider_by_prefix(""), None);
    }
}

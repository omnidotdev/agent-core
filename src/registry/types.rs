//! Provider registry types

use std::fmt;

use serde::{Deserialize, Serialize};

/// Provider API type
///
/// Determines which API format to use for communication. Known providers
/// map to named variants; unknown strings fall through to `Custom`
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ProviderApiType {
    /// Anthropic Messages API
    #[default]
    Anthropic,
    /// `OpenAI` Chat Completions API (also used by compatible providers)
    OpenAi,
    /// Google Gemini API
    Google,
    /// Groq API
    Groq,
    /// Mistral API
    Mistral,
    /// Synapse AI router (unified LLM gateway)
    Synapse,
    /// Extension point for consumer-specific providers
    Custom(String),
}

impl Serialize for ProviderApiType {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Anthropic => serializer.serialize_str("anthropic"),
            Self::OpenAi => serializer.serialize_str("openai"),
            Self::Google => serializer.serialize_str("google"),
            Self::Groq => serializer.serialize_str("groq"),
            Self::Mistral => serializer.serialize_str("mistral"),
            Self::Synapse => serializer.serialize_str("synapse"),
            Self::Custom(s) => serializer.serialize_str(s),
        }
    }
}

impl<'de> Deserialize<'de> for ProviderApiType {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Ok(Self::from(s.as_str()))
    }
}

impl From<&str> for ProviderApiType {
    fn from(s: &str) -> Self {
        match s {
            "anthropic" => Self::Anthropic,
            "openai" => Self::OpenAi,
            "google" => Self::Google,
            "groq" => Self::Groq,
            "mistral" => Self::Mistral,
            "synapse" => Self::Synapse,
            other => Self::Custom(other.to_string()),
        }
    }
}

impl fmt::Display for ProviderApiType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Anthropic => write!(f, "anthropic"),
            Self::OpenAi => write!(f, "openai"),
            Self::Google => write!(f, "google"),
            Self::Groq => write!(f, "groq"),
            Self::Mistral => write!(f, "mistral"),
            Self::Synapse => write!(f, "synapse"),
            Self::Custom(s) => write!(f, "{s}"),
        }
    }
}

/// Individual provider configuration
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// API type (determines which provider implementation to use)
    #[serde(rename = "type", default)]
    pub api_type: ProviderApiType,

    /// Base URL override (for OpenAI-compatible providers)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,

    /// Environment variable name for API key
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,

    /// Direct API key (discouraged, prefer `api_key_env`)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
}

/// Model information with provider association
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model identifier (e.g., "claude-sonnet-4-20250514", "gpt-4o")
    pub id: String,
    /// Provider name (e.g., "anthropic", "openai")
    pub provider: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_round_trip_known_variants() {
        let variants = [
            (ProviderApiType::Anthropic, "\"anthropic\""),
            (ProviderApiType::OpenAi, "\"openai\""),
            (ProviderApiType::Google, "\"google\""),
            (ProviderApiType::Groq, "\"groq\""),
            (ProviderApiType::Mistral, "\"mistral\""),
            (ProviderApiType::Synapse, "\"synapse\""),
        ];

        for (variant, expected_json) in &variants {
            let json = serde_json::to_string(variant).unwrap();
            assert_eq!(&json, expected_json);
            let deserialized: ProviderApiType = serde_json::from_str(&json).unwrap();
            assert_eq!(&deserialized, variant);
        }
    }

    #[test]
    fn serde_custom_variant() {
        let custom = ProviderApiType::Custom("some-provider".to_string());
        let json = serde_json::to_string(&custom).unwrap();
        assert_eq!(json, "\"some-provider\"");

        let deserialized: ProviderApiType = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, custom);
    }

    #[test]
    fn synapse_deserializes_to_named_variant() {
        let json = "\"synapse\"";
        let result: ProviderApiType = serde_json::from_str(json).unwrap();
        assert_eq!(result, ProviderApiType::Synapse);
    }

    #[test]
    fn unknown_string_deserializes_to_custom() {
        let json = "\"some-future-provider\"";
        let result: ProviderApiType = serde_json::from_str(json).unwrap();
        assert_eq!(
            result,
            ProviderApiType::Custom("some-future-provider".to_string())
        );
    }

    #[test]
    fn default_is_anthropic() {
        assert_eq!(ProviderApiType::default(), ProviderApiType::Anthropic);
    }

    #[test]
    fn provider_config_toml_round_trip() {
        let config = ProviderConfig {
            api_type: ProviderApiType::Synapse,
            base_url: Some("http://localhost:6111".to_string()),
            api_key_env: None,
            api_key: None,
        };

        let toml_str = toml::to_string(&config).unwrap();
        let deserialized: ProviderConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(deserialized, config);
    }
}

//! Provider factory with extension support

use std::collections::HashMap;

use super::types::{ProviderApiType, ProviderConfig};
use crate::provider::LlmProvider;
use crate::providers::{AnthropicProvider, OpenAiProvider, UnifiedProvider};

/// Factory function for creating provider instances
pub type ProviderFactoryFn =
    Box<dyn Fn(&str, &ProviderConfig) -> anyhow::Result<Box<dyn LlmProvider>> + Send + Sync>;

/// Provider registry with built-in and custom factory support
pub struct ProviderRegistry {
    custom_factories: HashMap<String, ProviderFactoryFn>,
}

impl std::fmt::Debug for ProviderRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderRegistry")
            .field(
                "custom_factories",
                &self.custom_factories.keys().collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ProviderRegistry {
    /// Create a new empty registry
    #[must_use]
    pub fn new() -> Self {
        Self {
            custom_factories: HashMap::new(),
        }
    }

    /// Register a factory for a custom provider type
    ///
    /// The `type_name` should match the string used in `Custom(type_name)`
    pub fn register_factory(&mut self, type_name: impl Into<String>, factory: ProviderFactoryFn) {
        self.custom_factories.insert(type_name.into(), factory);
    }

    /// Create a provider instance from a name and config
    ///
    /// Built-in types are handled directly; `Custom` types delegate
    /// to registered factories.
    ///
    /// # Errors
    ///
    /// Returns error if the provider type is unknown, no factory is
    /// registered for a custom type, or provider creation fails
    pub fn create_provider(
        &self,
        name: &str,
        config: &ProviderConfig,
    ) -> anyhow::Result<Box<dyn LlmProvider>> {
        match &config.api_type {
            ProviderApiType::Anthropic => {
                let key = resolve_api_key(config)
                    .ok_or_else(|| anyhow::anyhow!("API key not set for provider '{name}'"))?;
                Ok(Box::new(AnthropicProvider::new(key)?))
            }
            ProviderApiType::OpenAi => {
                let api_key = resolve_api_key(config);
                let base_url = config.base_url.clone();
                Ok(Box::new(OpenAiProvider::with_config(api_key, base_url)?))
            }
            ProviderApiType::Google => {
                let key = resolve_api_key(config)
                    .ok_or_else(|| anyhow::anyhow!("API key not set for provider '{name}'"))?;
                Ok(Box::new(UnifiedProvider::google(key)?))
            }
            ProviderApiType::Groq => {
                let key = resolve_api_key(config)
                    .ok_or_else(|| anyhow::anyhow!("API key not set for provider '{name}'"))?;
                Ok(Box::new(UnifiedProvider::groq(key)?))
            }
            ProviderApiType::Mistral => {
                let key = resolve_api_key(config)
                    .ok_or_else(|| anyhow::anyhow!("API key not set for provider '{name}'"))?;
                Ok(Box::new(UnifiedProvider::mistral(key)?))
            }
            ProviderApiType::Synapse => {
                let factory = self.custom_factories.get("synapse").ok_or_else(|| {
                    anyhow::anyhow!(
                        "synapse provider not available — register a factory for 'synapse' \
                         (e.g., via synapse-client with the agent-core feature)"
                    )
                })?;
                factory(name, config)
            }
            ProviderApiType::Custom(type_name) => {
                let factory = self.custom_factories.get(type_name).ok_or_else(|| {
                    anyhow::anyhow!("no factory registered for custom provider type '{type_name}'")
                })?;
                factory(name, config)
            }
        }
    }
}

/// Resolve API key for a provider config
///
/// Checks the environment variable first, then falls back to the
/// direct value
#[must_use]
pub fn resolve_api_key(config: &ProviderConfig) -> Option<String> {
    // First try env var
    if let Some(env_name) = &config.api_key_env {
        if let Ok(key) = std::env::var(env_name) {
            return Some(key);
        }
    }
    // Fall back to direct key
    config.api_key.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_api_key_from_direct_value() {
        let config = ProviderConfig {
            api_type: ProviderApiType::OpenAi,
            base_url: None,
            api_key_env: None,
            api_key: Some("sk-direct".to_string()),
        };
        assert_eq!(resolve_api_key(&config), Some("sk-direct".to_string()));
    }

    #[test]
    fn resolve_api_key_none_when_empty() {
        let config = ProviderConfig::default();
        assert_eq!(resolve_api_key(&config), None);
    }

    #[test]
    fn registry_custom_factory() {
        let mut registry = ProviderRegistry::new();

        // Custom type with no factory should fail
        let config = ProviderConfig {
            api_type: ProviderApiType::Custom("test".to_string()),
            ..Default::default()
        };
        assert!(registry.create_provider("test", &config).is_err());

        // After registering factory, custom type should use it
        registry.register_factory(
            "test",
            Box::new(|_name, _config| {
                // Return a mock -- just verify the factory is called
                Err(anyhow::anyhow!("test factory called"))
            }),
        );
        let result = registry.create_provider("test", &config);
        let err = result.err().expect("factory should return error");
        assert!(err.to_string().contains("test factory called"));
    }

    #[test]
    fn registry_default_is_empty() {
        let registry = ProviderRegistry::default();
        assert!(registry.custom_factories.is_empty());
    }
}

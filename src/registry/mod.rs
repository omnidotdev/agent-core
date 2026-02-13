//! Provider registry for shared model and provider configuration
//!
//! Provides default provider definitions, model catalogs, and a
//! factory for creating provider instances with extension support
//! for consumer-specific providers

mod defaults;
mod factory;
mod types;

pub use defaults::{default_models, default_providers, detect_provider_by_prefix};
pub use factory::{ProviderFactoryFn, ProviderRegistry, resolve_api_key};
pub use types::{ModelInfo, ProviderApiType, ProviderConfig};

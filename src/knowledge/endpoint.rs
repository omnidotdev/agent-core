//! Where retrieval AI requests may send which key
//!
//! A Synapse API key authenticates against an Omni Synapse gateway and must
//! never reach a third-party model vendor. Callers pick the base URL; these
//! helpers let the embedder and condenser refuse a Synapse key bound for a
//! known vendor host, so a misrouted key fails closed instead of leaking

/// `OpenAI` API base URL (OpenAI-compatible root, including `/v1`)
pub const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";

/// Prefix every Synapse API key carries
pub const SYNAPSE_KEY_PREFIX: &str = "synapse_";

/// Registrable domains of third-party model vendors
///
/// A host matches when it equals one of these or is a subdomain of one
const VENDOR_DOMAINS: &[&str] = &[
    "openai.com",
    "anthropic.com",
    "googleapis.com",
    "groq.com",
    "mistral.ai",
    "openrouter.ai",
    "together.xyz",
    "together.ai",
    "moonshot.cn",
    "moonshot.ai",
    "cohere.com",
    "cohere.ai",
    "deepseek.com",
    "x.ai",
    "voyageai.com",
    "jina.ai",
];

/// Whether a key is a Synapse API key
#[must_use]
pub fn is_synapse_key(key: &str) -> bool {
    key.trim().starts_with(SYNAPSE_KEY_PREFIX)
}

/// Lowercased host of a URL, without scheme, userinfo, port, or path
fn host_of(url: &str) -> String {
    let rest = url.trim();
    let rest = rest.split_once("://").map_or(rest, |(_, r)| r);
    let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
    let host = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
    let host = if let Some(v6) = host.strip_prefix('[') {
        // IPv6 literal
        v6.split_once(']').map_or(v6, |(h, _)| h)
    } else {
        host.split(':').next().unwrap_or_default()
    };
    host.trim_end_matches('.').to_ascii_lowercase()
}

/// Whether a URL points at a known third-party model vendor
#[must_use]
pub fn is_vendor_host(url: &str) -> bool {
    let host = host_of(url);
    VENDOR_DOMAINS
        .iter()
        .any(|domain| host == *domain || host.ends_with(&format!(".{domain}")))
}

/// Refuse to send a Synapse key to a known vendor host
///
/// # Errors
///
/// Returns the offending host when `key` is a Synapse key and `base_url`
/// points at a third-party vendor
pub fn ensure_key_destination(key: &str, base_url: &str) -> Result<(), String> {
    if is_synapse_key(key) && is_vendor_host(base_url) {
        return Err(host_of(base_url));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synapse_keys_are_recognized() {
        assert!(is_synapse_key("synapse_abc"));
        assert!(is_synapse_key("  synapse_abc\n"));
        assert!(!is_synapse_key("sk-proj-abc"));
        assert!(!is_synapse_key(""));
    }

    #[test]
    fn host_parsing_strips_scheme_port_path_and_userinfo() {
        assert_eq!(host_of("https://API.openai.com:443/v1"), "api.openai.com");
        assert_eq!(host_of("http://user:pw@localhost:6000/v1"), "localhost");
        assert_eq!(
            host_of("gateway.synapse.omni.dev"),
            "gateway.synapse.omni.dev"
        );
        assert_eq!(host_of("http://[::1]:6000/v1"), "::1");
    }

    #[test]
    fn vendor_hosts_are_detected() {
        for url in [
            OPENAI_BASE_URL,
            "https://api.openai.com/v1/",
            "https://api.anthropic.com/v1",
            "https://generativelanguage.googleapis.com/v1beta/openai",
            "https://api.groq.com/openai/v1",
            "https://api.mistral.ai/v1",
            "https://openrouter.ai/api/v1",
            "https://api.together.xyz/v1",
            "https://api.moonshot.cn/v1",
            "https://api.cohere.com/v2/rerank",
            "https://api.openai.com.:443/v1",
        ] {
            assert!(is_vendor_host(url), "{url} not treated as a vendor host");
        }
    }

    #[test]
    fn synapse_and_local_hosts_are_not_vendor_hosts() {
        for url in [
            "https://gateway.synapse.omni.dev/v1",
            "http://localhost:6000/v1",
            "http://127.0.0.1:6000/v1",
            "https://synapse.example.com/v1",
            // Lookalike must not match by suffix alone
            "https://notopenai.com/v1",
        ] {
            assert!(
                !is_vendor_host(url),
                "{url} wrongly treated as a vendor host"
            );
        }
    }

    #[test]
    fn every_default_remote_provider_is_a_vendor_host() {
        // Keep the vendor list in sync with the provider registry so a newly
        // added provider can never receive a Synapse key
        for (name, config) in crate::registry::default_providers() {
            let Some(base_url) = config.base_url else {
                continue;
            };
            let host = host_of(&base_url);
            if name == "synapse" || host == "localhost" || host == "127.0.0.1" {
                continue;
            }
            assert!(
                is_vendor_host(&base_url),
                "{name} ({base_url}) missing from VENDOR_DOMAINS"
            );
        }
    }

    #[test]
    fn synapse_key_destination_is_enforced() {
        assert!(ensure_key_destination("synapse_abc", OPENAI_BASE_URL).is_err());
        assert!(
            ensure_key_destination("synapse_abc", "https://gateway.synapse.omni.dev/v1").is_ok()
        );
        assert!(ensure_key_destination("synapse_abc", "http://localhost:6000/v1").is_ok());
        assert!(ensure_key_destination("sk-byok", OPENAI_BASE_URL).is_ok());
    }
}

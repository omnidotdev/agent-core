---
"@omnidotdev/agent-core": patch
---

`Embedder` now targets any OpenAI-compatible `/embeddings` endpoint: `Embedder::with_config(api_key, model, base_url)` takes the API root (e.g. a Synapse gateway's `/v1` with a provider-prefixed model id such as `openai/text-embedding-3-small`), and `endpoint_url()`, `base_url()`, and `model()` expose where requests go. `Embedder::new` and `with_model` still default to OpenAI. The embedder and `LlmCondenser` now refuse a Synapse API key (`synapse_` prefix) bound for a known third-party vendor host (new `EmbedderError::SynapseKeyToVendor`), so a misrouted key fails closed instead of leaking. New `knowledge::endpoint` helpers: `is_synapse_key`, `is_vendor_host`, `OPENAI_BASE_URL`

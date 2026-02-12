<div align="center">
  <h1 align="center">🤖 Agent Core</h1>

[Website](https://omni.dev) | [Docs](https://docs.omni.dev) | [Feedback](https://backfeed.omni.dev/workspaces/omni) | [Discord](https://discord.gg/omnidotdev) | [X](https://x.com/omnidotdev)

</div>

**Agent Core** is a reusable Rust library for building AI agents in the [Omni](https://omni.dev) ecosystem. It provides provider-agnostic LLM integration, conversation management, a permission system, and plan file management.

## Features

- **LLM Provider Abstraction** — BYOM (Bring Your Own Model) trait with streaming completions, supporting Anthropic, OpenAI, and a unified multi-provider backend
- **Conversation Management** — Multi-turn conversation state with serialization and persistence
- **Permission System** — Configurable ask/allow/deny presets, session caching, and TUI dialog integration for agent tool use
- **Plan Management** — Plan file storage for plan mode with project-local and global directories
- **Core Types** — Messages, roles, content blocks, tool definitions, streaming events, and usage tracking

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
agent-core = { git = "https://github.com/omnidotdev/agent-core" }
```

## Usage

### LLM Provider

Implement the `LlmProvider` trait to add support for any LLM backend:

```rust
use agent_core::provider::{CompletionRequest, CompletionStream, LlmProvider};

struct MyProvider;

#[async_trait::async_trait]
impl LlmProvider for MyProvider {
    fn name(&self) -> &'static str {
        "my-provider"
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> agent_core::error::Result<CompletionStream> {
        // Stream completion events from your LLM
        todo!()
    }
}
```

### Built-in Providers

```rust
use agent_core::providers::{AnthropicProvider, OpenAiProvider, UnifiedProvider};
```

### Conversation

```rust
use agent_core::conversation::Conversation;

let mut conv = Conversation::with_system("You are a helpful assistant.");
conv.add_user_message("Hello!");
conv.add_assistant_message("Hi there!");

// Persist to disk
conv.save(Path::new("conversation.json"))?;
```

### Permissions

```rust
use agent_core::permission::{PermissionActor, PermissionClient};

let (actor, tx) = PermissionActor::new();
let client = PermissionClient::new("session-id".into(), tx);

// Actor runs in background, client is used by tools to request permissions
tokio::spawn(actor.run());
```

## Development

```bash
cargo build   # Build
cargo test    # Run tests
cargo clippy  # Lint
```

## License

The code in this repository is licensed under MIT, &copy; [Omni LLC](https://omni.dev). See [LICENSE.md](LICENSE.md) for more information.

//! MCP (Model Context Protocol) client for stdio-based servers

mod client;
mod manager;
mod types;

pub use client::McpClient;
pub use manager::McpServerManager;
pub use types::{McpServerConfig, McpTool, McpToolResult};

//! MCP client - communicates with a single MCP server over stdio or HTTP

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{Mutex, oneshot};

use super::types::{
    InitializeResult, JsonRpcRequest, JsonRpcResponse, McpServerConfig, McpTool, McpToolResult,
    McpTransport, ToolCallResult, ToolContent, ToolsListResult,
};

/// Shared state for pending request tracking (stdio only)
type PendingMap = Arc<Mutex<HashMap<u64, oneshot::Sender<JsonRpcResponse>>>>;

/// Transport-specific state
enum TransportState {
    Stdio {
        stdin: Mutex<Option<tokio::process::ChildStdin>>,
        pending: PendingMap,
        child: Mutex<Option<Child>>,
    },
    Http {
        client: reqwest::Client,
        url: String,
    },
}

/// Client for a single MCP server (stdio or HTTP)
pub struct McpClient {
    config: McpServerConfig,
    transport: TransportState,
    next_id: AtomicU64,
    tools: Mutex<Vec<McpTool>>,
}

/// Build a reqwest client, optionally bypassing TLS certificate validation
fn build_http_client() -> Result<reqwest::Client, String> {
    let accept_invalid = std::env::var("NODE_TLS_REJECT_UNAUTHORIZED").is_ok_and(|v| v == "0");

    reqwest::Client::builder()
        .danger_accept_invalid_certs(accept_invalid)
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| format!("failed to build HTTP client: {e}"))
}

impl McpClient {
    /// Connect to an MCP server and perform the initialize handshake
    ///
    /// # Errors
    ///
    /// Returns error if the connection or initialization fails
    pub async fn start(config: McpServerConfig) -> Result<Self, String> {
        match &config.transport {
            McpTransport::Stdio { .. } => Self::start_stdio(config).await,
            McpTransport::Http { .. } => Self::start_http(config).await,
        }
    }

    /// Start a stdio-based MCP server
    async fn start_stdio(config: McpServerConfig) -> Result<Self, String> {
        let McpTransport::Stdio {
            ref command,
            ref args,
            ref env,
        } = config.transport
        else {
            unreachable!()
        };

        let mut cmd = Command::new(command);
        cmd.args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        for (key, value) in env {
            cmd.env(key, value);
        }

        let mut child = cmd
            .spawn()
            .map_err(|e| format!("failed to spawn MCP server '{}': {e}", config.name))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "failed to capture stdin".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "failed to capture stdout".to_string())?;

        let pending: PendingMap = Arc::new(Mutex::new(HashMap::new()));

        // Spawn reader task that dispatches responses to waiting callers
        let pending_clone = Arc::clone(&pending);
        let server_name = config.name.clone();
        tokio::spawn(async move {
            let reader = BufReader::new(stdout);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if line.is_empty() {
                    continue;
                }
                match serde_json::from_str::<JsonRpcResponse>(&line) {
                    Ok(resp) => {
                        if let Some(id) = resp.id {
                            let mut map = pending_clone.lock().await;
                            if let Some(tx) = map.remove(&id) {
                                let _ = tx.send(resp);
                            }
                        }
                    }
                    Err(e) => {
                        tracing::trace!(
                            server = %server_name,
                            error = %e,
                            "non-JSON-RPC line from MCP server"
                        );
                    }
                }
            }
            tracing::debug!(server = %server_name, "MCP server stdout closed");
        });

        let client = Self {
            config,
            transport: TransportState::Stdio {
                stdin: Mutex::new(Some(stdin)),
                pending,
                child: Mutex::new(Some(child)),
            },
            next_id: AtomicU64::new(1),
            tools: Mutex::new(Vec::new()),
        };

        client.initialize_and_fetch_tools().await?;
        Ok(client)
    }

    /// Start an HTTP-based MCP client
    async fn start_http(config: McpServerConfig) -> Result<Self, String> {
        let McpTransport::Http { ref url } = config.transport else {
            unreachable!()
        };
        let url = url.clone();

        let http_client = build_http_client()?;

        let client = Self {
            config,
            transport: TransportState::Http {
                client: http_client,
                url,
            },
            next_id: AtomicU64::new(1),
            tools: Mutex::new(Vec::new()),
        };

        client.initialize_and_fetch_tools().await?;
        Ok(client)
    }

    /// Perform MCP initialize handshake and fetch tools (shared by both transports)
    async fn initialize_and_fetch_tools(&self) -> Result<(), String> {
        let init_result = self
            .send_request::<InitializeResult>(
                "initialize",
                Some(serde_json::json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "agent-core",
                        "version": env!("CARGO_PKG_VERSION")
                    }
                })),
            )
            .await?;

        tracing::info!(
            server = %self.config.name,
            protocol = %init_result.protocol_version,
            server_name = ?init_result.server_info.as_ref().map(|s| &s.name),
            "MCP server initialized"
        );

        self.send_notification("notifications/initialized", None)
            .await?;

        self.refresh_tools().await?;
        Ok(())
    }

    /// Fetch the tool list from the server and cache it
    ///
    /// # Errors
    ///
    /// Returns error if the tools/list request fails
    pub async fn refresh_tools(&self) -> Result<(), String> {
        let result = self
            .send_request::<ToolsListResult>("tools/list", None)
            .await?;

        let tools: Vec<McpTool> = result
            .tools
            .into_iter()
            .map(|t| McpTool {
                name: t.name,
                description: t.description,
                input_schema: t.input_schema,
                server_name: self.config.name.clone(),
            })
            .collect();

        tracing::info!(
            server = %self.config.name,
            count = tools.len(),
            "fetched MCP tools"
        );

        *self.tools.lock().await = tools;
        Ok(())
    }

    /// Get cached tool definitions
    pub async fn tools(&self) -> Vec<McpTool> {
        self.tools.lock().await.clone()
    }

    /// Call a tool on this server
    ///
    /// # Errors
    ///
    /// Returns error if the tool call fails
    pub async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<McpToolResult, String> {
        let result = self
            .send_request::<ToolCallResult>(
                "tools/call",
                Some(serde_json::json!({
                    "name": name,
                    "arguments": arguments
                })),
            )
            .await?;

        let text = result
            .content
            .into_iter()
            .filter_map(|c| match c {
                ToolContent::Text { text } => Some(text),
                ToolContent::Other => None,
            })
            .collect::<Vec<_>>()
            .join("\n");

        Ok(McpToolResult {
            text,
            is_error: result.is_error,
        })
    }

    /// Stop the MCP server
    pub async fn stop(&self) {
        match &self.transport {
            TransportState::Stdio { stdin, child, .. } => {
                let _ = stdin.lock().await.take();
                let child = child.lock().await.take();
                if let Some(mut child) = child {
                    tokio::select! {
                        _ = child.wait() => {}
                        () = tokio::time::sleep(std::time::Duration::from_secs(5)) => {
                            let _ = child.kill().await;
                        }
                    }
                }
            }
            TransportState::Http { .. } => {
                // HTTP transport has no persistent connection to clean up
            }
        }
    }

    /// Send a JSON-RPC request and wait for the response
    async fn send_request<T: serde::de::DeserializeOwned>(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<T, String> {
        match &self.transport {
            TransportState::Stdio { .. } => self.send_request_stdio(method, params).await,
            TransportState::Http { client, url } => {
                Self::send_request_http(client, url, &self.next_id, method, params).await
            }
        }
    }

    /// Send a JSON-RPC request via stdio
    async fn send_request_stdio<T: serde::de::DeserializeOwned>(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<T, String> {
        let TransportState::Stdio { stdin, pending, .. } = &self.transport else {
            unreachable!()
        };

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);

        let request = JsonRpcRequest {
            jsonrpc: "2.0",
            id,
            method: method.to_string(),
            params,
        };

        let (tx, rx) = oneshot::channel();
        pending.lock().await.insert(id, tx);

        let mut line =
            serde_json::to_string(&request).map_err(|e| format!("serialize error: {e}"))?;
        line.push('\n');

        {
            let mut stdin_lock = stdin.lock().await;
            let writer = stdin_lock
                .as_mut()
                .ok_or_else(|| "MCP server stdin closed".to_string())?;
            writer
                .write_all(line.as_bytes())
                .await
                .map_err(|e| format!("write to MCP server failed: {e}"))?;
            writer
                .flush()
                .await
                .map_err(|e| format!("flush to MCP server failed: {e}"))?;
        }

        let response = tokio::time::timeout(std::time::Duration::from_secs(30), rx)
            .await
            .map_err(|_| format!("MCP request '{method}' timed out after 30s"))?
            .map_err(|_| "MCP response channel dropped".to_string())?;

        Self::parse_response(response)
    }

    /// Send a JSON-RPC request via HTTP
    async fn send_request_http<T: serde::de::DeserializeOwned>(
        client: &reqwest::Client,
        url: &str,
        next_id: &AtomicU64,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<T, String> {
        let id = next_id.fetch_add(1, Ordering::Relaxed);

        let request = JsonRpcRequest {
            jsonrpc: "2.0",
            id,
            method: method.to_string(),
            params,
        };

        let response = client
            .post(url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream")
            .json(&request)
            .send()
            .await
            .map_err(|e| format!("HTTP request to MCP server failed: {e}"))?;

        if !response.status().is_success() {
            return Err(format!(
                "MCP HTTP error: {} {}",
                response.status(),
                response
                    .text()
                    .await
                    .unwrap_or_else(|_| "no body".to_string())
            ));
        }

        let rpc_response: JsonRpcResponse = response
            .json()
            .await
            .map_err(|e| format!("MCP HTTP response parse error: {e}"))?;

        Self::parse_response(rpc_response)
    }

    /// Parse a JSON-RPC response into the expected type
    fn parse_response<T: serde::de::DeserializeOwned>(
        response: JsonRpcResponse,
    ) -> Result<T, String> {
        if let Some(error) = response.error {
            return Err(format!("MCP error ({}): {}", error.code, error.message));
        }

        let result = response
            .result
            .ok_or_else(|| "MCP response missing result".to_string())?;

        serde_json::from_value(result).map_err(|e| format!("MCP result parse error: {e}"))
    }

    /// Send a JSON-RPC notification (no response expected)
    async fn send_notification(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<(), String> {
        let msg = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params.unwrap_or_else(|| serde_json::json!({}))
        });

        match &self.transport {
            TransportState::Stdio { stdin, .. } => {
                let mut line =
                    serde_json::to_string(&msg).map_err(|e| format!("serialize error: {e}"))?;
                line.push('\n');

                let mut stdin_lock = stdin.lock().await;
                let writer = stdin_lock
                    .as_mut()
                    .ok_or_else(|| "MCP server stdin closed".to_string())?;
                writer
                    .write_all(line.as_bytes())
                    .await
                    .map_err(|e| format!("write notification failed: {e}"))?;
                writer
                    .flush()
                    .await
                    .map_err(|e| format!("flush notification failed: {e}"))?;
            }
            TransportState::Http { client, url } => {
                // Fire-and-forget for HTTP notifications
                let _ = client
                    .post(url)
                    .header("Content-Type", "application/json")
                    .header("Accept", "application/json, text/event-stream")
                    .json(&msg)
                    .send()
                    .await;
            }
        }

        Ok(())
    }
}

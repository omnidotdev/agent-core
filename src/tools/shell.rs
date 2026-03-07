//! Sandboxed shell execution tool.
//!
//! Runs commands via `/bin/sh -c`, captures stdout/stderr,
//! enforces timeouts, and augments `PATH`.

use std::path::PathBuf;
use std::time::Duration;

use tokio::process::Command;

/// Default command timeout in seconds.
const DEFAULT_TIMEOUT_SECS: u64 = 120;

/// Maximum allowed timeout in seconds.
const MAX_TIMEOUT_SECS: u64 = 600;

/// Error from shell execution.
#[derive(Debug, thiserror::Error)]
pub enum ShellError {
    /// Invalid arguments.
    #[error("invalid arguments: {0}")]
    InvalidArgs(String),
    /// Missing required parameter.
    #[error("{0}")]
    MissingParam(String),
    /// Failed to spawn process.
    #[error("failed to spawn: {0}")]
    Spawn(#[from] std::io::Error),
    /// Command timed out.
    #[error("command timed out after {0}s")]
    Timeout(u64),
}

/// Result of a shell command execution.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ShellOutput {
    /// Process exit code (-1 if killed).
    pub exit_code: i32,
    /// Captured stdout.
    pub stdout: String,
    /// Captured stderr.
    pub stderr: String,
}

/// Sandboxed shell execution tool.
#[derive(Debug, Clone)]
pub struct ShellTool {
    /// Working directory for command execution.
    working_dir: PathBuf,
    /// Additional PATH entries prepended to the environment.
    extra_path: Vec<PathBuf>,
}

impl ShellTool {
    /// Create a new shell tool with the given working directory and extra PATH entries.
    #[must_use]
    pub fn new(working_dir: PathBuf, extra_path: Vec<PathBuf>) -> Self {
        Self {
            working_dir,
            extra_path,
        }
    }

    /// Build the augmented PATH value.
    fn augmented_path(&self) -> String {
        let system_path = std::env::var("PATH").unwrap_or_default();
        let extra: Vec<String> = self
            .extra_path
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();

        if extra.is_empty() {
            system_path
        } else {
            format!("{}:{system_path}", extra.join(":"))
        }
    }

    /// Execute a shell command.
    ///
    /// # Errors
    ///
    /// Returns error if the command is empty, fails to spawn, or times out.
    pub async fn execute(
        &self,
        command: &str,
        timeout_secs: Option<u64>,
    ) -> Result<ShellOutput, ShellError> {
        let command = command.trim();
        if command.is_empty() {
            return Err(ShellError::MissingParam(
                "`command` parameter is required".to_string(),
            ));
        }

        let timeout_secs = timeout_secs
            .unwrap_or(DEFAULT_TIMEOUT_SECS)
            .min(MAX_TIMEOUT_SECS);

        tracing::debug!(command = %command, timeout_secs, "shell: executing command");

        let child = {
            let mut cmd = Command::new("/bin/sh");
            cmd.arg("-c")
                .arg(command)
                .current_dir(&self.working_dir)
                .env("PATH", self.augmented_path())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .kill_on_drop(true);

            #[cfg(unix)]
            cmd.process_group(0);

            cmd.spawn()?
        };

        let result =
            tokio::time::timeout(Duration::from_secs(timeout_secs), child.wait_with_output())
                .await;

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
                let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
                let exit_code = output.status.code().unwrap_or(-1);

                tracing::debug!(
                    exit_code,
                    stdout_len = stdout.len(),
                    stderr_len = stderr.len(),
                    "shell: command completed"
                );

                Ok(ShellOutput {
                    exit_code,
                    stdout,
                    stderr,
                })
            }
            Ok(Err(e)) => Err(ShellError::Spawn(e)),
            Err(_) => {
                tracing::warn!(command = %command, timeout_secs, "shell: command timed out");
                Err(ShellError::Timeout(timeout_secs))
            }
        }
    }
}

impl Default for ShellTool {
    fn default() -> Self {
        let home = std::env::var("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/tmp"));

        let extra_path = vec![
            home.join(".bun/bin"),
            home.join(".local/bin"),
            home.join(".cargo/bin"),
        ];

        Self::new(home, extra_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tool() -> ShellTool {
        ShellTool::new(PathBuf::from("/tmp"), vec![])
    }

    #[tokio::test]
    async fn simple_command_returns_stdout() {
        let tool = make_tool();
        let result = tool.execute("echo hello", None).await.unwrap();
        assert_eq!(result.exit_code, 0);
        assert!(result.stdout.contains("hello"));
    }

    #[tokio::test]
    async fn stderr_is_captured() {
        let tool = make_tool();
        let result = tool.execute("echo err >&2", None).await.unwrap();
        assert!(result.stderr.contains("err"));
    }

    #[tokio::test]
    async fn exit_code_is_reported() {
        let tool = make_tool();
        let result = tool.execute("exit 42", None).await.unwrap();
        assert_eq!(result.exit_code, 42);
    }

    #[tokio::test]
    async fn timeout_is_enforced() {
        let tool = make_tool();
        let result = tool.execute("sleep 10", Some(1)).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ShellError::Timeout(1)));
    }

    #[tokio::test]
    async fn empty_command_is_rejected() {
        let tool = make_tool();
        assert!(tool.execute("", None).await.is_err());
        assert!(tool.execute("   ", None).await.is_err());
    }
}

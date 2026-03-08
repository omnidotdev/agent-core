//! Skill registry trait for pluggable discovery backends.

use super::types::Skill;

/// Skill registry interface
pub trait SkillLookup: Send + Sync {
    /// Get a skill by ID
    fn get(&self, id: &str) -> Option<&Skill>;

    /// List all discovered skills
    fn list(&self) -> Vec<&Skill>;

    /// Check if a skill exists
    fn contains(&self, id: &str) -> bool;
}

//! Shared skill types for agent products.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Skill metadata parsed from YAML frontmatter
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SkillMetadata {
    /// Display name
    #[serde(default)]
    pub name: Option<String>,
    /// Description of what the skill does
    #[serde(default)]
    pub description: Option<String>,
    /// Version string
    #[serde(default)]
    pub version: Option<String>,
    /// Author
    #[serde(default)]
    pub author: Option<String>,
    /// Tags for categorization
    #[serde(default)]
    pub tags: Vec<String>,
    /// Whether the skill is always active
    #[serde(default)]
    pub always: bool,
    /// Whether the user can invoke via slash command
    #[serde(default)]
    pub user_invocable: bool,
}

/// Where a skill was discovered from
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkillSource {
    /// Local filesystem
    Local,
    /// Bundled with the product
    Bundled,
    /// From Manifold registry
    Manifold {
        namespace: String,
        repository: String,
    },
    /// From a plugin
    Plugin,
}

/// A discovered skill
#[derive(Debug, Clone)]
pub struct Skill {
    /// Unique identifier (usually the directory name)
    pub id: String,
    /// Parsed metadata
    pub metadata: SkillMetadata,
    /// Raw markdown content
    pub content: String,
    /// Where this skill came from
    pub source: SkillSource,
    /// Filesystem location (if applicable)
    pub location: Option<PathBuf>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skill_creation() {
        let skill = Skill {
            id: "test-skill".to_string(),
            metadata: SkillMetadata {
                name: Some("Test Skill".to_string()),
                description: Some("A test skill".to_string()),
                ..Default::default()
            },
            content: "# Test\nDo something.".to_string(),
            source: SkillSource::Local,
            location: Some(PathBuf::from("/path/to/skill")),
        };

        assert_eq!(skill.id, "test-skill");
        assert_eq!(skill.metadata.name.as_deref(), Some("Test Skill"));
        assert_eq!(skill.source, SkillSource::Local);
    }

    #[test]
    fn metadata_defaults() {
        let meta = SkillMetadata::default();
        assert!(meta.name.is_none());
        assert!(meta.tags.is_empty());
        assert!(!meta.always);
        assert!(!meta.user_invocable);
    }

    #[test]
    fn metadata_serialization() {
        let meta = SkillMetadata {
            name: Some("hello".to_string()),
            always: true,
            ..Default::default()
        };
        let json = serde_json::to_string(&meta).unwrap();
        assert!(json.contains("\"always\":true"));
        let deserialized: SkillMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name.as_deref(), Some("hello"));
        assert!(deserialized.always);
    }

    #[test]
    fn skill_source_equality() {
        assert_eq!(SkillSource::Local, SkillSource::Local);
        assert_eq!(SkillSource::Bundled, SkillSource::Bundled);
        assert_ne!(SkillSource::Local, SkillSource::Bundled);
        assert_eq!(
            SkillSource::Manifold {
                namespace: "omni".to_string(),
                repository: "tools".to_string(),
            },
            SkillSource::Manifold {
                namespace: "omni".to_string(),
                repository: "tools".to_string(),
            },
        );
    }
}

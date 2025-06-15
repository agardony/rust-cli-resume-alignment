//! Configuration management for the resume aligner

use crate::error::{Result, ResumeAlignerError};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub models: ModelConfig,
    pub processing: ProcessingConfig,
    pub scoring: ScoringConfig,
    pub output: OutputConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub models_dir: PathBuf,
    pub default_embedding_model: String,
    pub default_llm_model: String,
    pub available_models: Vec<AvailableModel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailableModel {
    pub name: String,
    pub repo_id: String,
    pub model_type: ModelType,
    pub size_mb: u64,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    Embedding,
    LLM,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub max_tokens: usize,
    pub enable_caching: bool,
    pub batch_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringConfig {
    pub embedding_weight: f32,
    pub keyword_weight: f32,
    pub llm_weight: f32,
    pub min_similarity_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    pub format: OutputFormat,
    pub detailed: bool,
    pub include_recommendations: bool,
    pub color_output: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputFormat {
    Console,
    Json,
    Markdown,
    Html,
    Pdf,
}

impl Default for Config {
    fn default() -> Self {
        let models_dir = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".resume-aligner")
            .join("models");

        Self {
            models: ModelConfig {
                models_dir,
                default_embedding_model: "minishlab/M2V_base_output".to_string(),
                default_llm_model: "microsoft/Phi-3-mini-4k-instruct".to_string(),
                available_models: vec![
                    // Model2Vec embedding models
                    AvailableModel {
                        name: "m2v-base".to_string(),
                        repo_id: "minishlab/M2V_base_output".to_string(),
                        model_type: ModelType::Embedding,
                        size_mb: 90,
                        description: "Fast Model2Vec base embeddings model".to_string(),
                    },
                    AvailableModel {
                        name: "m2v-large".to_string(),
                        repo_id: "minishlab/M2V_large_output".to_string(),
                        model_type: ModelType::Embedding,
                        size_mb: 250,
                        description: "High-quality Model2Vec large embeddings model".to_string(),
                    },
                    // LLM models
                    AvailableModel {
                        name: "phi-3-mini".to_string(),
                        repo_id: "microsoft/Phi-3-mini-4k-instruct".to_string(),
                        model_type: ModelType::LLM,
                        size_mb: 2300,
                        description: "Small but capable model for development".to_string(),
                    },
                    AvailableModel {
                        name: "llama-3.1-8b".to_string(),
                        repo_id: "meta-llama/Llama-3.1-8B-Instruct".to_string(),
                        model_type: ModelType::LLM,
                        size_mb: 8000,
                        description: "Production-ready model with excellent performance".to_string(),
                    },
                ],
            },
            processing: ProcessingConfig {
                chunk_size: 512,
                chunk_overlap: 50,
                max_tokens: 4096,
                enable_caching: true,
                batch_size: 32,
            },
            scoring: ScoringConfig {
                embedding_weight: 0.3,
                keyword_weight: 0.4,
                llm_weight: 0.3,
                min_similarity_threshold: 0.1,
            },
            output: OutputConfig {
                format: OutputFormat::Console,
                detailed: false,
                include_recommendations: true,
                color_output: true,
            },
        }
    }
}

impl Config {
    pub fn load() -> Result<Self> {
        let config_path = Self::config_path();
        
        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)?;
            let config: Config = toml::from_str(&content)
                .map_err(|e| ResumeAlignerError::Configuration(format!("Failed to parse config: {}", e)))?;
            Ok(config)
        } else {
            let config = Self::default();
            config.save()?;
            Ok(config)
        }
    }

    pub fn save(&self) -> Result<()> {
        let config_path = Self::config_path();
        
        if let Some(parent) = config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        let content = toml::to_string_pretty(self)
            .map_err(|e| ResumeAlignerError::Configuration(format!("Failed to serialize config: {}", e)))?;
        
        std::fs::write(&config_path, content)?;
        Ok(())
    }

    fn config_path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| dirs::home_dir().unwrap_or_else(|| PathBuf::from(".")))
            .join("resume-aligner")
            .join("config.toml")
    }

    pub fn models_dir(&self) -> &PathBuf {
        &self.models.models_dir
    }
    
    pub fn get_models_dir(&self) -> PathBuf {
        self.models.models_dir.clone()
    }

    pub fn ensure_models_dir(&self) -> Result<()> {
        std::fs::create_dir_all(&self.models.models_dir)?;
        Ok(())
    }

    pub fn get_model_by_name(&self, name: &str) -> Option<&AvailableModel> {
        self.models.available_models.iter().find(|m| m.name == name)
    }

    pub fn list_embedding_models(&self) -> Vec<&AvailableModel> {
        self.models.available_models
            .iter()
            .filter(|m| matches!(m.model_type, ModelType::Embedding))
            .collect()
    }

    pub fn list_llm_models(&self) -> Vec<&AvailableModel> {
        self.models.available_models
            .iter()
            .filter(|m| matches!(m.model_type, ModelType::LLM))
            .collect()
    }
}


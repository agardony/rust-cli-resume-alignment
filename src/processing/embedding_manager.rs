//! Embedding model management for downloading and managing Model2Vec models

use crate::error::{Result, ResumeAlignerError};
use hf_hub::api::tokio::Api;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use tokio::fs;

/// Information about an available embedding model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingModelInfo {
    pub name: String,
    pub repo_id: String,
    pub size_mb: u64,
    pub description: String,
    pub model_type: EmbeddingModelType,
    pub dimensions: u32,
    pub capabilities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmbeddingModelType {
    Model2Vec,
    Potion,
}

/// Manager for embedding models - handles download, caching, and selection
pub struct EmbeddingModelManager {
    models_dir: PathBuf,
    available_models: HashMap<String, EmbeddingModelInfo>,
    downloaded_models: HashSet<String>,
    api: Api,
}

impl EmbeddingModelManager {
    /// Create a new embedding model manager
    pub async fn new(models_dir: PathBuf) -> Result<Self> {
        // Ensure models directory exists
        if !models_dir.exists() {
            fs::create_dir_all(&models_dir).await.map_err(|e| {
                ResumeAlignerError::ModelError(format!(
                    "Failed to create models directory: {}", e
                ))
            })?;
        }
        
        let api = Api::new().map_err(|e| {
            ResumeAlignerError::ModelError(format!("Failed to initialize HF API: {}", e))
        })?;
        
        let mut manager = Self {
            models_dir,
            available_models: HashMap::new(),
            downloaded_models: HashSet::new(),
            api,
        };
        
        // Initialize available models
        manager.init_available_models();
        
        // Scan for already downloaded models
        manager.scan_downloaded_models().await?;
        
        Ok(manager)
    }
    
    /// Initialize the list of available embedding models
    fn init_available_models(&mut self) {
        // Potion Base 8M (recommended default)
        self.available_models.insert(
            "potion-base-8M".to_string(),
            EmbeddingModelInfo {
                name: "Potion Base 8M".to_string(),
                repo_id: "minishlab/potion-base-8M".to_string(),
                size_mb: 33, // ~33MB
                description: "High-quality Model2Vec embeddings with 8M parameters".to_string(),
                model_type: EmbeddingModelType::Potion,
                dimensions: 256,
                capabilities: vec![
                    "text-embeddings".to_string(),
                    "semantic-search".to_string(),
                    "similarity-analysis".to_string(),
                ],
            },
        );
        
        // Model2Vec Base (legacy)
        self.available_models.insert(
            "m2v-base".to_string(),
            EmbeddingModelInfo {
                name: "Model2Vec Base".to_string(),
                repo_id: "minishlab/M2V_base_output".to_string(),
                size_mb: 90, // ~90MB
                description: "Legacy Model2Vec base embeddings model".to_string(),
                model_type: EmbeddingModelType::Model2Vec,
                dimensions: 256,
                capabilities: vec![
                    "text-embeddings".to_string(),
                    "semantic-search".to_string(),
                ],
            },
        );
        
        // Model2Vec Large (for high-accuracy use cases)
        self.available_models.insert(
            "m2v-large".to_string(),
            EmbeddingModelInfo {
                name: "Model2Vec Large".to_string(),
                repo_id: "minishlab/M2V_large_output".to_string(),
                size_mb: 250, // ~250MB
                description: "High-capacity Model2Vec large embeddings model".to_string(),
                model_type: EmbeddingModelType::Model2Vec,
                dimensions: 512,
                capabilities: vec![
                    "text-embeddings".to_string(),
                    "semantic-search".to_string(),
                    "high-accuracy".to_string(),
                ],
            },
        );
    }
    
    /// Scan for already downloaded models
    async fn scan_downloaded_models(&mut self) -> Result<()> {
        let mut entries = fs::read_dir(&self.models_dir).await.map_err(|e| {
            ResumeAlignerError::ModelError(format!(
                "Failed to scan models directory: {}", e
            ))
        })?;
        
        while let Some(entry) = entries.next_entry().await.map_err(|e| {
            ResumeAlignerError::ModelError(format!(
                "Failed to read directory entry: {}", e
            ))
        })? {
            if entry.file_type().await.map_err(|e| {
                ResumeAlignerError::ModelError(format!(
                    "Failed to get file type: {}", e
                ))
            })?.is_dir() {
                let model_name = entry.file_name().to_string_lossy().to_string();
                
                // Check if this is a valid embedding model directory
                if self.is_valid_embedding_model_directory(&entry.path()).await? {
                    self.downloaded_models.insert(model_name);
                }
            }
        }
        
        Ok(())
    }
    
    /// Check if a directory contains a valid embedding model
    async fn is_valid_embedding_model_directory(&self, path: &Path) -> Result<bool> {
        // Check for essential Model2Vec files
        let required_files = ["tokenizer.json"];
        
        // Check for either model.onnx or model.safetensors
        let model_file_exists = fs::metadata(path.join("model.onnx")).await.is_ok() ||
                               fs::metadata(path.join("model.safetensors")).await.is_ok();
        
        if !model_file_exists {
            return Ok(false);
        }
        
        for file in &required_files {
            let file_path = path.join(file);
            if !fs::metadata(&file_path).await.is_ok() {
                return Ok(false);
            }
        }
        
        Ok(true)
    }
    
    /// Download an embedding model from Hugging Face Hub
    pub async fn download_model(&mut self, model_id: &str) -> Result<PathBuf> {
        let model_info = self.available_models.get(model_id)
            .ok_or_else(|| ResumeAlignerError::ModelError(
                format!("Unknown embedding model: {}", model_id)
            ))?;
        
        let model_dir = self.models_dir.join(model_id);
        
        // Check if already downloaded
        if self.downloaded_models.contains(model_id) {
            return Ok(model_dir);
        }
        
        println!("📥 Downloading embedding model: {} ({} MB)", model_info.name, model_info.size_mb);
        println!("📍 Repository: {}", model_info.repo_id);
        
        // Create model directory
        fs::create_dir_all(&model_dir).await.map_err(|e| {
            ResumeAlignerError::ModelError(format!(
                "Failed to create model directory: {}", e
            ))
        })?;
        
        let repo = self.api.repo(hf_hub::Repo::model(model_info.repo_id.clone()));
        
        // Download essential Model2Vec files
        let essential_files = [
            "model.safetensors",
            "model.onnx",
            "tokenizer.json",
            "config.json",
            "README.md",
        ];
        
        for file in &essential_files {
            match repo.get(file).await {
                Ok(file_path) => {
                    let dest_path = model_dir.join(file);
                    fs::copy(&file_path, &dest_path).await.map_err(|e| {
                        ResumeAlignerError::ModelError(format!(
                            "Failed to copy {}: {}", file, e
                        ))
                    })?;
                    println!("  ✅ Downloaded: {}", file);
                }
                Err(e) => {
                    if *file == "README.md" || *file == "config.json" || *file == "model.onnx" {
                        // These files are optional - we only need one model format (safetensors OR onnx)
                        println!("  ⚠️  Optional file {} not found: {}", file, e);
                    } else {
                        return Err(ResumeAlignerError::ModelError(format!(
                            "Failed to download required file {}: {}", file, e
                        )));
                    }
                }
            }
        }
        
        // Mark as downloaded
        self.downloaded_models.insert(model_id.to_string());
        
        println!("✅ Embedding model {} downloaded successfully!", model_info.name);
        Ok(model_dir)
    }
    
    /// Get path to a downloaded model
    pub fn get_model_path(&self, model_id: &str) -> Option<PathBuf> {
        if self.downloaded_models.contains(model_id) {
            Some(self.models_dir.join(model_id))
        } else {
            None
        }
    }
    
    /// Get or download a model, returning its path
    pub async fn ensure_model_available(&mut self, model_id: &str) -> Result<PathBuf> {
        if let Some(path) = self.get_model_path(model_id) {
            return Ok(path);
        }
        
        // Model not downloaded, download it
        self.download_model(model_id).await
    }
    
    /// List all available models
    pub fn list_available_models(&self) -> Vec<&EmbeddingModelInfo> {
        self.available_models.values().collect()
    }
    
    /// List downloaded models
    pub fn list_downloaded_models(&self) -> Vec<String> {
        self.downloaded_models.iter().cloned().collect()
    }
    
    /// Auto-select the best available embedding model
    pub async fn auto_select_model(&self) -> Result<String> {
        // Priority order: downloaded models first, then by quality/size balance
        let preferred_order = ["potion-base-8M", "m2v-base", "m2v-large"];
        
        // First, try to find a downloaded model in preference order
        for model_id in &preferred_order {
            if self.downloaded_models.contains(*model_id) {
                return Ok(model_id.to_string());
            }
        }
        
        // If no models are downloaded, recommend potion-base-8M for best quality/size ratio
        Ok("potion-base-8M".to_string())
    }
    
    /// Get model info by ID
    pub fn get_model_info(&self, model_id: &str) -> Option<&EmbeddingModelInfo> {
        self.available_models.get(model_id)
    }
    
    /// Check if a model is downloaded
    pub fn is_model_downloaded(&self, model_id: &str) -> bool {
        self.downloaded_models.contains(model_id)
    }
    
    /// Resolve model ID from various formats (repo_id, name, etc.)
    pub fn resolve_model_id(&self, input: &str) -> Option<String> {
        // First check if it's a direct model ID
        if self.available_models.contains_key(input) {
            return Some(input.to_string());
        }
        
        // Check if it matches a repo_id
        for (id, info) in &self.available_models {
            if info.repo_id == input {
                return Some(id.clone());
            }
        }
        
        // Check if it matches a name (case-insensitive)
        let input_lower = input.to_lowercase();
        for (id, info) in &self.available_models {
            if info.name.to_lowercase() == input_lower {
                return Some(id.clone());
            }
        }
        
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[tokio::test]
    async fn test_embedding_model_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let manager = EmbeddingModelManager::new(temp_dir.path().to_path_buf()).await;
        assert!(manager.is_ok());
        
        let manager = manager.unwrap();
        assert!(!manager.list_available_models().is_empty());
    }
    
    #[tokio::test]
    async fn test_auto_select_model() {
        let temp_dir = TempDir::new().unwrap();
        let manager = EmbeddingModelManager::new(temp_dir.path().to_path_buf()).await.unwrap();
        
        let selected = manager.auto_select_model().await.unwrap();
        assert_eq!(selected, "potion-base-8M");
    }
    
    #[tokio::test]
    async fn test_resolve_model_id() {
        let temp_dir = TempDir::new().unwrap();
        let manager = EmbeddingModelManager::new(temp_dir.path().to_path_buf()).await.unwrap();
        
        // Test direct ID
        assert_eq!(manager.resolve_model_id("potion-base-8M"), Some("potion-base-8M".to_string()));
        
        // Test repo_id
        assert_eq!(manager.resolve_model_id("minishlab/potion-base-8M"), Some("potion-base-8M".to_string()));
        
        // Test name
        assert_eq!(manager.resolve_model_id("Potion Base 8M"), Some("potion-base-8M".to_string()));
    }
}


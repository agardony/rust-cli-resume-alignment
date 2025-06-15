//! LLM model management for downloading and managing Hugging Face models

use crate::error::{Result, ResumeAlignerError};
use hf_hub::api::tokio::Api;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use tokio::fs;

/// Information about an available LLM model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub repo_id: String,
    pub size_mb: u64,
    pub description: String,
    pub model_type: ModelType,
    pub capabilities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    Phi3Mini,
    Llama31,
    TinyLlama,
    Qwen,
}

/// Manager for LLM models - handles download, caching, and selection
pub struct ModelManager {
    models_dir: PathBuf,
    available_models: HashMap<String, ModelInfo>,
    downloaded_models: HashSet<String>,
    api: Api,
}

impl ModelManager {
    /// Create a new model manager
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
    
    /// Initialize the list of available models
    fn init_available_models(&mut self) {
        // Microsoft Phi-3-mini (development/testing)
        self.available_models.insert(
            "phi-3-mini".to_string(),
            ModelInfo {
                name: "Phi-3-mini-4k-instruct".to_string(),
                repo_id: "microsoft/Phi-3-mini-4k-instruct".to_string(),
                size_mb: 2300, // ~2.3GB
                description: "Lightweight instruction-tuned model for development".to_string(),
                model_type: ModelType::Phi3Mini,
                capabilities: vec![
                    "instruction-following".to_string(),
                    "text-analysis".to_string(),
                    "resume-analysis".to_string(),
                ],
            },
        );
        
        // Meta Llama 3.1 8B (production)
        self.available_models.insert(
            "llama-3.1-8b".to_string(),
            ModelInfo {
                name: "Llama-3.1-8B-Instruct".to_string(),
                repo_id: "meta-llama/Llama-3.1-8B-Instruct".to_string(),
                size_mb: 8000, // ~8GB
                description: "High-quality instruction-tuned model for production".to_string(),
                model_type: ModelType::Llama31,
                capabilities: vec![
                    "instruction-following".to_string(),
                    "text-analysis".to_string(),
                    "resume-analysis".to_string(),
                    "reasoning".to_string(),
                ],
            },
        );
        
        // TinyLlama (fallback for resource-constrained environments)
        self.available_models.insert(
            "tinyllama".to_string(),
            ModelInfo {
                name: "TinyLlama-1.1B-Chat".to_string(),
                repo_id: "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string(),
                size_mb: 1100, // ~1.1GB
                description: "Ultra-lightweight model for basic analysis".to_string(),
                model_type: ModelType::TinyLlama,
                capabilities: vec![
                    "basic-analysis".to_string(),
                    "text-generation".to_string(),
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
                
                // Check if this is a valid model directory with required files
                if self.is_valid_model_directory(&entry.path()).await? {
                    self.downloaded_models.insert(model_name);
                }
            }
        }
        
        Ok(())
    }
    
    /// Check if a directory contains a valid model
    async fn is_valid_model_directory(&self, path: &Path) -> Result<bool> {
        // Check for essential model files
        let required_files = ["config.json", "tokenizer.json"];
        
        for file in &required_files {
            let file_path = path.join(file);
            if !fs::metadata(&file_path).await.is_ok() {
                return Ok(false);
            }
        }
        
        // Check for at least one model weight file
        let weight_extensions = ["safetensors", "bin"];
        let mut has_weights = false;
        
        let mut entries = fs::read_dir(path).await.map_err(|e| {
            ResumeAlignerError::ModelError(format!(
                "Failed to read model directory: {}", e
            ))
        })?;
        
        while let Some(entry) = entries.next_entry().await.map_err(|e| {
            ResumeAlignerError::ModelError(format!(
                "Failed to read directory entry: {}", e
            ))
        })? {
            let file_name = entry.file_name().to_string_lossy().to_string();
            for ext in &weight_extensions {
                if file_name.ends_with(ext) {
                    has_weights = true;
                    break;
                }
            }
            if has_weights {
                break;
            }
        }
        
        Ok(has_weights)
    }
    
    /// Download a model from Hugging Face Hub
    pub async fn download_model(&mut self, model_id: &str) -> Result<PathBuf> {
        let model_info = self.available_models.get(model_id)
            .ok_or_else(|| ResumeAlignerError::ModelError(
                format!("Unknown model: {}", model_id)
            ))?;
        
        let model_dir = self.models_dir.join(model_id);
        
        // Check if already downloaded
        if self.downloaded_models.contains(model_id) {
            return Ok(model_dir);
        }
        
        println!("📥 Downloading model: {} ({} MB)", model_info.name, model_info.size_mb);
        println!("📍 Repository: {}", model_info.repo_id);
        
        // Create model directory
        fs::create_dir_all(&model_dir).await.map_err(|e| {
            ResumeAlignerError::ModelError(format!(
                "Failed to create model directory: {}", e
            ))
        })?;
        
        let repo = self.api.repo(hf_hub::Repo::model(model_info.repo_id.clone()));
        
        // Download essential files
        let essential_files = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "generation_config.json",
        ];
        
        for file in &essential_files {
            if let Ok(file_path) = repo.get(file).await {
                let dest_path = model_dir.join(file);
                fs::copy(&file_path, &dest_path).await.map_err(|e| {
                    ResumeAlignerError::ModelError(format!(
                        "Failed to copy {}: {}", file, e
                    ))
                })?;
                println!("  ✅ Downloaded: {}", file);
            }
        }
        
        let mut weights_downloaded = false;
        
        // Try sharded safetensors first (model-xxxxx-of-xxxxx.safetensors)
        match repo.get("model.safetensors.index.json").await {
            Ok(index_path) => {
                let dest_index = model_dir.join("model.safetensors.index.json");
                fs::copy(&index_path, &dest_index).await.map_err(|e| {
                    ResumeAlignerError::ModelError(format!("Failed to copy safetensors index: {}", e))
                })?;
                println!("  ✅ Downloaded: model.safetensors.index.json");
                
                let index_content = fs::read_to_string(&dest_index).await.map_err(|e| {
                    ResumeAlignerError::ModelError(format!("Failed to read safetensors index: {}", e))
                })?;
                
                let index_json: serde_json::Value = serde_json::from_str(&index_content).map_err(|e| {
                    ResumeAlignerError::ModelError(format!("Failed to parse safetensors index: {}", e))
                })?;
                
                if let Some(weight_map) = index_json.get("weight_map").and_then(|v| v.as_object()) {
                    let mut shard_files: std::collections::HashSet<String> = std::collections::HashSet::new();
                    for filename in weight_map.values() {
                        if let Some(filename_str) = filename.as_str() {
                            shard_files.insert(filename_str.to_string());
                        }
                    }
                    
                    for shard_file in shard_files {
                        match repo.get(&shard_file).await {
                            Ok(shard_path) => {
                                let dest_shard = model_dir.join(&shard_file);
                                fs::copy(&shard_path, &dest_shard).await.map_err(|e| {
                                    ResumeAlignerError::ModelError(format!("Failed to copy shard {}: {}", shard_file, e))
                                })?;
                                println!("  ✅ Downloaded: {}", shard_file);
                            }
                            Err(e) => {
                                return Err(ResumeAlignerError::ModelError(format!("Failed to download shard {}: {}", shard_file, e)));
                            }
                        }
                    }
                    weights_downloaded = true;
                }
            }
            Err(_) => {
                match repo.get("model.safetensors").await {
                    Ok(weights_path) => {
                        let dest_path = model_dir.join("model.safetensors");
                        fs::copy(&weights_path, &dest_path).await.map_err(|e| {
                            ResumeAlignerError::ModelError(format!("Failed to copy model weights: {}", e))
                        })?;
                        println!("  ✅ Downloaded: model.safetensors");
                        weights_downloaded = true;
                    }
                    Err(_) => {
                    }
                }
            }
        }
        
        if !weights_downloaded {
            match repo.get("pytorch_model.bin").await {
                Ok(weights_path) => {
                    let dest_path = model_dir.join("pytorch_model.bin");
                    fs::copy(&weights_path, &dest_path).await.map_err(|e| {
                        ResumeAlignerError::ModelError(format!("Failed to copy model weights: {}", e))
                    })?;
                    println!("  ✅ Downloaded: pytorch_model.bin");
                }
                Err(e) => {
                    return Err(ResumeAlignerError::ModelError(format!("Failed to download model weights: {}", e)));
                }
            }
        }
        
        // Mark as downloaded
        self.downloaded_models.insert(model_id.to_string());
        
        println!("✅ Model {} downloaded successfully!", model_info.name);
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
    
    /// List all available models
    pub fn list_available_models(&self) -> Vec<&ModelInfo> {
        self.available_models.values().collect()
    }
    
    /// List downloaded models
    pub fn list_downloaded_models(&self) -> Vec<String> {
        self.downloaded_models.iter().cloned().collect()
    }
    
    /// Auto-select the best available model based on system resources and preferences
    pub async fn auto_select_model(&self) -> Result<String> {
        // Priority order: downloaded models first, then by capability
        let preferred_order = ["phi-3-mini", "tinyllama", "llama-3.1-8b"];
        
        // First, try to find a downloaded model in preference order
        for model_id in &preferred_order {
            if self.downloaded_models.contains(*model_id) {
                return Ok(model_id.to_string());
            }
        }
        
        // If no models are downloaded, recommend phi-3-mini for balance of size and quality
        Ok("phi-3-mini".to_string())
    }
    
    /// Get model info by ID
    pub fn get_model_info(&self, model_id: &str) -> Option<&ModelInfo> {
        self.available_models.get(model_id)
    }
    
    /// Check if a model is downloaded
    pub fn is_model_downloaded(&self, model_id: &str) -> bool {
        self.downloaded_models.contains(model_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[tokio::test]
    async fn test_model_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ModelManager::new(temp_dir.path().to_path_buf()).await;
        assert!(manager.is_ok());
        
        let manager = manager.unwrap();
        assert!(!manager.list_available_models().is_empty());
    }
    
    #[tokio::test]
    async fn test_auto_select_model() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ModelManager::new(temp_dir.path().to_path_buf()).await.unwrap();
        
        let selected = manager.auto_select_model().await.unwrap();
        assert_eq!(selected, "phi-3-mini");
    }
}

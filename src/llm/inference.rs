//! LLM inference engine using Candle for local model execution

use crate::error::{Result, ResumeAlignerError};
use candle_core::{Device, Tensor, IndexOp, DType};
use candle_nn::VarBuilder;
use candle_transformers::models::phi3;
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokenizers::Tokenizer;

/// Configuration for LLM inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: Option<usize>,
    pub repeat_penalty: f64,
    pub seed: Option<u64>,
    pub batch_size: usize,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.9,
            top_k: Some(50),
            repeat_penalty: 1.1,
            seed: None,
            batch_size: 1,
        }
    }
}

/// Result of LLM inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    pub text: String,
    pub token_count: usize,
    pub inference_time_ms: u64,
    pub tokens_per_second: f64,
}

/// LLM inference engine using Candle
pub struct LLMEngine {
    model: Box<dyn LLMModel>,
    tokenizer: Tokenizer,
    device: Device,
    config: InferenceConfig,
}

/// Trait for different LLM model implementations
trait LLMModel: Send + Sync {
    fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> Result<Tensor>;
    fn get_vocab_size(&self) -> usize;
}

/// Phi-3 model implementation
struct Phi3Model {
    model: phi3::Model,
    vocab_size: usize,
}

impl LLMModel for Phi3Model {
    fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> Result<Tensor> {
        self.model.forward(input_ids, start_pos).map_err(|e| {
            ResumeAlignerError::ModelError(format!("Phi-3 forward pass failed: {}", e))
        })
    }
    
    fn get_vocab_size(&self) -> usize {
        self.vocab_size
    }
}

/// Get the best available device for inference (GPU if available, CPU fallback)
pub fn get_best_device() -> Result<Device> {
    // Try CUDA first (NVIDIA GPUs on Linux/Windows)
    #[cfg(feature = "cuda")]
    {
        if let Ok(device) = Device::new_cuda(0) {
            println!("🚀 Using CUDA GPU for acceleration");
            return Ok(device);
        }
    }
    
    // Try Metal (Apple GPUs on macOS) - but some operations may fall back to CPU
    if cfg!(target_os = "macos") {
        match Device::new_metal(0) {
            Ok(device) => {
                println!("🚀 Using Metal GPU for acceleration (some ops may fallback to CPU)");
                return Ok(device);
            }
            Err(e) => {
                println!("⚠️  Metal GPU initialization failed: {}", e);
            }
        }
    }
    
    // Fallback to CPU
    println!("⚠️  No GPU available, using CPU (this will be slower)");
    Ok(Device::Cpu)
}

/// Get device with optional user override from environment variable
pub fn get_device_with_override() -> Result<Device> {
    // Check for environment variable override
    if let Ok(device_preference) = std::env::var("RESUME_ALIGNER_DEVICE") {
        match device_preference.to_lowercase().as_str() {
            "cuda" => {
                println!("🔧 Forcing CUDA usage (from environment)");
                #[cfg(feature = "cuda")]
                {
                    return Device::new_cuda(0).map_err(|e| {
                        ResumeAlignerError::ModelError(format!("Failed to initialize CUDA: {}", e))
                    });
                }
                #[cfg(not(feature = "cuda"))]
                {
                    return Err(ResumeAlignerError::ModelError(
                        "CUDA support not compiled in".to_string()
                    ));
                }
            }
            "metal" => {
                println!("🔧 Forcing Metal usage (from environment)");
                #[cfg(feature = "metal")]
                {
                    return Device::new_metal(0).map_err(|e| {
                        ResumeAlignerError::ModelError(format!("Failed to initialize Metal: {}", e))
                    });
                }
                #[cfg(not(feature = "metal"))]
                {
                    return Err(ResumeAlignerError::ModelError(
                        "Metal support not compiled in".to_string()
                    ));
                }
            }
            "cpu" => {
                println!("🔧 Forcing CPU usage (from environment)");
                return Ok(Device::Cpu);
            }
            _ => {
                println!("⚠️  Unknown device '{}', falling back to auto-detection", device_preference);
            }
        }
    }
    
    // Auto-detect if no override
    get_best_device()
}

impl LLMEngine {
    /// Load a model from the specified path
    pub async fn load_model(
        model_path: &Path,
        config: InferenceConfig,
    ) -> Result<Self> {
        println!("🔄 Loading LLM model from: {}", model_path.display());
        
        // Initialize device with GPU support
        let device = get_device_with_override()?;
        
        // Load tokenizer
        let tokenizer_path = model_path.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            ResumeAlignerError::ModelError(format!(
                "Failed to load tokenizer: {}", e
            ))
        })?;
        
        // Load model configuration
        let config_path = model_path.join("config.json");
        let config_content = std::fs::read_to_string(&config_path).map_err(|e| {
            ResumeAlignerError::ModelError(format!(
                "Failed to read model config: {}", e
            ))
        })?;
        
        let model_config: serde_json::Value = serde_json::from_str(&config_content)
            .map_err(|e| {
                ResumeAlignerError::ModelError(format!(
                    "Failed to parse model config: {}", e
                ))
            })?;
        
        // Determine model type and load accordingly
        let model_type = model_config["model_type"]
            .as_str()
            .unwrap_or("unknown");
        
        let model: Box<dyn LLMModel> = match model_type {
            "phi3" | "phi3-mini" => {
                Self::load_phi3_model(model_path, &device, &model_config)?
            }
            _ => {
                // Default to Phi-3 for now
                println!("⚠️  Unknown model type '{}', defaulting to Phi-3", model_type);
                Self::load_phi3_model(model_path, &device, &model_config)?
            }
        };
        
        println!("✅ LLM model loaded successfully!");
        
        Ok(Self {
            model,
            tokenizer,
            device,
            config,
        })
    }
    
    /// Load a Phi-3 model specifically
    fn load_phi3_model(
        model_path: &Path,
        device: &Device,
        config: &serde_json::Value,
    ) -> Result<Box<dyn LLMModel>> {
        // Parse model configuration
        let vocab_size = config["vocab_size"].as_u64().unwrap_or(32064) as usize;
        let hidden_size = config["hidden_size"].as_u64().unwrap_or(3072) as usize;
        let num_layers = config["num_hidden_layers"].as_u64().unwrap_or(32) as usize;
        let num_heads = config["num_attention_heads"].as_u64().unwrap_or(32) as usize;
        let intermediate_size = config["intermediate_size"].as_u64().unwrap_or(8192) as usize;
        
        // Load model weights - handle both single and sharded safetensors
        let mut tensors_map = std::collections::HashMap::new();
        
        // Check for sharded safetensors first
        let index_path = model_path.join("model.safetensors.index.json");
        if index_path.exists() {
            println!("  📁 Loading sharded safetensors model...");
            
            // Read the index file to get the weight mapping
            let index_content = std::fs::read_to_string(&index_path).map_err(|e| {
                ResumeAlignerError::ModelError(format!(
                    "Failed to read safetensors index: {}", e
                ))
            })?;
            
            let index_json: serde_json::Value = serde_json::from_str(&index_content)
                .map_err(|e| {
                    ResumeAlignerError::ModelError(format!(
                        "Failed to parse safetensors index: {}", e
                    ))
                })?;
            
            if let Some(weight_map) = index_json.get("weight_map").and_then(|v| v.as_object()) {
                // Collect all unique shard files
                let mut shard_files = std::collections::HashSet::new();
                for filename in weight_map.values() {
                    if let Some(filename_str) = filename.as_str() {
                        shard_files.insert(filename_str.to_string());
                    }
                }
                
                // Load each shard file
                for shard_file in shard_files {
                    let shard_path = model_path.join(&shard_file);
                    if shard_path.exists() {
                        println!("    📦 Loading shard: {}", shard_file);
                        let shard_tensors = unsafe {
                            candle_core::safetensors::MmapedSafetensors::new(&shard_path)
                                .map_err(|e| {
                                    ResumeAlignerError::ModelError(format!(
                                        "Failed to load shard {}: {}", shard_file, e
                                    ))
                                })?
                        };
                        
                        // Add all tensors from this shard to our map
                        for (tensor_name, _) in shard_tensors.tensors() {
                            let tensor_view = shard_tensors.get(&tensor_name).map_err(|e| {
                                ResumeAlignerError::ModelError(format!(
                                    "Failed to get tensor {} from shard {}: {}", tensor_name, shard_file, e
                                ))
                            })?;
                            
                            // Convert TensorView to Tensor
                            let tensor = Tensor::from_raw_buffer(
                                tensor_view.data(),
                                tensor_view.dtype().try_into().map_err(|e| {
                                    ResumeAlignerError::ModelError(format!(
                                        "Failed to convert dtype for tensor {}: {:?}", tensor_name, e
                                    ))
                                })?,
                                tensor_view.shape(),
                                device
                            ).map_err(|e| {
                                ResumeAlignerError::ModelError(format!(
                                    "Failed to create tensor {} from raw buffer: {}", tensor_name, e
                                ))
                            })?;
                            
                            tensors_map.insert(tensor_name, tensor);
                        }
                    } else {
                        return Err(ResumeAlignerError::ModelError(format!(
                            "Shard file not found: {}", shard_file
                        )));
                    }
                }
            } else {
                return Err(ResumeAlignerError::ModelError(
                    "Invalid safetensors index format: missing weight_map".to_string()
                ));
            }
        } else {
            // Try single safetensors file
            let weights_path = model_path.join("model.safetensors");
            if weights_path.exists() {
                println!("  📦 Loading single safetensors model...");
                let single_tensors = unsafe {
                    candle_core::safetensors::MmapedSafetensors::new(&weights_path)
                        .map_err(|e| {
                            ResumeAlignerError::ModelError(format!(
                                "Failed to load safetensors: {}", e
                            ))
                        })?
                };
                
                // Add all tensors to our map
                for (tensor_name, _) in single_tensors.tensors() {
                    let tensor_view = single_tensors.get(&tensor_name).map_err(|e| {
                        ResumeAlignerError::ModelError(format!(
                            "Failed to get tensor {}: {}", tensor_name, e
                        ))
                    })?;
                    
                    // Convert TensorView to Tensor
                    let tensor = Tensor::from_raw_buffer(
                        tensor_view.data(),
                        tensor_view.dtype().try_into().map_err(|e| {
                            ResumeAlignerError::ModelError(format!(
                                "Failed to convert dtype for tensor {}: {:?}", tensor_name, e
                            ))
                        })?,
                        tensor_view.shape(),
                        device
                    ).map_err(|e| {
                        ResumeAlignerError::ModelError(format!(
                            "Failed to create tensor {} from raw buffer: {}", tensor_name, e
                        ))
                    })?;
                    
                    tensors_map.insert(tensor_name, tensor);
                }
            } else {
                return Err(ResumeAlignerError::ModelError(
                    "Model weights file not found (neither sharded nor single safetensors)".to_string()
                ));
            }
        }
        
        println!("  ✅ Loaded {} tensors", tensors_map.len());
        
        let vb = VarBuilder::from_tensors(
            tensors_map, 
            DType::F32, 
            device
        );
        
        // Create Phi-3 configuration
        let phi3_config = phi3::Config {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers: num_layers,
            num_attention_heads: num_heads,
            num_key_value_heads: num_heads,
            hidden_act: candle_nn::Activation::Silu,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_scaling: None,
            bos_token_id: Some(1),
            eos_token_id: Some(32000),
        };
        
        // Load the model
        let phi3_model = phi3::Model::new(&phi3_config, vb).map_err(|e| {
            ResumeAlignerError::ModelError(format!(
                "Failed to load Phi-3 model: {}", e
            ))
        })?;
        
        Ok(Box::new(Phi3Model {
            model: phi3_model,
            vocab_size,
        }))
    }
    
    /// Generate text from a prompt
    pub async fn generate(&mut self, prompt: &str) -> Result<InferenceResult> {
        let start_time = std::time::Instant::now();
        
        // Tokenize input
        let encoding = self.tokenizer.encode(prompt, true).map_err(|e| {
            ResumeAlignerError::ModelError(format!(
                "Failed to tokenize input: {}", e
            ))
        })?;
        
        let input_ids = encoding.get_ids();
        let input_tensor = Tensor::new(input_ids, &self.device).map_err(|e| {
            ResumeAlignerError::ModelError(format!(
                "Failed to create input tensor: {}", e
            ))
        })?;
        
        // Add batch dimension if needed
        let input_tensor = if input_tensor.dims().len() == 1 {
            input_tensor.unsqueeze(0).map_err(|e| {
                ResumeAlignerError::ModelError(format!(
                    "Failed to add batch dimension: {}", e
                ))
            })?
        } else {
            input_tensor
        };
        
        // Generate tokens
        let mut generated_tokens = Vec::new();
        let mut all_tokens = input_ids.to_vec();
        let mut position = input_ids.len(); // Start position after the prompt
        
        // Initial forward pass with full prompt
        let mut logits = self.model.forward(&input_tensor, 0)?;
        
        for _ in 0..self.config.max_tokens {
            // Get last token logits
            let last_token_logits = logits
                .i((.., logits.dim(1)? - 1, ..))
                .map_err(|e| {
                    ResumeAlignerError::ModelError(format!(
                        "Failed to extract last token logits: {}", e
                    ))
                })?;
            
            // Apply temperature
            let scaled_logits = if self.config.temperature != 1.0 {
                (&last_token_logits / self.config.temperature).map_err(|e| {
                    ResumeAlignerError::ModelError(format!(
                        "Failed to apply temperature: {}", e
                    ))
                })?
            } else {
                last_token_logits
            };
            
            // Sample next token (simplified - using argmax for now)
            let argmax_result = scaled_logits
                .argmax(candle_core::D::Minus1)
                .map_err(|e| {
                    ResumeAlignerError::ModelError(format!(
                        "Failed to sample next token: {}", e
                    ))
                })?;
            
            // Handle the case where argmax returns a tensor with dimensions
            let next_token_id = if argmax_result.dims().len() == 0 {
                // Scalar case
                argmax_result.to_scalar::<u32>().map_err(|e| {
                    ResumeAlignerError::ModelError(format!(
                        "Failed to convert token to scalar: {}", e
                    ))
                })?
            } else {
                // Non-scalar case - get the first element
                argmax_result.i(0).map_err(|e| {
                    ResumeAlignerError::ModelError(format!(
                        "Failed to index argmax result: {}", e
                    ))
                })?.to_scalar::<u32>().map_err(|e| {
                    ResumeAlignerError::ModelError(format!(
                        "Failed to convert indexed token to scalar: {}", e
                    ))
                })?
            };
            
            generated_tokens.push(next_token_id);
            all_tokens.push(next_token_id);
            
            // Check for EOS token
            if next_token_id == 32000 || next_token_id == 2 { // Common EOS token IDs
                break;
            }
            
            // Prepare input for next iteration - only use the new token
            let next_token_tensor = Tensor::new(&[next_token_id], &self.device)
                .map_err(|e| {
                    ResumeAlignerError::ModelError(format!(
                        "Failed to create next token tensor: {}", e
                    ))
                })?
                .unsqueeze(0)
                .map_err(|e| {
                    ResumeAlignerError::ModelError(format!(
                        "Failed to add batch dimension to next token: {}", e
                    ))
                })?;
            
            // Forward pass with just the new token and current position
            logits = self.model.forward(&next_token_tensor, position)?;
            position += 1; // Increment position for next iteration
        }
        
        // Decode generated tokens
        let generated_text = self.tokenizer.decode(&generated_tokens, true)
            .map_err(|e| {
                ResumeAlignerError::ModelError(format!(
                    "Failed to decode generated tokens: {}", e
                ))
            })?;
        
        let inference_time = start_time.elapsed();
        let tokens_per_second = generated_tokens.len() as f64 / inference_time.as_secs_f64();
        
        Ok(InferenceResult {
            text: generated_text.trim().to_string(),
            token_count: generated_tokens.len(),
            inference_time_ms: inference_time.as_millis() as u64,
            tokens_per_second,
        })
    }
    
    /// Generate text from multiple prompts in batch
    pub async fn batch_generate(&mut self, prompts: &[String]) -> Result<Vec<InferenceResult>> {
        // For now, process sequentially. Can be optimized for true batching later
        let mut results = Vec::new();
        
        for prompt in prompts {
            let result = self.generate(prompt).await?;
            results.push(result);
        }
        
        Ok(results)
    }
    
    /// Get inference configuration
    pub fn get_config(&self) -> &InferenceConfig {
        &self.config
    }
    
    /// Update inference configuration
    pub fn update_config(&mut self, config: InferenceConfig) {
        self.config = config;
    }
    
    /// Get model information
    pub fn get_model_info(&self) -> ModelInferenceInfo {
        ModelInferenceInfo {
            vocab_size: self.model.get_vocab_size(),
            device: format!("{:?}", self.device),
            max_tokens: self.config.max_tokens,
            temperature: self.config.temperature,
        }
    }
}

/// Information about the loaded model for inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInferenceInfo {
    pub vocab_size: usize,
    pub device: String,
    pub max_tokens: usize,
    pub temperature: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_inference_config_default() {
        let config = InferenceConfig::default();
        assert_eq!(config.max_tokens, 512);
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.batch_size, 1);
    }
    
    #[test]
    fn test_inference_result_creation() {
        let result = InferenceResult {
            text: "Test output".to_string(),
            token_count: 5,
            inference_time_ms: 100,
            tokens_per_second: 50.0,
        };
        
        assert_eq!(result.text, "Test output");
        assert_eq!(result.token_count, 5);
    }
}

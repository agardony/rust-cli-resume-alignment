//! LLM inference engine using Candle for local model execution

use crate::error::{Result, ResumeAlignerError};
use candle_core::{Device, Tensor, IndexOp, DType};
use candle_nn::VarBuilder;
use candle_transformers::models::{phi3, llama};
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
    fn reset_kv_cache(&mut self) -> Result<()>;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

/// Phi-4 model implementation
struct Phi4Model {
    model: phi3::Model,
    vocab_size: usize,
}

impl LLMModel for Phi4Model {
    fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> Result<Tensor> {
        self.model.forward(input_ids, start_pos).map_err(|e| {
            ResumeAlignerError::ModelError(format!("Phi-4 forward pass failed: {}", e))
        })
    }
    
    fn get_vocab_size(&self) -> usize {
        self.vocab_size
    }
    
    fn reset_kv_cache(&mut self) -> Result<()> {
        // Phi-4 doesn't use explicit cache in our implementation
        Ok(())
    }
    
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Llama model implementation
struct LlamaModel {
    model: llama::Llama,
    cache: llama::Cache,
    vocab_size: usize,
    config: llama::Config,
    device: Device,
}

impl LlamaModel {
    /// Reset the KV cache for a fresh generation session
    pub fn reset_cache(&mut self) -> Result<()> {
        self.cache = llama::Cache::new(true, DType::F32, &self.config, &self.device)
            .map_err(|e| ResumeAlignerError::ModelError(format!("Failed to reset cache: {}", e)))?;
        Ok(())
    }
}

impl LLMModel for LlamaModel {
    fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> Result<Tensor> {
        self.model.forward(input_ids, start_pos, &mut self.cache).map_err(|e| {
            ResumeAlignerError::ModelError(format!("Llama forward pass failed: {}", e))
        })
    }
    
    fn get_vocab_size(&self) -> usize {
        self.vocab_size
    }
    
    fn reset_kv_cache(&mut self) -> Result<()> {
        self.reset_cache()
    }
    
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

fn is_verbose_mode() -> bool {
    match std::env::var("RESUME_ALIGNER_VERBOSE") {
        Ok(val) => val == "1" || val.to_lowercase() == "true",
        Err(_) => false,
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
        
        // Also check architecture field which is more reliable
        let architecture = model_config["architectures"]
            .as_array()
            .and_then(|arr| arr.first())
            .and_then(|v| v.as_str())
            .unwrap_or("");
        
        let model: Box<dyn LLMModel> = match (model_type, architecture) {
            ("phi", _) | (_, "PhiForCausalLM") | ("phi4", _) | (_, "Phi4ForCausalLM") | (_, "Phi3ForCausalLM") => {
                println!("🔧 Loading Phi model (architecture: {})", architecture);
                Self::load_phi4_model(model_path, &device, &model_config)?
            }
            ("llama", _) | (_, "LlamaForCausalLM") => {
                println!("🔧 Loading Llama model");
                Self::load_llama_model(model_path, &device, &model_config)?
            }
            _ => {
                println!("⚠️  Unknown model type '{}'/'architecture' '{}', trying Phi first then Llama", model_type, architecture);
                // Try Phi first (since phi-4-mini is a common model), then fallback to Llama
                if let Ok(model) = Self::load_phi4_model(model_path, &device, &model_config) {
                    model
                } else {
                    println!("⚠️  Phi loading failed, falling back to Llama");
                    Self::load_llama_model(model_path, &device, &model_config)?
                }
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

    /// Check if token is end-of-sequence
    fn is_eos_token(&self, token: u32) -> bool {
        // Common EOS tokens for different models
        // phi-4-mini uses token 199999 as EOS (<|endoftext|>)
        // Other models use different tokens
        matches!(token, 2 | 32000 | 32001 | 32007 | 128001 | 128009 | 199999 | 200020)
    }
    
    /// Load a Phi-4 model specifically
    fn load_phi4_model(
        model_path: &Path,
        device: &Device,
        config: &serde_json::Value,
    ) -> Result<Box<dyn LLMModel>> {
        // Parse model configuration
        let vocab_size = config["vocab_size"].as_u64().unwrap_or(200064) as usize;
        let _hidden_size = config["hidden_size"].as_u64().unwrap_or(3072) as usize;
        let _num_layers = config["num_hidden_layers"].as_u64().unwrap_or(32) as usize;
        let _num_heads = config["num_attention_heads"].as_u64().unwrap_or(32) as usize;
        let _intermediate_size = config["intermediate_size"].as_u64().unwrap_or(8192) as usize;
        
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
        
        // Handle tied word embeddings for Phi-3 models (similar to Llama)
        // Check if the config indicates tied embeddings and lm_head.weight is missing
        let tie_embeddings = config["tie_word_embeddings"].as_bool().unwrap_or(false);
        if tie_embeddings {
            if let Some(embed_weights) = tensors_map.get("model.embed_tokens.weight") {
                if !tensors_map.contains_key("lm_head.weight") {
                    println!("  🔗 Creating tied lm_head.weight from embed_tokens.weight");
                    tensors_map.insert("lm_head.weight".to_string(), embed_weights.clone());
                }
            } else {
                return Err(ResumeAlignerError::ModelError(
                    "Cannot tie embeddings: model.embed_tokens.weight not found".to_string()
                ));
            }
        }
        
        let vb = VarBuilder::from_tensors(
            tensors_map, 
            DType::F32, 
            device
        );
        
        // Create Phi-3 configuration - handle complex fields that need conversion
        let mut phi_config_value = config.clone();
        
        // Correctly parse the rope_scaling object if it exists
        if phi_config_value.get("rope_scaling").is_some() {
            // No conversion needed; let the model handle the complete configuration
            println!("  ✅ Rope scaling configuration detected, parsing as complete struct");
        }
        
        let phi_config = serde_json::from_value::<phi3::Config>(phi_config_value)
            .map_err(|e| {
                ResumeAlignerError::ModelError(format!(
                    "Failed to parse Phi-3 config: {}", e
                ))
            })?;
        
        // Load the model
        let phi_model = phi3::Model::new(&phi_config, vb).map_err(|e| {
            ResumeAlignerError::ModelError(format!(
                "Failed to load Phi-3 model: {}", e
            ))
        })?;
        
        Ok(Box::new(Phi4Model {
            model: phi_model,
            vocab_size,
        }))
    }
    
    /// Load a Llama model specifically  
    fn load_llama_model(
        model_path: &Path,
        device: &Device,
        config: &serde_json::Value,
    ) -> Result<Box<dyn LLMModel>> {
        // Parse model configuration
        let vocab_size = config["vocab_size"].as_u64().unwrap_or(32000) as usize;
        let hidden_size = config["hidden_size"].as_u64().unwrap_or(3072) as usize;
        let num_layers = config["num_hidden_layers"].as_u64().unwrap_or(28) as usize;
        let num_heads = config["num_attention_heads"].as_u64().unwrap_or(24) as usize;
        let num_kv_heads = config["num_key_value_heads"].as_u64().unwrap_or(8) as usize;
        let intermediate_size = config["intermediate_size"].as_u64().unwrap_or(8192) as usize;
        let rope_theta = config["rope_theta"].as_f64().unwrap_or(500000.0);
        let tie_word_embeddings = config["tie_word_embeddings"].as_bool().unwrap_or(false);
        
        // Load model weights using the same logic as Phi-3
        let mut tensors_map = std::collections::HashMap::new();
        
        // Check for sharded safetensors first
        let index_path = model_path.join("model.safetensors.index.json");
        if index_path.exists() {
            println!("  📁 Loading sharded safetensors model...");
            
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
                let mut shard_files = std::collections::HashSet::new();
                for filename in weight_map.values() {
                    if let Some(filename_str) = filename.as_str() {
                        shard_files.insert(filename_str.to_string());
                    }
                }
                
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
                        
                        for (tensor_name, _) in shard_tensors.tensors() {
                            let tensor_view = shard_tensors.get(&tensor_name).map_err(|e| {
                                ResumeAlignerError::ModelError(format!(
                                    "Failed to get tensor {} from shard {}: {}", tensor_name, shard_file, e
                                ))
                            })?;
                            
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
                    }
                }
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
                
                for (tensor_name, _) in single_tensors.tensors() {
                    let tensor_view = single_tensors.get(&tensor_name).map_err(|e| {
                        ResumeAlignerError::ModelError(format!(
                            "Failed to get tensor {}: {}", tensor_name, e
                        ))
                    })?;
                    
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
        
        // Handle tied word embeddings for Llama models
        if tie_word_embeddings {
            if let Some(embed_weights) = tensors_map.get("model.embed_tokens.weight") {
                if !tensors_map.contains_key("lm_head.weight") {
                    println!("  🔗 Creating tied lm_head.weight from embed_tokens.weight");
                    tensors_map.insert("lm_head.weight".to_string(), embed_weights.clone());
                }
            } else {
                return Err(ResumeAlignerError::ModelError(
                    "Cannot tie embeddings: model.embed_tokens.weight not found".to_string()
                ));
            }
        }
        
        let vb = VarBuilder::from_tensors(
            tensors_map, 
            DType::F32, 
            device
        );
        
        // Create Llama configuration
        let llama_config = llama::Config {
            hidden_size,
            intermediate_size,
            vocab_size,
            num_hidden_layers: num_layers,
            num_attention_heads: num_heads,
            num_key_value_heads: num_kv_heads,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-5,
            rope_theta: rope_theta as f32,
            rope_scaling: None,
            tie_word_embeddings: false,
            bos_token_id: Some(1),
            eos_token_id: Some(llama::LlamaEosToks::Single(2)),
            use_flash_attn: false, // Disable flash attention for compatibility
        };
        
        // Load the model
        let llama_model = llama::Llama::load(vb, &llama_config).map_err(|e| {
            ResumeAlignerError::ModelError(format!(
                "Failed to load Llama model: {}", e
            ))
        })?;
        
        // Initialize cache with KV cache ENABLED for optimization
        let cache = llama::Cache::new(true, DType::F32, &llama_config, device).map_err(|e| {
            ResumeAlignerError::ModelError(format!(
                "Failed to initialize Llama cache: {}", e
            ))
        })?;
        
        Ok(Box::new(LlamaModel {
            model: llama_model,
            cache,
            vocab_size,
            config: llama_config,
            device: device.clone(),
        }))
    }

    pub async fn generate_with_timing(&mut self, prompt: &str) -> Result<(InferenceResult, TimingBreakdown)> {
        let total_start = std::time::Instant::now();
        
        // Reset KV cache for fresh generation
        self.model.reset_kv_cache()?;
        
        // Format prompt for phi-4-mini if needed
        let formatted_prompt = if self.is_phi_model() {
            format!("<|user|>\n{}\n<|end|>\n<|assistant|>\n", prompt.trim())
        } else {
            prompt.to_string()
        };
        
        if is_verbose_mode() {
            println!("🔤 Input prompt: '{}'", &formatted_prompt);
        }
        
        // Tokenization
        let tokenize_start = std::time::Instant::now();
        let encoding = self.tokenizer.encode(formatted_prompt.as_str(), true).map_err(|e| {
            ResumeAlignerError::ModelError(format!("Failed to tokenize input: {}", e))
        })?;
        let input_ids = encoding.get_ids();
        let mut tokens = input_ids.to_vec();
        let tokenize_time = tokenize_start.elapsed();
        
        if is_verbose_mode() {
            println!("🔤 Tokenized {} input tokens in {}ms", tokens.len(), tokenize_time.as_millis());
        }
    
        // Process entire prompt at once (this is the key fix)
        let first_pass_start = std::time::Instant::now();
        let input_tensor = Tensor::new(&tokens[..], &self.device)?.unsqueeze(0)?;
        let mut logits = self.model.forward(&input_tensor, 0)?;
        let first_pass_time = first_pass_start.elapsed();
    
        if is_verbose_mode() {
            println!("🚀 Processed entire prompt ({} tokens) in {}ms", tokens.len(), first_pass_time.as_millis());
        }
    
        // Generation loop - now properly incremental
        let generation_start = std::time::Instant::now();
        let mut generated_tokens = Vec::new();
        let input_length = tokens.len();
        
        for step in 0..self.config.max_tokens {
            let token_start = std::time::Instant::now();
            
            // Get the last token's logits from current output
            let final_logits = if logits.dims().len() == 3 {
                // [batch_size, seq_len, vocab_size] - get last position
                let seq_len = logits.dims()[1];
                logits.i((0, seq_len - 1))?
            } else if logits.dims().len() == 2 {
                // [seq_len, vocab_size] - get last position  
                let seq_len = logits.dims()[0];
                logits.i(seq_len - 1)?
            } else {
                // Already correct shape
                logits.clone()
            };
            
            // Sample next token
            //let next_token = final_logits.argmax(candle_core::D::Minus1)?
             //   .to_scalar::<u32>()?;

            // Sample next token with improved sampling and temperature
            let next_token = {
                let top_values = final_logits.to_vec1::<f32>()?;
                
                // Apply temperature scaling for better diversity
                let temperature = self.config.temperature as f32;
                let scaled_logits: Vec<f32> = if temperature > 0.0 && temperature != 1.0 {
                    top_values.iter().map(|&x| x / temperature).collect()
                } else {
                    top_values.clone()
                };
                
                // Convert to probabilities using softmax
                let max_logit = scaled_logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let exp_logits: Vec<f32> = scaled_logits.iter().map(|&x| (x - max_logit).exp()).collect();
                let sum_exp: f32 = exp_logits.iter().sum();
                let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum_exp).collect();
                
                // Apply top-k filtering if specified
                let mut candidates: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
                candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                
                if let Some(top_k) = self.config.top_k {
                    candidates.truncate(top_k);
                }
                
                // Apply top-p (nucleus) sampling
                let mut cumulative_prob = 0.0;
                let top_p = self.config.top_p as f32;
                let mut nucleus_candidates = Vec::new();
                
                for (idx, prob) in candidates {
                    cumulative_prob += prob;
                    nucleus_candidates.push((idx, prob));
                    if cumulative_prob >= top_p {
                        break;
                    }
                }
                
                // Select token (for now, just use the top candidate to avoid randomness issues)
                let chosen_token = nucleus_candidates[0].0 as u32;
                
                // Validate token range
                if chosen_token >= top_values.len() as u32 {
                    if is_verbose_mode() {
                        println!("⚠️ Token {} out of range, using 0", chosen_token);
                    }
                    0
                } else {
                    chosen_token
                }
            };

            // Only check for EOS after we've generated some meaningful content
            if step >= 10 && self.is_eos_token(next_token) {
                if is_verbose_mode() {
                    println!("🛑 EOS token detected ({}) after {} tokens, stopping generation", next_token, step);
                }
                break;
            }

            tokens.push(next_token);
            generated_tokens.push(next_token);
            
            // Process only the new token for next iteration
            let new_token_tensor = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            logits = self.model.forward(&new_token_tensor, input_length + step)?;
            
            let _token_time = token_start.elapsed();
            
            if is_verbose_mode() && step > 0 && step % 10 == 0 {
                println!("📊 Generated {} tokens, avg: {:.1}ms/token", 
                    step, generation_start.elapsed().as_millis() as f64 / step as f64);
            }
        }
        
        let generation_time = generation_start.elapsed();
        
        // Decode all tokens (input + generated) for proper context
        let decode_start = std::time::Instant::now();
        let all_tokens = tokens.clone();
        let full_text = self.tokenizer.decode(&all_tokens, true)
            .map_err(|e| ResumeAlignerError::ModelError(format!("Failed to decode full sequence: {}", e)))?;
        
        // Extract only the generated portion by removing the input prompt
        let generated_text = if full_text.len() > formatted_prompt.len() {
            full_text[formatted_prompt.len()..].trim().to_string()
        } else {
            // Fallback: decode only generated tokens if full decode fails
            self.tokenizer.decode(&generated_tokens, true)
                .map_err(|e| ResumeAlignerError::ModelError(format!("Failed to decode generated tokens: {}", e)))?
        };
        let decode_time = decode_start.elapsed();

        // DEBUG: Show what was actually generated
        println!("🎯 Generated text: '{}'", generated_text);
        println!("🔢 Generated tokens: {:?}", generated_tokens);
        
        let total_time = total_start.elapsed();
        
        let timing = TimingBreakdown {
            total_ms: total_time.as_millis() as u64,
            tokenization_ms: tokenize_time.as_millis() as u64,
            first_pass_ms: first_pass_time.as_millis() as u64,
            generation_ms: generation_time.as_millis() as u64,
            decode_ms: decode_time.as_millis() as u64,
            input_tokens: input_length,
            output_tokens: generated_tokens.len(),
            avg_ms_per_token: if generated_tokens.is_empty() { 0.0 } else { 
                generation_time.as_millis() as f64 / generated_tokens.len() as f64 
            },
        };
        
        let result = InferenceResult {
            text: generated_text.trim().to_string(),
            token_count: generated_tokens.len(),
            inference_time_ms: total_time.as_millis() as u64,
            tokens_per_second: if generated_tokens.is_empty() { 0.0 } else {
                generated_tokens.len() as f64 / total_time.as_secs_f64()
            },
        };
        
        if is_verbose_mode() {
            println!("⏱️  TIMING BREAKDOWN:");
            println!("   Tokenization: {}ms", timing.tokenization_ms);
            println!("   Prompt Processing: {}ms ({} tokens)", timing.first_pass_ms, timing.input_tokens);
            println!("   Generation: {}ms ({} tokens, {:.1}ms/token)", 
                timing.generation_ms, timing.output_tokens, timing.avg_ms_per_token);
            println!("   Decoding: {}ms", timing.decode_ms);
            println!("   TOTAL: {}ms ({:.1} tokens/sec)", 
                timing.total_ms, 
                timing.output_tokens as f64 / (timing.total_ms as f64 / 1000.0));
        }
        
        Ok((result, timing))
    }

    pub async fn generate(&mut self, prompt: &str) -> Result<InferenceResult> {
        let (result, timing) = self.generate_with_timing(prompt).await?;
        // Only show timing if explicitly requested
        if is_verbose_mode() {
            println!("⏱️ Generation took {}ms ({:.1} tokens/sec)", 
                timing.total_ms, 
                timing.output_tokens as f64 / (timing.total_ms as f64 / 1000.0));
        }
        
        Ok(result)
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
   
    /// Check if the current model is a Phi model
    fn is_phi_model(&self) -> bool {
        // Check vocab size as a heuristic for phi-4-mini
        self.model.get_vocab_size() > 100000
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

/// Detailed timing breakdown for performance analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingBreakdown {
    pub total_ms: u64,
    pub tokenization_ms: u64,
    pub first_pass_ms: u64,
    pub generation_ms: u64,
    pub decode_ms: u64,
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub avg_ms_per_token: f64,
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
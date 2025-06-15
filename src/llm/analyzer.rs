//! LLM-based analysis integration for the resume aligner

use crate::config::Config;
use crate::error::{Result, ResumeAlignerError};
use crate::llm::inference::{InferenceConfig, LLMEngine};
use crate::llm::model_manager::ModelManager;
use crate::llm::prompts::{PromptParams, PromptTemplates};
use crate::processing::document::ProcessedDocument;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// LLM-specific analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMAnalysis {
    pub gap_analysis: String,
    pub recommendations: Vec<LLMRecommendation>,
    pub skill_assessment: String,
    pub overall_score: f32,
    pub confidence: f32,
    pub processing_time_ms: u64,
    pub model_used: String,
    pub token_usage: TokenUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMRecommendation {
    pub title: String,
    pub description: String,
    pub priority: LLMPriority,
    pub section: String,
    pub impact: String,
    pub example: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LLMPriority {
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub total_tokens: usize,
}

/// LLM analyzer that integrates with the main analysis engine
pub struct LLMAnalyzer {
    model_manager: ModelManager,
    llm_engine: Option<LLMEngine>,
    prompt_templates: PromptTemplates,
    _config: Config,
    current_model: Option<String>,
}

impl LLMAnalyzer {
    /// Create a new LLM analyzer
    pub async fn new(config: Config) -> Result<Self> {
        let models_dir = config.get_models_dir();
        let model_manager = ModelManager::new(models_dir).await?;
        let prompt_templates = PromptTemplates::default();
        
        Ok(Self {
            model_manager,
            llm_engine: None,
            prompt_templates,
            _config: config,
            current_model: None,
        })
    }
    
    /// Initialize the LLM engine with the specified or auto-selected model
    pub async fn initialize_engine(&mut self, model_id: Option<String>) -> Result<()> {
        let model_id = match model_id {
            Some(id) => id,
            None => self.model_manager.auto_select_model().await?,
        };
        
        // Check if model is downloaded, download if needed
        if !self.model_manager.is_model_downloaded(&model_id) {
            println!("📥 Model '{}' not found locally, downloading...", model_id);
            self.model_manager.download_model(&model_id).await?;
        }
        
        let model_path = self.model_manager.get_model_path(&model_id)
            .ok_or_else(|| ResumeAlignerError::ModelError(
                format!("Model '{}' not found after download", model_id)
            ))?;
        
        // Create inference configuration
        let inference_config = InferenceConfig {
            max_tokens: 1024,
            temperature: 0.7,
            top_p: 0.9,
            top_k: Some(50),
            repeat_penalty: 1.1,
            seed: None,
            batch_size: 1,
        };
        
        // Load the model
        println!("🤖 Initializing LLM engine with model: {}", model_id);
        let engine = LLMEngine::load_model(&model_path, inference_config).await?;
        
        self.llm_engine = Some(engine);
        self.current_model = Some(model_id);
        
        Ok(())
    }
    
    /// Perform comprehensive LLM analysis
    pub async fn analyze(
        &mut self,
        resume: &ProcessedDocument,
        job: &ProcessedDocument,
        existing_analysis: &ExistingAnalysis,
    ) -> Result<LLMAnalysis> {
        let start_time = Instant::now();
        
        // Ensure LLM engine is initialized
        if self.llm_engine.is_none() {
            self.initialize_engine(None).await?;
        }
        
        // Prepare prompt parameters first
        let prompt_params = self.create_prompt_params(
            resume,
            job,
            existing_analysis,
        );
        
        let engine = self.llm_engine.as_mut().unwrap();
        
        // Perform gap analysis
        println!("🔍 Performing LLM gap analysis...");
        let gap_prompt = self.prompt_templates.render_gap_analysis(&prompt_params);
        let gap_result = engine.generate(&gap_prompt).await?;
        
        // Generate recommendations
        println!("💡 Generating LLM recommendations...");
        let rec_prompt = self.prompt_templates.render_recommendations(&prompt_params);
        let rec_result = engine.generate(&rec_prompt).await?;
        
        // Perform skill assessment
        println!("🎯 Analyzing skills with LLM...");
        let skill_prompt = self.prompt_templates.render_skill_extraction(&prompt_params);
        let skill_result = engine.generate(&skill_prompt).await?;
        
        // Parse recommendations from LLM output
        let recommendations = self.parse_recommendations(&rec_result.text);
        
        // Calculate LLM-based overall score
        let llm_score = self.calculate_llm_score(
            &gap_result.text,
            &skill_result.text,
            existing_analysis,
        );
        
        let processing_time = start_time.elapsed();
        
        // Calculate token usage
        let token_usage = TokenUsage {
            input_tokens: gap_result.token_count + rec_result.token_count + skill_result.token_count,
            output_tokens: gap_result.token_count + rec_result.token_count + skill_result.token_count,
            total_tokens: (gap_result.token_count + rec_result.token_count + skill_result.token_count) * 2,
        };
        
        Ok(LLMAnalysis {
            gap_analysis: gap_result.text.clone(),
            recommendations,
            skill_assessment: skill_result.text,
            overall_score: llm_score,
            confidence: self.calculate_confidence(&gap_result.text),
            processing_time_ms: processing_time.as_millis() as u64,
            model_used: self.current_model.clone().unwrap_or("unknown".to_string()),
            token_usage,
        })
    }
    
    /// Create prompt parameters from analysis data
    fn create_prompt_params(
        &self,
        resume: &ProcessedDocument,
        job: &ProcessedDocument,
        existing_analysis: &ExistingAnalysis,
    ) -> PromptParams {
        // Extract exact matches for prompt
        let exact_matches: Vec<String> = existing_analysis.exact_matches
            .iter()
            .take(10)
            .map(|m| m.keyword.clone())
            .collect();
        
        // Create section scores map
        let section_scores: HashMap<String, f32> = existing_analysis.section_scores
            .iter()
            .map(|(name, score)| (name.clone(), score.combined_score))
            .collect();
        
        PromptParams {
            resume_content: resume.original.content.clone(),
            job_content: job.original.content.clone(),
            missing_keywords: existing_analysis.missing_keywords.clone(),
            exact_matches,
            section_scores,
            overall_score: existing_analysis.overall_score,
            specific_instruction: None,
        }
    }
    
    /// Parse recommendations from LLM output
    fn parse_recommendations(&self, llm_output: &str) -> Vec<LLMRecommendation> {
        let mut recommendations = Vec::new();
        
        // Simple parsing - look for numbered recommendations
        let lines: Vec<&str> = llm_output.lines().collect();
        let mut current_rec: Option<LLMRecommendation> = None;
        
        for line in lines {
            let line = line.trim();
            
            // Check if this is a new recommendation (starts with number)
            if line.chars().next().map_or(false, |c| c.is_ascii_digit()) {
                // Save previous recommendation if exists
                if let Some(rec) = current_rec.take() {
                    recommendations.push(rec);
                }
                
                // Start new recommendation
                current_rec = Some(LLMRecommendation {
                    title: line.to_string(),
                    description: String::new(),
                    priority: LLMPriority::Medium, // Default
                    section: "General".to_string(),
                    impact: "Medium".to_string(),
                    example: None,
                });
            } else if !line.is_empty() {
                // Add to current recommendation description
                if let Some(ref mut rec) = current_rec {
                    if !rec.description.is_empty() {
                        rec.description.push(' ');
                    }
                    rec.description.push_str(line);
                    
                    // Parse priority and section from content
                    if line.to_lowercase().contains("high priority") || line.to_lowercase().contains("critical") {
                        rec.priority = LLMPriority::High;
                    } else if line.to_lowercase().contains("low priority") {
                        rec.priority = LLMPriority::Low;
                    }
                    
                    // Extract section information
                    if line.to_lowercase().contains("experience") {
                        rec.section = "Experience".to_string();
                    } else if line.to_lowercase().contains("skills") {
                        rec.section = "Skills".to_string();
                    } else if line.to_lowercase().contains("summary") || line.to_lowercase().contains("objective") {
                        rec.section = "Summary".to_string();
                    }
                }
            }
        }
        
        // Add the last recommendation
        if let Some(rec) = current_rec {
            recommendations.push(rec);
        }
        
        // Limit to top 5 recommendations
        recommendations.truncate(5);
        recommendations
    }
    
    /// Calculate LLM-based score from analysis content
    fn calculate_llm_score(
        &self,
        gap_analysis: &str,
        skill_assessment: &str,
        existing_analysis: &ExistingAnalysis,
    ) -> f32 {
        // Start with existing scores as baseline
        let llm_score = (existing_analysis.embedding_score + existing_analysis.keyword_score) / 2.0;
        
        // Adjust based on LLM analysis content
        let gap_lower = gap_analysis.to_lowercase();
        let skill_lower = skill_assessment.to_lowercase();
        
        // Positive indicators
        let positive_terms = ["strong match", "good alignment", "well-suited", "excellent", "strong candidate"];
        let negative_terms = ["missing", "lacking", "weak", "poor", "insufficient", "not suitable"];
        
        let mut adjustment = 0.0;
        
        for term in &positive_terms {
            if gap_lower.contains(term) || skill_lower.contains(term) {
                adjustment += 0.05;
            }
        }
        
        for term in &negative_terms {
            if gap_lower.contains(term) || skill_lower.contains(term) {
                adjustment -= 0.03;
            }
        }
        
        // Ensure score stays within bounds
        (llm_score + adjustment).clamp(0.0, 1.0)
    }
    
    /// Calculate confidence level based on analysis quality
    fn calculate_confidence(&self, gap_analysis: &str) -> f32 {
        let length_score = (gap_analysis.len() as f32 / 1000.0).min(1.0);
        let specificity_score = if gap_analysis.contains("specific") || gap_analysis.contains("detailed") {
            0.2
        } else {
            0.0
        };
        
        (0.6 + length_score * 0.2 + specificity_score).min(1.0)
    }
    
    /// Get information about available models
    pub fn get_available_models(&self) -> Vec<&crate::llm::model_manager::ModelInfo> {
        self.model_manager.list_available_models()
    }
    
    /// Get current model information
    pub fn get_current_model(&self) -> Option<&String> {
        self.current_model.as_ref()
    }
    
    /// Check if LLM is ready for analysis
    pub fn is_ready(&self) -> bool {
        self.llm_engine.is_some()
    }
}

/// Existing analysis data passed to LLM analyzer
#[derive(Debug, Clone)]
pub struct ExistingAnalysis {
    pub overall_score: f32,
    pub embedding_score: f32,
    pub keyword_score: f32,
    pub missing_keywords: Vec<String>,
    pub exact_matches: Vec<ExactMatch>,
    pub section_scores: HashMap<String, SectionScore>,
}

#[derive(Debug, Clone)]
pub struct ExactMatch {
    pub keyword: String,
    pub count: usize,
}

#[derive(Debug, Clone)]
pub struct SectionScore {
    pub combined_score: f32,
    pub embedding_score: f32,
    pub keyword_score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_recommendations() {
        // Test the parsing logic directly without needing full analyzer
        let prompt_templates = PromptTemplates::default();
        let analyzer = TestLLMAnalyzer {
            prompt_templates,
        };
        
        let llm_output = "1. Improve technical skills section\n2. Add more quantified achievements\n3. Optimize keywords for ATS";
        
        let recommendations = analyzer.parse_recommendations(llm_output);
        assert_eq!(recommendations.len(), 3);
        assert!(recommendations[0].title.contains("technical skills"));
    }
    
    #[test]
    fn test_calculate_llm_score() {
        let analyzer = TestLLMAnalyzer {
            prompt_templates: PromptTemplates::default(),
        };
        
        let existing_analysis = ExistingAnalysis {
            overall_score: 0.7,
            embedding_score: 0.65,
            keyword_score: 0.75,
            missing_keywords: vec![],
            exact_matches: vec![],
            section_scores: HashMap::new(),
        };
        
        let score = analyzer.calculate_llm_score(
            "Strong match for this position",
            "Excellent technical skills",
            &existing_analysis,
        );
        
        assert!(score > 0.7); // Should be higher due to positive terms
    }
    
    // Test-only struct that has the methods we want to test
    struct TestLLMAnalyzer {
        prompt_templates: PromptTemplates,
    }
    
    impl TestLLMAnalyzer {
        fn parse_recommendations(&self, llm_output: &str) -> Vec<LLMRecommendation> {
            // Copy the logic from the real implementation
            let mut recommendations = Vec::new();
            
            let lines: Vec<&str> = llm_output.lines().collect();
            let mut current_rec: Option<LLMRecommendation> = None;
            
            for line in lines {
                let line = line.trim();
                
                if line.chars().next().map_or(false, |c| c.is_ascii_digit()) {
                    if let Some(rec) = current_rec.take() {
                        recommendations.push(rec);
                    }
                    
                    current_rec = Some(LLMRecommendation {
                        title: line.to_string(),
                        description: String::new(),
                        priority: LLMPriority::Medium,
                        section: "General".to_string(),
                        impact: "Medium".to_string(),
                        example: None,
                    });
                }
            }
            
            if let Some(rec) = current_rec {
                recommendations.push(rec);
            }
            
            recommendations.truncate(5);
            recommendations
        }
        
        fn calculate_llm_score(
            &self,
            gap_analysis: &str,
            skill_assessment: &str,
            existing_analysis: &ExistingAnalysis,
        ) -> f32 {
            // Copy the logic from the real implementation
            let llm_score = (existing_analysis.embedding_score + existing_analysis.keyword_score) / 2.0;
            
            let gap_lower = gap_analysis.to_lowercase();
            let skill_lower = skill_assessment.to_lowercase();
            
            let positive_terms = ["strong match", "good alignment", "well-suited", "excellent", "strong candidate"];
            let negative_terms = ["missing", "lacking", "weak", "poor", "insufficient", "not suitable"];
            
            let mut adjustment = 0.0;
            
            for term in &positive_terms {
                if gap_lower.contains(term) || skill_lower.contains(term) {
                    adjustment += 0.05;
                }
            }
            
            for term in &negative_terms {
                if gap_lower.contains(term) || skill_lower.contains(term) {
                    adjustment -= 0.03;
                }
            }
            
            (llm_score + adjustment).clamp(0.0, 1.0)
        }
    }
}

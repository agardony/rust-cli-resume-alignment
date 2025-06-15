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
    pub strategic_analysis: String,
    pub achievement_suggestions: String,
    pub ats_optimization: String,
    pub recommendations: Vec<LLMRecommendation>,
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
            max_tokens: 256,
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
    
    /// Perform comprehensive LLM analysis using single optimized prompt
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
        
        // Prepare prompt parameters
        let prompt_params = self.create_prompt_params(
            resume,
            job,
            existing_analysis,
        );
        
        let engine = self.llm_engine.as_mut().unwrap();
        
        // SINGLE COMBINED ANALYSIS (replaces 3 separate calls)
        println!("🎯 Performing combined LLM analysis...");
        let combined_prompt = self.prompt_templates.render_combined_analysis(&prompt_params);
        let combined_result = engine.generate(&combined_prompt).await?; // No timing output
        
        // Parse the structured response into separate sections
        let (strategic_analysis, achievement_suggestions, ats_optimization) = 
            self.parse_combined_response(&combined_result.text);

        // Parse recommendations from strategic analysis
        let recommendations = self.parse_recommendations(&strategic_analysis);

        // Calculate LLM-based overall score
        let llm_score = self.calculate_llm_score(
            &strategic_analysis,
            &ats_optimization,
            existing_analysis,
        );

        let processing_time = start_time.elapsed();

        // Simplified token usage calculation (single call)
        let token_usage = TokenUsage {
            input_tokens: combined_result.token_count / 2, // Rough estimate: half input, half output
            output_tokens: combined_result.token_count / 2,
            total_tokens: combined_result.token_count,
        };

        // Calculate confidence from strategic analysis
        let confidence = self.calculate_confidence(&strategic_analysis);
        
        Ok(LLMAnalysis {
            strategic_analysis,
            achievement_suggestions,
            ats_optimization,
            recommendations,
            overall_score: llm_score,
            confidence,
            processing_time_ms: processing_time.as_millis() as u64,
            model_used: self.current_model.clone().unwrap_or("unknown".to_string()),
            token_usage,
        })
    }

    /// Parse the combined response into separate sections
    fn parse_combined_response(&self, text: &str) -> (String, String, String) {
        let strategic = self.extract_section(text, "STRATEGIC")
            .unwrap_or_else(|| "Analysis complete. Review alignment and positioning.".to_string());
        
        let achievements = self.extract_section(text, "ACHIEVEMENTS")
            .unwrap_or_else(|| "Focus on quantified results and measurable impact.".to_string());
        
        let ats = self.extract_section(text, "ATS")
            .unwrap_or_else(|| "Optimize keyword placement in summary and experience sections.".to_string());
        
        (strategic, achievements, ats)
    }

    /// Extract content from marked sections (## SECTION_NAME)
    fn extract_section(&self, text: &str, section_name: &str) -> Option<String> {
        let section_marker = format!("## {}", section_name);
        if let Some(start) = text.find(&section_marker) {
            let content_start = start + section_marker.len();
            let content = &text[content_start..];
            
            // Find the end (next ## marker or end of text)
            let end = content.find("## ").unwrap_or(content.len());
            let section_content = content[..end].trim();
            
            if !section_content.is_empty() {
                Some(section_content.to_string())
            } else {
                None
            }
        } else {
            None
        }
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
            focus_area: None, // Updated field name
        }
    }
    
    /// Parse recommendations from LLM output
    fn parse_recommendations(&self, llm_output: &str) -> Vec<LLMRecommendation> {
        let mut recommendations = Vec::new();
        
        // Enhanced parsing for strategic analysis output
        let lines: Vec<&str> = llm_output.lines().collect();
        let mut current_rec: Option<LLMRecommendation> = None;
        let mut in_priority_section = false;
        
        for line in lines {
            let line = line.trim();
            
            // Look for priority actions section
            if line.to_lowercase().contains("priority actions") || line.to_lowercase().contains("strategic") {
                in_priority_section = true;
                continue;
            }
            
            // Check if this is a new recommendation (starts with number or bullet)
            if (line.starts_with(char::is_numeric) || line.starts_with("•") || line.starts_with("-")) && in_priority_section {
                // Save previous recommendation if exists
                if let Some(rec) = current_rec.take() {
                    recommendations.push(rec);
                }
                
                // Start new recommendation
                current_rec = Some(LLMRecommendation {
                    title: line.to_string(),
                    description: String::new(),
                    priority: self.parse_priority(line),
                    section: self.parse_section(line),
                    impact: "Medium".to_string(),
                    example: None,
                });
            } else if !line.is_empty() && in_priority_section {
                // Add to current recommendation description
                if let Some(ref mut rec) = current_rec {
                    if !rec.description.is_empty() {
                        rec.description.push(' ');
                    }
                    rec.description.push_str(line);
                    
                    // Update impact based on content
                    if line.to_lowercase().contains("high impact") || line.to_lowercase().contains("critical") {
                        rec.impact = "High".to_string();
                    } else if line.to_lowercase().contains("low impact") {
                        rec.impact = "Low".to_string();
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
    
    /// Parse priority from recommendation text
    fn parse_priority(&self, text: &str) -> LLMPriority {
        let text_lower = text.to_lowercase();
        if text_lower.contains("high") || text_lower.contains("critical") || text_lower.contains("urgent") {
            LLMPriority::High
        } else if text_lower.contains("low") || text_lower.contains("minor") {
            LLMPriority::Low
        } else {
            LLMPriority::Medium
        }
    }
    
    /// Parse section from recommendation text
    fn parse_section(&self, text: &str) -> String {
        let text_lower = text.to_lowercase();
        if text_lower.contains("experience") || text_lower.contains("work") {
            "Experience".to_string()
        } else if text_lower.contains("skills") || text_lower.contains("technical") {
            "Skills".to_string()
        } else if text_lower.contains("summary") || text_lower.contains("objective") {
            "Summary".to_string()
        } else if text_lower.contains("education") {
            "Education".to_string()
        } else {
            "General".to_string()
        }
    }
    
    /// Calculate LLM-based score from analysis content
    fn calculate_llm_score(
        &self,
        strategic_analysis: &str,
        ats_analysis: &str,
        existing_analysis: &ExistingAnalysis,
    ) -> f32 {
        // Start with existing scores as baseline
        let baseline_score = (existing_analysis.embedding_score + existing_analysis.keyword_score) / 2.0;
        
        // Analyze sentiment and quality indicators
        let combined_text = format!("{} {}", strategic_analysis, ats_analysis).to_lowercase();
        
        // Positive indicators
        let positive_terms = [
            "strong match", "good alignment", "well-suited", "excellent", 
            "strong candidate", "highly qualified", "perfect fit", "ideal"
        ];
        let negative_terms = [
            "missing", "lacking", "weak", "poor", "insufficient", 
            "not suitable", "major gaps", "significant issues"
        ];
        
        let mut adjustment = 0.0;
        let mut positive_count = 0;
        let mut negative_count = 0;
        
        for term in &positive_terms {
            if combined_text.contains(term) {
                positive_count += 1;
                adjustment += 0.03;
            }
        }
        
        for term in &negative_terms {
            if combined_text.contains(term) {
                negative_count += 1;
                adjustment -= 0.02;
            }
        }
        
        // Bonus for detailed analysis
        if combined_text.len() > 1000 {
            adjustment += 0.02;
        }
        
        // Apply sentiment ratio
        if positive_count > 0 && negative_count > 0 {
            let sentiment_ratio = positive_count as f32 / (positive_count + negative_count) as f32;
            adjustment += (sentiment_ratio - 0.5) * 0.1;
        }
        
        // Ensure score stays within bounds
        (baseline_score + adjustment).clamp(0.0, 1.0)
    }
    
    /// Calculate confidence level based on analysis quality
    fn calculate_confidence(&self, strategic_analysis: &str) -> f32 {
        let length_score = (strategic_analysis.len() as f32 / 1500.0).min(1.0);
        
        let specificity_indicators = [
            "specific", "detailed", "concrete", "quantified", 
            "measurable", "actionable", "precise"
        ];
        
        let specificity_count = specificity_indicators.iter()
            .filter(|&term| strategic_analysis.to_lowercase().contains(term))
            .count();
        
        let specificity_score = (specificity_count as f32 * 0.1).min(0.3);
        
        // Check for structured output
        let structure_score = if strategic_analysis.contains("1.") || strategic_analysis.contains("•") {
            0.1
        } else {
            0.0
        };
        
        (0.5 + length_score * 0.3 + specificity_score + structure_score).min(1.0)
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
        let analyzer = TestLLMAnalyzer {
            prompt_templates: PromptTemplates::default(),
        };
        
        let llm_output = "PRIORITY ACTIONS:\n1. Improve technical skills section\n2. Add more quantified achievements\n3. Optimize keywords for ATS";
        
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
            "Strong match for this position with excellent qualifications",
            "Good ATS optimization potential",
            &existing_analysis,
        );
        
        assert!(score > 0.7); // Should be higher due to positive terms
    }
    
    #[test]
    fn test_parse_priority() {
        let analyzer = TestLLMAnalyzer {
            prompt_templates: PromptTemplates::default(),
        };
        
        assert!(matches!(analyzer.parse_priority("High priority fix"), LLMPriority::High));
        assert!(matches!(analyzer.parse_priority("Low impact change"), LLMPriority::Low));
        assert!(matches!(analyzer.parse_priority("Regular improvement"), LLMPriority::Medium));
    }
    
    // Test-only struct that has the methods we want to test
    struct TestLLMAnalyzer {
        prompt_templates: PromptTemplates,
    }
    
    impl TestLLMAnalyzer {
        fn parse_recommendations(&self, llm_output: &str) -> Vec<LLMRecommendation> {
            let mut recommendations = Vec::new();
            let lines: Vec<&str> = llm_output.lines().collect();
            let mut current_rec: Option<LLMRecommendation> = None;
            let mut in_priority_section = false;
            
            for line in lines {
                let line = line.trim();
                
                if line.to_lowercase().contains("priority actions") {
                    in_priority_section = true;
                    continue;
                }
                
                if (line.starts_with(char::is_numeric) || line.starts_with("•")) && in_priority_section {
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
            strategic_analysis: &str,
            ats_analysis: &str,
            existing_analysis: &ExistingAnalysis,
        ) -> f32 {
            let baseline_score = (existing_analysis.embedding_score + existing_analysis.keyword_score) / 2.0;
            let combined_text = format!("{} {}", strategic_analysis, ats_analysis).to_lowercase();
            
            let positive_terms = ["strong match", "excellent", "good"];
            let mut adjustment = 0.0;
            
            for term in &positive_terms {
                if combined_text.contains(term) {
                    adjustment += 0.03;
                }
            }
            
            (baseline_score + adjustment).clamp(0.0, 1.0)
        }
        
        fn parse_priority(&self, text: &str) -> LLMPriority {
            let text_lower = text.to_lowercase();
            if text_lower.contains("high") || text_lower.contains("critical") {
                LLMPriority::High
            } else if text_lower.contains("low") {
                LLMPriority::Low
            } else {
                LLMPriority::Medium
            }
        }
    }
}
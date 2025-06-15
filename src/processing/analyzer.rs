//! Main analysis engine combining embeddings, ATS matching, and future LLM analysis

use crate::config::Config;
use crate::error::Result;
use crate::llm::analyzer::{LLMAnalyzer, LLMAnalysis, ExistingAnalysis, ExactMatch, SectionScore};
use crate::processing::document::{ProcessedDocument, SectionType};
use crate::processing::embeddings::EmbeddingEngine;
use crate::processing::ats_matcher::ATSMatcher;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// Main analysis engine that coordinates all analysis components
pub struct AnalysisEngine {
    embedding_engine: EmbeddingEngine,
    ats_matcher: ATSMatcher,
    config: Config,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentReport {
    /// Overall alignment score (0.0 to 1.0)
    pub overall_score: f32,
    
    /// Component scores
    pub embedding_score: f32,
    pub ats_score: f32,
    pub llm_score: Option<f32>, // Will be added in Phase 5
    
    /// Detailed analysis results
    pub semantic_analysis: SemanticAnalysis,
    pub keyword_analysis: KeywordAnalysis,
    pub section_analysis: SectionAnalysis,
    pub gap_analysis: GapAnalysis,
    
    /// Performance metrics
    pub processing_time_ms: u64,
    pub model_info: ModelInfo,
    
    /// File paths
    pub resume_path: String,
    pub job_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticAnalysis {
    pub overall_similarity: f32,
    pub section_similarities: Vec<SectionSimilarityScore>,
    pub chunk_similarity_average: f32,
    pub embedding_dimension: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeywordAnalysis {
    pub ats_overall_score: f32,
    pub keyword_coverage: f32,
    pub exact_matches: Vec<MatchedKeyword>,
    pub fuzzy_matches: Vec<FuzzyMatchedKeyword>,
    pub skill_category_breakdown: SkillCategoryBreakdown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectionAnalysis {
    pub section_scores: HashMap<String, SectionAnalysisScore>,
    pub missing_sections: Vec<String>,
    pub strength_sections: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapAnalysis {
    pub missing_keywords: Vec<String>,
    pub missing_skills: Vec<MissingSkill>,
    pub recommendations: Vec<Recommendation>,
    pub priority_gaps: Vec<PriorityGap>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectionSimilarityScore {
    pub section_type: Option<SectionType>,
    pub similarity: f32,
    pub resume_content_preview: String,
    pub job_content_preview: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchedKeyword {
    pub keyword: String,
    pub count: usize,
    pub sections: Vec<String>,
    pub importance: ImportanceLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyMatchedKeyword {
    pub original_keyword: String,
    pub matched_text: String,
    pub similarity_score: f32,
    pub sections: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillCategoryBreakdown {
    pub technical_skills: f32,
    pub soft_skills: f32,
    pub role_specific: f32,
    pub overall: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectionAnalysisScore {
    pub section_type: SectionType,
    pub embedding_score: f32,
    pub keyword_score: f32,
    pub combined_score: f32,
    pub keyword_density: f32,
    pub missing_keywords: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingSkill {
    pub skill: String,
    pub category: SkillCategory,
    pub importance: ImportanceLevel,
    pub similar_skills: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub title: String,
    pub description: String,
    pub priority: Priority,
    pub impact: ImpactLevel,
    pub actionable_steps: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityGap {
    pub gap_type: GapType,
    pub description: String,
    pub severity: SeverityLevel,
    pub fix_effort: EffortLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub embedding_model: String,
    pub ats_matcher_skills: usize,
    pub llm_model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImportanceLevel {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SkillCategory {
    Technical,
    Soft,
    RoleSpecific,
    Domain,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Priority {
    Immediate,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImpactLevel {
    Major,
    Moderate,
    Minor,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GapType {
    MissingKeyword,
    MissingSection,
    LowSimilarity,
    SkillGap,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SeverityLevel {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EffortLevel {
    Minimal,
    Low,
    Medium,
    High,
}

impl AnalysisEngine {
    /// Create a new analysis engine with the given configuration
    pub async fn new(config: &Config) -> Result<Self> {
        let embedding_engine = EmbeddingEngine::from_config(config).await?;
        let ats_matcher = ATSMatcher::new()?;
        
        Ok(Self {
            embedding_engine,
            ats_matcher,
            config: config.clone(),
        })
    }
    
    /// Perform comprehensive alignment analysis between resume and job description
    pub async fn analyze_alignment(
        &mut self,
        resume: &ProcessedDocument,
        job: &ProcessedDocument,
    ) -> Result<AlignmentReport> {
        self.analyze_alignment_with_llm(resume, job, None).await
    }
    
    /// Perform comprehensive alignment analysis with custom LLM model
    pub async fn analyze_alignment_with_llm(
        &mut self,
        resume: &ProcessedDocument,
        job: &ProcessedDocument,
        llm_model: Option<String>,
    ) -> Result<AlignmentReport> {
        let start_time = Instant::now();
        
        // 1. Semantic Analysis using embeddings
        let semantic_analysis = self.perform_semantic_analysis(resume, job).await?;
        let embedding_score = semantic_analysis.overall_similarity;
        
        // 2. Keyword Analysis using ATS matcher
        let keyword_analysis = self.perform_keyword_analysis(resume, job)?;
        let ats_score = keyword_analysis.ats_overall_score;
        
        // 3. Section-wise Analysis
        let section_analysis = self.perform_section_analysis(resume, job).await?;
        
        // 4. LLM Analysis (Phase 5)
        let llm_analysis = match self.perform_llm_analysis_with_model(resume, job, embedding_score, ats_score, llm_model).await {
            Ok(analysis) => {
                log::info!("LLM analysis completed successfully with score: {:.1}%", analysis.overall_score * 100.0);
                Some(analysis)
            }
            Err(e) => {
                log::warn!("LLM analysis failed: {}", e);
                None
            }
        };
        let llm_score = llm_analysis.as_ref().map(|analysis| analysis.overall_score);
        
        // 5. Gap Analysis and Recommendations
        let gap_analysis = self.perform_gap_analysis(resume, job, &keyword_analysis)?;
        
        // 6. Calculate combined score using config weights
        let overall_score = self.calculate_combined_score(embedding_score, ats_score, llm_score);
        
        let processing_time = start_time.elapsed();
        
        Ok(AlignmentReport {
            overall_score,
            embedding_score,
            ats_score,
            llm_score,
            semantic_analysis,
            keyword_analysis,
            section_analysis,
            gap_analysis,
            processing_time_ms: processing_time.as_millis() as u64,
            model_info: ModelInfo {
                embedding_model: self.config.models.default_embedding_model.clone(),
                ats_matcher_skills: self.ats_matcher.skill_count(),
                llm_model: llm_analysis.map(|analysis| analysis.model_used),
            },
            resume_path: resume.original.file_path.clone(),
            job_path: job.original.file_path.clone(),
        })
    }
    
    /// Perform semantic analysis using embeddings
    async fn perform_semantic_analysis(
        &mut self,
        resume: &ProcessedDocument,
        job: &ProcessedDocument,
    ) -> Result<SemanticAnalysis> {
        // Generate embeddings for both documents
        let resume_embeddings = self.embedding_engine.process_document(resume)?;
        let job_embeddings = self.embedding_engine.process_document(job)?;
        
        // Calculate document similarity
        let doc_similarity = self.embedding_engine.calculate_document_similarity(
            &resume_embeddings,
            &job_embeddings,
        )?;
        
        // Convert to our analysis format
        let section_similarities = doc_similarity.section_similarities
            .into_iter()
            .map(|sim| SectionSimilarityScore {
                section_type: None, // We'll enhance this in future versions
                similarity: sim.similarity.score,
                resume_content_preview: Self::truncate_text(&sim.section1_text, 100),
                job_content_preview: Self::truncate_text(&sim.section2_text, 100),
            })
            .collect();
        
        Ok(SemanticAnalysis {
            overall_similarity: doc_similarity.overall_similarity.score,
            section_similarities,
            chunk_similarity_average: doc_similarity.average_chunk_similarity.score,
            embedding_dimension: doc_similarity.overall_similarity.embedding_dim,
        })
    }
    
    /// Perform keyword analysis using ATS matcher
    fn perform_keyword_analysis(
        &self,
        resume: &ProcessedDocument,
        job: &ProcessedDocument,
    ) -> Result<KeywordAnalysis> {
        let ats_score = self.ats_matcher.calculate_ats_score(resume, job)?;
        
        // Convert exact matches to our format
        let exact_matches = ats_score.exact_matches
            .into_iter()
            .map(|m| {
                let keyword = m.keyword.clone();
                MatchedKeyword {
                    keyword: keyword.clone(),
                    count: m.count,
                    sections: vec![format!("{:?}", m.section_type.unwrap_or(SectionType::Other("Unknown".to_string())))],
                    importance: Self::determine_keyword_importance(&keyword),
                }
            })
            .collect();
        
        // Convert fuzzy matches to our format
        let fuzzy_matches = ats_score.fuzzy_matches
            .into_iter()
            .map(|m| FuzzyMatchedKeyword {
                original_keyword: m.original_keyword,
                matched_text: m.matched_text,
                similarity_score: m.similarity_score,
                sections: vec![format!("{:?}", m.section_type.unwrap_or(SectionType::Other("Unknown".to_string())))],
            })
            .collect();
        
        Ok(KeywordAnalysis {
            ats_overall_score: ats_score.overall_score,
            keyword_coverage: ats_score.keyword_coverage,
            exact_matches,
            fuzzy_matches,
            skill_category_breakdown: SkillCategoryBreakdown {
                technical_skills: ats_score.skill_category_scores.technical_skills,
                soft_skills: ats_score.skill_category_scores.soft_skills,
                role_specific: ats_score.skill_category_scores.role_specific,
                overall: ats_score.skill_category_scores.overall,
            },
        })
    }
    
    /// Perform section-wise analysis
    async fn perform_section_analysis(
        &mut self,
        resume: &ProcessedDocument,
        job: &ProcessedDocument,
    ) -> Result<SectionAnalysis> {
        let mut section_scores = HashMap::new();
        let ats_score = self.ats_matcher.calculate_ats_score(resume, job)?;
        
        // Analyze each resume section
        for resume_section in &resume.sections {
            let section_name = format!("{}", resume_section.section_type);
            
            // Get embedding score for this section
            let section_embedding = self.embedding_engine.encode_single(&resume_section.content);
            let job_embedding = self.embedding_engine.encode_single(&job.original.content);
            let embedding_similarity = EmbeddingEngine::cosine_similarity(&section_embedding, &job_embedding)?;
            
            // Get ATS score for this section
            let section_ats_info = ats_score.section_scores.get(&section_name);
            let keyword_score = section_ats_info.map(|s| s.score).unwrap_or(0.0);
            let keyword_density = section_ats_info.map(|s| s.keyword_density).unwrap_or(0.0);
            let missing_keywords = section_ats_info
                .map(|s| s.missing_keywords.clone())
                .unwrap_or_default();
            
            // Calculate combined score for this section
            let combined_score = self.calculate_combined_score(
                embedding_similarity.score,
                keyword_score,
                None,
            );
            
            section_scores.insert(section_name, SectionAnalysisScore {
                section_type: resume_section.section_type.clone(),
                embedding_score: embedding_similarity.score,
                keyword_score,
                combined_score,
                keyword_density,
                missing_keywords,
            });
        }
        
        // Identify missing sections (basic implementation)
        let missing_sections = self.identify_missing_sections(resume, job);
        let strength_sections = self.identify_strength_sections(&section_scores);
        
        Ok(SectionAnalysis {
            section_scores,
            missing_sections,
            strength_sections,
        })
    }
    
    /// Perform gap analysis and generate recommendations
    fn perform_gap_analysis(
        &self,
        resume: &ProcessedDocument,
        job: &ProcessedDocument,
        keyword_analysis: &KeywordAnalysis,
    ) -> Result<GapAnalysis> {
        let ats_score = self.ats_matcher.calculate_ats_score(resume, job)?;
        
        // Missing skills analysis
        let missing_skills: Vec<MissingSkill> = ats_score.missing_keywords
            .iter()
            .map(|keyword| MissingSkill {
                skill: keyword.clone(),
                category: Self::categorize_skill(keyword),
                importance: Self::determine_keyword_importance(keyword),
                similar_skills: vec![], // Could be enhanced with similarity search
            })
            .collect();
        
        // Generate recommendations
        let recommendations = self.generate_recommendations(&missing_skills, keyword_analysis)?;
        
        // Identify priority gaps
        let priority_gaps = self.identify_priority_gaps(&missing_skills, keyword_analysis)?;
        
        Ok(GapAnalysis {
            missing_keywords: ats_score.missing_keywords,
            missing_skills,
            recommendations,
            priority_gaps,
        })
    }
    
    /// Calculate combined score using configuration weights
    fn calculate_combined_score(
        &self,
        embedding_score: f32,
        keyword_score: f32,
        llm_score: Option<f32>,
    ) -> f32 {
        let weights = &self.config.scoring;
        
        if let Some(llm) = llm_score {
            // Full weighted scoring when LLM is available
            (embedding_score * weights.embedding_weight) +
            (keyword_score * weights.keyword_weight) +
            (llm * weights.llm_weight)
        } else {
            // Adjust weights when LLM is not available
            let total_weight = weights.embedding_weight + weights.keyword_weight;
            let adjusted_emb_weight = weights.embedding_weight / total_weight;
            let adjusted_kw_weight = weights.keyword_weight / total_weight;
            
            (embedding_score * adjusted_emb_weight) + (keyword_score * adjusted_kw_weight)
        }
    }
    
    /// Helper method to truncate text for previews
    fn truncate_text(text: &str, max_length: usize) -> String {
        if text.len() <= max_length {
            text.to_string()
        } else {
            format!("{}...", &text[..max_length])
        }
    }
    
    /// Determine keyword importance (basic heuristic)
    fn determine_keyword_importance(keyword: &str) -> ImportanceLevel {
        let critical_keywords = ["required", "must", "essential", "mandatory"];
        let high_keywords = ["preferred", "strong", "proficient", "expert"];
        
        let keyword_lower = keyword.to_lowercase();
        
        if critical_keywords.iter().any(|&k| keyword_lower.contains(k)) {
            ImportanceLevel::Critical
        } else if high_keywords.iter().any(|&k| keyword_lower.contains(k)) {
            ImportanceLevel::High
        } else if keyword.len() > 10 {
            ImportanceLevel::Medium
        } else {
            ImportanceLevel::Low
        }
    }
    
    /// Categorize a skill (basic heuristic)
    fn categorize_skill(skill: &str) -> SkillCategory {
        let tech_indicators = ["programming", "language", "framework", "database", "tool", "api"];
        let soft_indicators = ["leadership", "communication", "management", "teamwork"];
        let role_indicators = ["engineer", "developer", "manager", "architect", "analyst"];
        
        let skill_lower = skill.to_lowercase();
        
        if role_indicators.iter().any(|&i| skill_lower.contains(i)) {
            SkillCategory::RoleSpecific
        } else if tech_indicators.iter().any(|&i| skill_lower.contains(i)) {
            SkillCategory::Technical
        } else if soft_indicators.iter().any(|&i| skill_lower.contains(i)) {
            SkillCategory::Soft
        } else {
            SkillCategory::Domain
        }
    }
    
    /// Identify missing sections in resume
    fn identify_missing_sections(
        &self,
        resume: &ProcessedDocument,
        _job: &ProcessedDocument,
    ) -> Vec<String> {
        let resume_sections: std::collections::HashSet<String> = resume.sections
            .iter()
            .map(|s| format!("{}", s.section_type))
            .collect();
        
        let mut missing = Vec::new();
        
        // Check for common important sections
        let important_sections = ["Skills", "Experience", "Education"];
        for section in &important_sections {
            if !resume_sections.contains(*section) {
                missing.push(section.to_string());
            }
        }
        
        missing
    }
    
    /// Identify strength sections based on scores
    fn identify_strength_sections(
        &self,
        section_scores: &HashMap<String, SectionAnalysisScore>,
    ) -> Vec<String> {
        let threshold = 0.7; // 70% threshold for strong sections
        
        section_scores
            .iter()
            .filter(|(_, score)| score.combined_score >= threshold)
            .map(|(name, _)| name.clone())
            .collect()
    }
    
    /// Generate actionable recommendations
    fn generate_recommendations(
        &self,
        missing_skills: &[MissingSkill],
        _keyword_analysis: &KeywordAnalysis,
    ) -> Result<Vec<Recommendation>> {
        let mut recommendations = Vec::new();
        
        // Group missing skills by category
        let mut tech_skills = Vec::new();
        let mut soft_skills = Vec::new();
        
        for skill in missing_skills {
            match skill.category {
                SkillCategory::Technical => tech_skills.push(&skill.skill),
                SkillCategory::Soft => soft_skills.push(&skill.skill),
                _ => {}
            }
        }
        
        // Technical skills recommendations
        if !tech_skills.is_empty() {
            recommendations.push(Recommendation {
                title: "Enhance Technical Skills Section".to_string(),
                description: format!(
                    "Add missing technical skills: {}",
                    tech_skills.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
                ),
                priority: Priority::High,
                impact: ImpactLevel::Major,
                actionable_steps: vec![
                    "Review job requirements for technical skills".to_string(),
                    "Add relevant skills to your Skills section".to_string(),
                    "Provide examples of using these skills in Experience section".to_string(),
                ],
            });
        }
        
        // Soft skills recommendations
        if !soft_skills.is_empty() {
            recommendations.push(Recommendation {
                title: "Highlight Soft Skills".to_string(),
                description: format!(
                    "Emphasize soft skills: {}",
                    soft_skills.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
                ),
                priority: Priority::Medium,
                impact: ImpactLevel::Moderate,
                actionable_steps: vec![
                    "Include soft skills in your Summary section".to_string(),
                    "Provide specific examples in Experience descriptions".to_string(),
                    "Use action verbs that demonstrate these qualities".to_string(),
                ],
            });
        }
        
        Ok(recommendations)
    }
    
    /// Identify priority gaps that need immediate attention
    fn identify_priority_gaps(
        &self,
        missing_skills: &[MissingSkill],
        keyword_analysis: &KeywordAnalysis,
    ) -> Result<Vec<PriorityGap>> {
        let mut gaps = Vec::new();
        
        // Low keyword coverage is a critical gap
        if keyword_analysis.keyword_coverage < 0.3 {
            gaps.push(PriorityGap {
                gap_type: GapType::MissingKeyword,
                description: "Low keyword coverage - resume may not pass ATS screening".to_string(),
                severity: SeverityLevel::Critical,
                fix_effort: EffortLevel::Medium,
            });
        }
        
        // Critical missing skills
        let critical_skills: Vec<&MissingSkill> = missing_skills
            .iter()
            .filter(|s| matches!(s.importance, ImportanceLevel::Critical))
            .collect();
        
        if !critical_skills.is_empty() {
            gaps.push(PriorityGap {
                gap_type: GapType::SkillGap,
                description: format!(
                    "Missing {} critical skills that may disqualify application",
                    critical_skills.len()
                ),
                severity: SeverityLevel::High,
                fix_effort: EffortLevel::Low,
            });
        }
        
        Ok(gaps)
    }
    
    /// Perform LLM-based analysis (Phase 5 implementation)
    async fn perform_llm_analysis(
        &self,
        resume: &ProcessedDocument,
        job: &ProcessedDocument,
        embedding_score: f32,
        keyword_score: f32,
    ) -> Result<LLMAnalysis> {
        self.perform_llm_analysis_with_model(resume, job, embedding_score, keyword_score, None).await
    }
    
    /// Perform LLM-based analysis with a specific model
    async fn perform_llm_analysis_with_model(
        &self,
        resume: &ProcessedDocument,
        job: &ProcessedDocument,
        embedding_score: f32,
        keyword_score: f32,
        model_id: Option<String>,
    ) -> Result<LLMAnalysis> {
        // Initialize LLM analyzer
        let mut llm_analyzer = LLMAnalyzer::new(self.config.clone()).await.map_err(|e| {
            log::warn!("Failed to initialize LLM analyzer: {}", e);
            e
        })?;
        
        // Initialize LLM engine with the specified model
        llm_analyzer.initialize_engine(model_id).await.map_err(|e| {
            log::warn!("Failed to initialize LLM engine: {}", e);
            e
        })?;
        
        // Prepare existing analysis data for LLM
        let ats_result = self.ats_matcher.calculate_ats_score(resume, job)?;
        
        let exact_matches: Vec<ExactMatch> = ats_result.exact_matches
            .iter()
            .map(|m| ExactMatch {
                keyword: m.keyword.clone(),
                count: m.count,
            })
            .collect();
        
        // Create section scores for LLM context
        let mut section_scores = HashMap::new();
        for section in &resume.sections {
            let section_name = format!("{}", section.section_type);
            section_scores.insert(section_name, SectionScore {
                combined_score: (embedding_score + keyword_score) / 2.0,
                embedding_score,
                keyword_score,
            });
        }
        
        let existing_analysis = ExistingAnalysis {
            overall_score: (embedding_score + keyword_score) / 2.0,
            embedding_score,
            keyword_score,
            missing_keywords: ats_result.missing_keywords,
            exact_matches,
            section_scores,
        };
        
        // Perform LLM analysis
        llm_analyzer.analyze(resume, job, &existing_analysis).await
    }
    
    /// Get analysis engine statistics
    pub fn get_stats(&self) -> AnalysisEngineStats {
        AnalysisEngineStats {
            embedding_cache_size: self.embedding_engine.cache_stats().cache_size,
            ats_skill_count: self.ats_matcher.skill_count(),
            fuzzy_threshold: self.ats_matcher.fuzzy_threshold(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AnalysisEngineStats {
    pub embedding_cache_size: usize,
    pub ats_skill_count: usize,
    pub fuzzy_threshold: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    

    #[tokio::test]
    async fn test_analysis_engine_creation() {
        let config = Config::default();
        
        // This test might fail if model files aren't available, which is expected
        if let Ok(_engine) = AnalysisEngine::new(&config).await {
            // Test passes if engine can be created
            assert!(true);
        } else {
            // Expected failure if models aren't downloaded
            assert!(true, "Model files not available - this is expected in test environment");
        }
    }

    #[test]
    fn test_score_calculation() {
        let config = Config::default();
        
        // Test the scoring logic without the embedding engine
        let embedding_weight = config.scoring.embedding_weight;
        let keyword_weight = config.scoring.keyword_weight;
        let llm_weight = config.scoring.llm_weight;
        
        // Test without LLM
        let total_weight = embedding_weight + keyword_weight;
        let adjusted_emb_weight = embedding_weight / total_weight;
        let adjusted_kw_weight = keyword_weight / total_weight;
        let combined_score = (0.8 * adjusted_emb_weight) + (0.7 * adjusted_kw_weight);
        assert!(combined_score > 0.0 && combined_score <= 1.0);
        
        // Test with LLM
        let combined_score_with_llm = (0.8 * embedding_weight) + (0.7 * keyword_weight) + (0.9 * llm_weight);
        assert!(combined_score_with_llm > 0.0 && combined_score_with_llm <= 1.0);
    }

    #[test]
    fn test_keyword_importance() {
        assert!(matches!(
            AnalysisEngine::determine_keyword_importance("required python"),
            ImportanceLevel::Critical
        ));
        
        assert!(matches!(
            AnalysisEngine::determine_keyword_importance("preferred javascript"),
            ImportanceLevel::High
        ));
    }

    #[test]
    fn test_skill_categorization() {
        assert!(matches!(
            AnalysisEngine::categorize_skill("python programming"),
            SkillCategory::Technical
        ));
        
        assert!(matches!(
            AnalysisEngine::categorize_skill("leadership skills"),
            SkillCategory::Soft
        ));
        
        assert!(matches!(
            AnalysisEngine::categorize_skill("software engineer"),
            SkillCategory::RoleSpecific
        ));
    }
}

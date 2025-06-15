//! Rich report structures with AI insights for Phase 6

use crate::processing::analyzer::{AlignmentReport, Priority};
use crate::llm::analyzer::{LLMAnalysis, LLMPriority};
use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// Comprehensive report that combines all analysis results with enhanced AI insights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveReport {
    /// Executive summary with key insights
    pub analysis_summary: AnalysisSummary,
    
    /// LLM-powered insights and natural language explanations
    pub llm_insights: LLMInsights,
    
    /// Actionable recommendations with specific examples
    pub recommendations: Vec<ActionableRecommendation>,
    
    /// Detailed gap analysis with improvement priorities
    pub gap_analysis: DetailedGapAnalysis,
    
    /// Structured improvement plan with timeline
    pub improvement_plan: ImprovementPlan,
    
    /// Report metadata and generation info
    pub metadata: ReportMetadata,
    
    /// Original analysis data (for detailed output)
    pub raw_analysis: AlignmentReport,
}

/// Executive summary with key findings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisSummary {
    /// Overall alignment score (0-100)
    pub overall_score_percentage: u8,
    
    /// Score breakdown by component
    pub score_breakdown: ScoreBreakdown,
    
    /// Key strengths identified
    pub strengths: Vec<String>,
    
    /// Top areas for improvement
    pub improvement_areas: Vec<String>,
    
    /// One-line verdict
    pub verdict: String,
    
    /// Confidence level in the analysis
    pub confidence_level: ConfidenceLevel,
}

/// Enhanced LLM insights with natural language explanations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMInsights {
    /// Natural language summary of the analysis
    pub natural_language_summary: String,
    
    /// Detailed skill assessment
    pub skill_assessment: SkillAssessment,
    
    /// Hidden requirements detected in the job description
    pub hidden_requirements: Vec<String>,
    
    /// Company culture indicators
    pub culture_indicators: Vec<String>,
    
    /// Transferable skills that could be highlighted
    pub transferable_skills: Vec<TransferableSkill>,
    
    /// LLM confidence and reasoning
    pub llm_confidence: f32,
    
    /// Model used for analysis
    pub model_used: String,
}

/// Actionable recommendations with examples
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionableRecommendation {
    /// Clear title of the recommendation
    pub title: String,
    
    /// Detailed description of what to do
    pub description: String,
    
    /// Priority level for this recommendation
    pub priority: RecommendationPriority,
    
    /// Resume section this applies to
    pub section: ResumeSection,
    
    /// Expected impact if implemented
    pub expected_impact: String,
    
    /// Specific action to take
    pub specific_action: String,
    
    /// Example of current text (if applicable)
    pub before_example: Option<String>,
    
    /// Example of improved text
    pub after_example: Option<String>,
    
    /// Estimated effort to implement
    pub effort_level: EffortLevel,
    
    /// Skills or keywords this addresses
    pub addresses_keywords: Vec<String>,
}

/// Detailed gap analysis with prioritized improvements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedGapAnalysis {
    /// Critical gaps that must be addressed
    pub critical_gaps: Vec<GapItem>,
    
    /// Important gaps that should be addressed
    pub important_gaps: Vec<GapItem>,
    
    /// Minor gaps that could be addressed
    pub minor_gaps: Vec<GapItem>,
    
    /// Skills analysis breakdown
    pub skills_analysis: SkillsGapAnalysis,
    
    /// Content analysis
    pub content_analysis: ContentGapAnalysis,
}

/// Structured improvement plan with timeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementPlan {
    /// Immediate actions (can be done in < 1 hour)
    pub immediate_actions: Vec<ImprovementStep>,
    
    /// Short-term improvements (1-3 days)
    pub short_term_improvements: Vec<ImprovementStep>,
    
    /// Long-term development (1+ weeks)
    pub long_term_development: Vec<ImprovementStep>,
    
    /// Cover letter talking points
    pub cover_letter_points: Vec<String>,
    
    /// Interview preparation suggestions
    pub interview_prep: InterviewPrepGuide,
}

/// Report metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMetadata {
    /// When the report was generated
    pub generated_at: SystemTime,
    
    /// Version of the aligner used
    pub aligner_version: String,
    
    /// Resume file analyzed
    pub resume_file: String,
    
    /// Job description file analyzed
    pub job_file: String,
    
    /// Total processing time
    pub processing_time_ms: u64,
    
    /// Models used in analysis
    pub models_used: ModelsUsed,
}

// Supporting structures

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    pub embedding_score: u8,
    pub keyword_score: u8,
    pub llm_score: Option<u8>,
    pub component_weights: ComponentWeights,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentWeights {
    pub embedding_weight: f32,
    pub keyword_weight: f32,
    pub llm_weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillAssessment {
    pub technical_skills: Vec<SkillMatch>,
    pub soft_skills: Vec<SkillMatch>,
    pub domain_skills: Vec<SkillMatch>,
    pub missing_critical_skills: Vec<String>,
    pub skill_coverage_percentage: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillMatch {
    pub skill: String,
    pub match_strength: MatchStrength,
    pub evidence: Vec<String>,
    pub improvement_suggestion: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferableSkill {
    pub skill: String,
    pub from_context: String,
    pub how_to_highlight: String,
    pub relevance_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapItem {
    pub gap_type: GapType,
    pub title: String,
    pub description: String,
    pub impact: ImpactAssessment,
    pub fix_suggestion: String,
    pub keywords_affected: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillsGapAnalysis {
    pub missing_technical_skills: Vec<String>,
    pub missing_soft_skills: Vec<String>,
    pub underrepresented_skills: Vec<String>,
    pub skill_level_gaps: Vec<SkillLevelGap>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentGapAnalysis {
    pub missing_sections: Vec<String>,
    pub weak_sections: Vec<WeakSection>,
    pub content_density_issues: Vec<String>,
    pub formatting_suggestions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementStep {
    pub title: String,
    pub description: String,
    pub estimated_time: String,
    pub expected_score_improvement: f32,
    pub resources_needed: Vec<String>,
    pub success_criteria: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterviewPrepGuide {
    pub key_talking_points: Vec<String>,
    pub questions_to_prepare: Vec<String>,
    pub skill_demonstration_examples: Vec<String>,
    pub potential_weaknesses_to_address: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsUsed {
    pub embedding_model: String,
    pub llm_model: Option<String>,
    pub ats_keywords_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillLevelGap {
    pub skill: String,
    pub required_level: String,
    pub current_level: String,
    pub gap_description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeakSection {
    pub section_name: String,
    pub current_score: f32,
    pub issues: Vec<String>,
    pub improvement_suggestions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactAssessment {
    pub ats_impact: String,
    pub human_reviewer_impact: String,
    pub interview_impact: String,
}

// Enums

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfidenceLevel {
    VeryHigh,   // > 90%
    High,       // 70-90%
    Medium,     // 50-70%
    Low,        // 30-50%
    VeryLow,    // < 30%
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Critical,   // Must fix
    High,       // Should fix
    Medium,     // Good to fix
    Low,        // Nice to fix
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResumeSection {
    Summary,
    Experience,
    Skills,
    Education,
    Projects,
    Certifications,
    Other(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EffortLevel {
    Minimal,    // < 30 minutes
    Low,        // 30 minutes - 2 hours
    Medium,     // 2-8 hours
    High,       // 1-3 days
    Significant, // > 3 days
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MatchStrength {
    Excellent,  // Clear match with strong evidence
    Good,       // Good match with some evidence
    Fair,       // Partial match or implied
    Weak,       // Barely detectable
    Missing,    // Not found
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GapType {
    MissingKeyword,
    MissingSkill,
    MissingSection,
    WeakContent,
    FormatIssue,
    QuantificationMissing,
    AchievementMissing,
}

impl ComprehensiveReport {
    /// Create a comprehensive report from basic alignment analysis
    pub fn from_alignment_analysis(
        alignment_report: AlignmentReport,
        llm_analysis: Option<LLMAnalysis>,
        config_weights: ComponentWeights,
    ) -> Self {
        let analysis_summary = Self::create_analysis_summary(&alignment_report, &llm_analysis, &config_weights);
        let llm_insights = Self::create_llm_insights(&llm_analysis);
        let recommendations = Self::create_actionable_recommendations(&alignment_report, &llm_analysis);
        let gap_analysis = Self::create_detailed_gap_analysis(&alignment_report, &llm_analysis);
        let improvement_plan = Self::create_improvement_plan(&recommendations, &gap_analysis);
        let metadata = Self::create_metadata(&alignment_report, &llm_analysis);
        
        Self {
            analysis_summary,
            llm_insights,
            recommendations,
            gap_analysis,
            improvement_plan,
            metadata,
            raw_analysis: alignment_report,
        }
    }
    
    fn create_analysis_summary(
        alignment_report: &AlignmentReport,
        llm_analysis: &Option<LLMAnalysis>,
        config_weights: &ComponentWeights,
    ) -> AnalysisSummary {
        let overall_score_percentage = (alignment_report.overall_score * 100.0) as u8;
        
        let score_breakdown = ScoreBreakdown {
            embedding_score: (alignment_report.embedding_score * 100.0) as u8,
            keyword_score: (alignment_report.ats_score * 100.0) as u8,
            llm_score: alignment_report.llm_score.map(|s| (s * 100.0) as u8),
            component_weights: config_weights.clone(),
        };
        
        // Determine strengths based on scores
        let mut strengths = Vec::new();
        if alignment_report.embedding_score > 0.7 {
            strengths.push("Strong semantic alignment with job requirements".to_string());
        }
        if alignment_report.ats_score > 0.7 {
            strengths.push("Good keyword coverage for ATS systems".to_string());
        }
        if let Some(llm_score) = alignment_report.llm_score {
            if llm_score > 0.7 {
                strengths.push("AI analysis shows strong overall fit".to_string());
            }
        }
        
        // Determine improvement areas
        let mut improvement_areas = Vec::new();
        if alignment_report.embedding_score < 0.5 {
            improvement_areas.push("Improve content relevance and semantic alignment".to_string());
        }
        if alignment_report.ats_score < 0.5 {
            improvement_areas.push("Add more relevant keywords for ATS optimization".to_string());
        }
        
        let verdict = match overall_score_percentage {
            90..=100 => "Excellent match - strong candidate for this role".to_string(),
            80..=89 => "Very good match - minor improvements could help".to_string(),
            70..=79 => "Good match - some targeted improvements recommended".to_string(),
            60..=69 => "Fair match - several improvements needed".to_string(),
            50..=59 => "Below average match - significant improvements required".to_string(),
            _ => "Poor match - major revisions needed".to_string(),
        };
        
        let confidence_level = match llm_analysis {
            Some(llm) if llm.confidence > 0.8 => ConfidenceLevel::VeryHigh,
            Some(llm) if llm.confidence > 0.6 => ConfidenceLevel::High,
            Some(llm) if llm.confidence > 0.4 => ConfidenceLevel::Medium,
            Some(_) => ConfidenceLevel::Low,
            None => ConfidenceLevel::Medium, // No LLM analysis
        };
        
        AnalysisSummary {
            overall_score_percentage,
            score_breakdown,
            strengths,
            improvement_areas,
            verdict,
            confidence_level,
        }
    }
    
    fn create_llm_insights(llm_analysis: &Option<LLMAnalysis>) -> LLMInsights {
        match llm_analysis {
            Some(llm) => {
                LLMInsights {
                    natural_language_summary: llm.gap_analysis.clone(),
                    skill_assessment: SkillAssessment {
                        technical_skills: vec![], // Will be populated from LLM analysis
                        soft_skills: vec![],
                        domain_skills: vec![],
                        missing_critical_skills: vec![],
                        skill_coverage_percentage: 75, // Placeholder
                    },
                    hidden_requirements: vec![], // Extract from LLM analysis
                    culture_indicators: vec![],
                    transferable_skills: vec![],
                    llm_confidence: llm.confidence,
                    model_used: llm.model_used.clone(),
                }
            }
            None => {
                LLMInsights {
                    natural_language_summary: "LLM analysis not available - using embeddings and keyword analysis only.".to_string(),
                    skill_assessment: SkillAssessment {
                        technical_skills: vec![],
                        soft_skills: vec![],
                        domain_skills: vec![],
                        missing_critical_skills: vec![],
                        skill_coverage_percentage: 0,
                    },
                    hidden_requirements: vec![],
                    culture_indicators: vec![],
                    transferable_skills: vec![],
                    llm_confidence: 0.0,
                    model_used: "None".to_string(),
                }
            }
        }
    }
    
    fn create_actionable_recommendations(
        alignment_report: &AlignmentReport,
        llm_analysis: &Option<LLMAnalysis>,
    ) -> Vec<ActionableRecommendation> {
        let mut recommendations = Vec::new();
        
        // Add recommendations based on LLM analysis
        if let Some(llm) = llm_analysis {
            for llm_rec in &llm.recommendations {
                recommendations.push(ActionableRecommendation {
                    title: llm_rec.title.clone(),
                    description: llm_rec.description.clone(),
                    priority: match llm_rec.priority {
                        LLMPriority::High => RecommendationPriority::High,
                        LLMPriority::Medium => RecommendationPriority::Medium,
                        LLMPriority::Low => RecommendationPriority::Low,
                    },
                    section: ResumeSection::Other(llm_rec.section.clone()),
                    expected_impact: llm_rec.impact.clone(),
                    specific_action: llm_rec.description.clone(),
                    before_example: None,
                    after_example: llm_rec.example.clone(),
                    effort_level: EffortLevel::Medium,
                    addresses_keywords: vec![],
                });
            }
        }
        
        // Add recommendations based on keyword analysis
        for missing_keyword in &alignment_report.gap_analysis.missing_keywords {
            recommendations.push(ActionableRecommendation {
                title: format!("Add missing keyword: {}", missing_keyword),
                description: format!("Include '{}' in your resume to match job requirements", missing_keyword),
                priority: RecommendationPriority::High,
                section: ResumeSection::Skills,
                expected_impact: "Improve ATS score and keyword matching".to_string(),
                specific_action: format!("Find relevant context to naturally include '{}' in your resume", missing_keyword),
                before_example: None,
                after_example: None,
                effort_level: EffortLevel::Low,
                addresses_keywords: vec![missing_keyword.clone()],
            });
        }
        
        recommendations
    }
    
    fn create_detailed_gap_analysis(
        alignment_report: &AlignmentReport,
        _llm_analysis: &Option<LLMAnalysis>,
    ) -> DetailedGapAnalysis {
        let mut critical_gaps = Vec::new();
        let mut important_gaps = Vec::new();
        let mut minor_gaps = Vec::new();
        
        // Categorize gaps based on priority
        for rec in &alignment_report.gap_analysis.recommendations {
            let gap_item = GapItem {
                gap_type: match rec.priority {
                    Priority::Immediate => GapType::MissingKeyword,
                    _ => GapType::MissingSkill,
                },
                title: rec.title.clone(),
                description: rec.description.clone(),
                impact: ImpactAssessment {
                    ats_impact: "Medium".to_string(),
                    human_reviewer_impact: "High".to_string(),
                    interview_impact: "Medium".to_string(),
                },
                fix_suggestion: rec.actionable_steps.join("; "),
                keywords_affected: vec![],
            };
            
            match rec.priority {
                Priority::Immediate => critical_gaps.push(gap_item),
                Priority::High => important_gaps.push(gap_item),
                _ => minor_gaps.push(gap_item),
            }
        }
        
        DetailedGapAnalysis {
            critical_gaps,
            important_gaps,
            minor_gaps,
            skills_analysis: SkillsGapAnalysis {
                missing_technical_skills: alignment_report.gap_analysis.missing_keywords.clone(),
                missing_soft_skills: vec![],
                underrepresented_skills: vec![],
                skill_level_gaps: vec![],
            },
            content_analysis: ContentGapAnalysis {
                missing_sections: alignment_report.section_analysis.missing_sections.clone(),
                weak_sections: vec![],
                content_density_issues: vec![],
                formatting_suggestions: vec![],
            },
        }
    }
    
    fn create_improvement_plan(
        recommendations: &[ActionableRecommendation],
        gap_analysis: &DetailedGapAnalysis,
    ) -> ImprovementPlan {
        let mut immediate_actions = Vec::new();
        let mut short_term_improvements = Vec::new();
        let mut long_term_development = Vec::new();
        
        // Categorize recommendations by effort level
        for rec in recommendations {
            let improvement_step = ImprovementStep {
                title: rec.title.clone(),
                description: rec.description.clone(),
                estimated_time: match rec.effort_level {
                    EffortLevel::Minimal => "< 30 minutes".to_string(),
                    EffortLevel::Low => "30 minutes - 2 hours".to_string(),
                    EffortLevel::Medium => "2-8 hours".to_string(),
                    EffortLevel::High => "1-3 days".to_string(),
                    EffortLevel::Significant => "> 3 days".to_string(),
                },
                expected_score_improvement: 5.0, // Placeholder
                resources_needed: vec![],
                success_criteria: "Improvement visible in alignment score".to_string(),
            };
            
            match rec.effort_level {
                EffortLevel::Minimal | EffortLevel::Low => immediate_actions.push(improvement_step),
                EffortLevel::Medium => short_term_improvements.push(improvement_step),
                EffortLevel::High | EffortLevel::Significant => long_term_development.push(improvement_step),
            }
        }
        
        ImprovementPlan {
            immediate_actions,
            short_term_improvements,
            long_term_development,
            cover_letter_points: vec![
                "Highlight specific technical skills mentioned in job requirements".to_string(),
                "Emphasize relevant experience and achievements".to_string(),
            ],
            interview_prep: InterviewPrepGuide {
                key_talking_points: vec![
                    "Discuss specific examples of relevant experience".to_string(),
                ],
                questions_to_prepare: vec![
                    "How does your experience align with our requirements?".to_string(),
                ],
                skill_demonstration_examples: vec![],
                potential_weaknesses_to_address: gap_analysis.critical_gaps
                    .iter()
                    .map(|gap| gap.title.clone())
                    .collect(),
            },
        }
    }
    
    fn create_metadata(
        alignment_report: &AlignmentReport,
        llm_analysis: &Option<LLMAnalysis>,
    ) -> ReportMetadata {
        ReportMetadata {
            generated_at: SystemTime::now(),
            aligner_version: env!("CARGO_PKG_VERSION").to_string(),
            resume_file: alignment_report.resume_path.clone(),
            job_file: alignment_report.job_path.clone(),
            processing_time_ms: alignment_report.processing_time_ms + 
                llm_analysis.as_ref().map(|llm| llm.processing_time_ms).unwrap_or(0),
            models_used: ModelsUsed {
                embedding_model: alignment_report.model_info.embedding_model.clone(),
                llm_model: llm_analysis.as_ref().map(|llm| llm.model_used.clone()),
                ats_keywords_count: alignment_report.model_info.ats_matcher_skills,
            },
        }
    }
}


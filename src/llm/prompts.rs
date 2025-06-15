//! Optimized LLM prompt templates for strategic resume analysis
//! Balances token efficiency with CareerForge-quality strategic output

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Strategic prompt templates optimized for local LLMs
#[derive(Debug, Clone)]
pub struct PromptTemplates {
    pub strategic_analysis: String,
    pub achievement_optimizer: String,
    pub ats_human_alignment: String,
}

impl Default for PromptTemplates {
    fn default() -> Self {
        Self {
            strategic_analysis: STRATEGIC_ANALYSIS_TEMPLATE.to_string(),
            achievement_optimizer: ACHIEVEMENT_OPTIMIZER_TEMPLATE.to_string(),
            ats_human_alignment: ATS_HUMAN_ALIGNMENT_TEMPLATE.to_string(),
        }
    }
}

/// Parameters for prompt template substitution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptParams {
    pub resume_content: String,
    pub job_content: String,
    pub missing_keywords: Vec<String>,
    pub exact_matches: Vec<String>,
    pub section_scores: HashMap<String, f32>,
    pub overall_score: f32,
    pub focus_area: Option<String>, // e.g., "technical_skills", "leadership", "ats_optimization"
}

impl PromptTemplates {
    /// Generate comprehensive strategic analysis (primary method)
    pub fn render_strategic_analysis(&self, params: &PromptParams) -> String {
        let priority_keywords = params.missing_keywords.get(0..5)
            .unwrap_or(&params.missing_keywords)
            .join(", ");
            
        let key_matches = params.exact_matches.get(0..3)
            .unwrap_or(&params.exact_matches)
            .join(", ");
        
        self.strategic_analysis
            .replace("{resume}", &Self::smart_truncate(&params.resume_content, 800))
            .replace("{job}", &Self::smart_truncate(&params.job_content, 600))
            .replace("{missing}", &priority_keywords)
            .replace("{matches}", &key_matches)
            .replace("{score}", &format!("{:.0}", params.overall_score * 100.0))
    }
    
    /// Generate achievement-focused optimization prompts
    pub fn render_achievement_optimizer(&self, params: &PromptParams) -> String {
        let focus = params.focus_area.as_deref().unwrap_or("general");
        
        self.achievement_optimizer
            .replace("{resume}", &Self::extract_experience_section(&params.resume_content, 600))
            .replace("{job}", &Self::extract_key_requirements(&params.job_content, 400))
            .replace("{focus}", focus)
            .replace("{missing}", &params.missing_keywords.get(0..3).unwrap_or(&params.missing_keywords).join(", "))
    }
    
    /// Generate ATS + human psychology alignment analysis
    pub fn render_ats_human_alignment(&self, params: &PromptParams) -> String {
        self.ats_human_alignment
            .replace("{resume}", &Self::smart_truncate(&params.resume_content, 500))
            .replace("{job}", &Self::smart_truncate(&params.job_content, 400))
            .replace("{missing}", &params.missing_keywords.get(0..4).unwrap_or(&params.missing_keywords).join(", "))
    }
    
    /// Smart content truncation that preserves structure
    fn smart_truncate(content: &str, max_chars: usize) -> String {
        if content.len() <= max_chars {
            return content.to_string();
        }
        
        // Try to break at sentence boundaries
        let truncated = &content[..max_chars];
        if let Some(last_period) = truncated.rfind('.') {
            if last_period > max_chars / 2 {
                return content[..=last_period].to_string();
            }
        }
        
        // Fall back to character limit
        format!("{}...", &content[..max_chars.saturating_sub(3)])
    }
    
    /// Extract experience section for focused analysis
    fn extract_experience_section(resume: &str, max_chars: usize) -> String {
        // Look for experience-related keywords and extract relevant sections
        let experience_indicators = ["experience", "employment", "work history", "professional", "career"];
        
        for indicator in experience_indicators {
            if let Some(start) = resume.to_lowercase().find(indicator) {
                let from_experience = &resume[start..];
                return Self::smart_truncate(from_experience, max_chars);
            }
        }
        
        // Fallback to general truncation
        Self::smart_truncate(resume, max_chars)
    }
    
    /// Extract key requirements from job posting
    fn extract_key_requirements(job: &str, max_chars: usize) -> String {
        // Look for requirements sections
        let req_indicators = ["requirements", "qualifications", "skills", "must have", "preferred"];
        
        for indicator in req_indicators {
            if let Some(start) = job.to_lowercase().find(indicator) {
                let from_reqs = &job[start..];
                return Self::smart_truncate(from_reqs, max_chars);
            }
        }
        
        Self::smart_truncate(job, max_chars)
    }
}

/// Strategic analysis template - combines gap analysis, recommendations, and optimization
const STRATEGIC_ANALYSIS_TEMPLATE: &str = r#"You are an elite ATS optimization expert and career strategist. Analyze this resume-job alignment for both automated screening and hiring manager psychology.

RESUME: {resume}
JOB: {job}
MISSING KEYWORDS: {missing}
FOUND KEYWORDS: {matches}
CURRENT SCORE: {score}/100

STRATEGIC ANALYSIS:
1. **FIT ASSESSMENT**: Score/100 with key strength and critical gap
2. **ATS OPTIMIZATION**: Top 3 keyword integration opportunities (where + how)
3. **ACHIEVEMENT TRANSFORMATION**: Convert 1-2 current bullets to achievement statements with metrics
4. **STRATEGIC POSITIONING**: How to frame candidacy as natural solution to their needs
5. **PRIORITY ACTIONS**: Most impactful changes (order by ROI)

Focus on measurable results, natural keyword flow, and psychological appeal to hiring managers."#;

/// Achievement-focused optimization for experience transformation
const ACHIEVEMENT_OPTIMIZER_TEMPLATE: &str = r#"You are a professional resume writer specializing in achievement narratives. Transform job descriptions into compelling achievement statements.

EXPERIENCE SECTION: {resume}
TARGET ROLE REQUIREMENTS: {job}
FOCUS AREA: {focus}
MISSING ELEMENTS: {missing}

TRANSFORM TO ACHIEVEMENTS:
1. **BEFORE/AFTER**: Take 2-3 current bullets and rewrite using Situation-Action-Result-Impact format
2. **QUANTIFICATION**: Add specific metrics, percentages, dollar amounts where possible
3. **KEYWORD INTEGRATION**: Naturally incorporate missing keywords into achievement context
4. **STRATEGIC FRAMING**: Position achievements to address job requirements directly

Example format: "Led [specific action] resulting in [quantified outcome] by [method/timeframe]"
Keep language active, specific, and results-focused."#;

/// ATS and human psychology alignment template
const ATS_HUMAN_ALIGNMENT_TEMPLATE: &str = r#"You are an ATS algorithm expert and hiring psychology specialist. Optimize for both automated parsing and human decision-making.

RESUME: {resume}
JOB POSTING: {job}
MISSING KEYWORDS: {missing}

DUAL OPTIMIZATION:
1. **ATS SCORING**: Keyword placement strategy (summary, skills, experience) with density recommendations
2. **HUMAN PSYCHOLOGY**: Language that triggers positive cognitive biases and creates narrative inevitability
3. **FORMAT BALANCE**: Structure that parses cleanly while maintaining visual appeal
4. **NATURAL INTEGRATION**: Seamless keyword incorporation without stuffing
5. **DECISION TRIGGERS**: Specific phrases that make hiring managers want to interview

Provide concrete placement suggestions and exact phrasing examples."#;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_strategic_analysis_rendering() {
        let templates = PromptTemplates::default();
        let params = PromptParams {
            resume_content: "Software Engineer with Python experience".to_string(),
            job_content: "Senior Software Engineer role requiring React and Python".to_string(),
            missing_keywords: vec!["React".to_string(), "Leadership".to_string()],
            exact_matches: vec!["Python".to_string(), "Software".to_string()],
            section_scores: HashMap::new(),
            overall_score: 0.75,
            focus_area: Some("technical_skills".to_string()),
        };
        
        let prompt = templates.render_strategic_analysis(&params);
        assert!(prompt.contains("React, Leadership"));
        assert!(prompt.contains("Python, Software"));
        assert!(prompt.contains("75/100"));
    }
    
    #[test]
    fn test_smart_truncation() {
        let content = "First sentence. Second sentence. Third sentence.";
        let truncated = PromptTemplates::smart_truncate(content, 20);
        assert!(truncated.ends_with("First sentence."));
        
        let short_content = "Short text";
        let not_truncated = PromptTemplates::smart_truncate(short_content, 100);
        assert_eq!(not_truncated, short_content);
    }
    
    #[test]
    fn test_experience_extraction() {
        let resume = "Education: BS Computer Science\n\nProfessional Experience:\n- Software Engineer at Tech Corp\n- Built web applications";
        let extracted = PromptTemplates::extract_experience_section(resume, 100);
        assert!(extracted.to_lowercase().contains("experience"));
        assert!(extracted.contains("Software Engineer"));
    }
}
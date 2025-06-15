//! Ultra-optimized single prompt for comprehensive resume analysis

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Single optimized prompt template
#[derive(Debug, Clone)]
pub struct PromptTemplates {
    pub combined_analysis: String,
}

impl Default for PromptTemplates {
    fn default() -> Self {
        Self {
            combined_analysis: COMBINED_ANALYSIS_TEMPLATE.to_string(),
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
    pub focus_area: Option<String>,
}

impl PromptTemplates {
    /// Generate single combined analysis (replaces all 3 methods)
    pub fn render_combined_analysis(&self, params: &PromptParams) -> String {
        let priority_keywords = params.missing_keywords.get(0..3)
            .unwrap_or(&params.missing_keywords)
            .join(", ");
            
        let key_matches = params.exact_matches.get(0..2)
            .unwrap_or(&params.exact_matches)
            .join(", ");
        
        self.combined_analysis
            .replace("{resume}", &Self::smart_truncate(&params.resume_content, 400))
            .replace("{job}", &Self::smart_truncate(&params.job_content, 300))
            .replace("{missing}", &priority_keywords)
            .replace("{matches}", &key_matches)
            .replace("{score}", &format!("{:.0}", params.overall_score * 100.0))
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
}

/// Combined template - strategic analysis + achievements + ATS in one focused prompt
const COMBINED_ANALYSIS_TEMPLATE: &str = r#"Resume: {resume}

Job: {job}

Missing: {missing} | Found: {matches} | Score: {score}/100

## STRATEGIC
Score/100, top 2 gaps, positioning strategy

## ACHIEVEMENTS  
Transform 1 bullet to metric-driven result

## ATS
2 keyword placement tips

Keep under 75 words total."#;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_combined_analysis_rendering() {
        let templates = PromptTemplates::default();
        let params = PromptParams {
            resume_content: "Software Engineer with Python experience at Tech Corp. Built web applications using Django framework.".to_string(),
            job_content: "Senior Software Engineer role requiring React, Python, and leadership skills. Remote position.".to_string(),
            missing_keywords: vec!["React".to_string(), "Leadership".to_string(), "TypeScript".to_string()],
            exact_matches: vec!["Python".to_string(), "Software".to_string(), "Engineer".to_string()],
            section_scores: HashMap::new(),
            overall_score: 0.75,
            focus_area: Some("technical_skills".to_string()),
        };
        
        let prompt = templates.render_combined_analysis(&params);
        
        // Test keyword limiting (should only include first 3 missing, first 2 matches)
        assert!(prompt.contains("React, Leadership, TypeScript"));
        assert!(prompt.contains("Python, Software"));
        assert!(prompt.contains("75/100"));
        assert!(prompt.contains("## STRATEGIC"));
        assert!(prompt.contains("## ACHIEVEMENTS"));
        assert!(prompt.contains("## ATS"));
    }
    
    #[test]
    fn test_empty_keywords_handling() {
        let templates = PromptTemplates::default();
        let params = PromptParams {
            resume_content: "Software Engineer".to_string(),
            job_content: "Software Engineer role".to_string(),
            missing_keywords: vec![],
            exact_matches: vec![],
            section_scores: HashMap::new(),
            overall_score: 0.5,
            focus_area: None,
        };
        
        let prompt = templates.render_combined_analysis(&params);
        
        // Should handle empty keywords gracefully
        assert!(prompt.contains("Missing:  |"));
        assert!(prompt.contains("Found:  |"));
        assert!(prompt.contains("50/100"));
    }
    
    #[test]
    fn test_single_keyword_handling() {
        let templates = PromptTemplates::default();
        let params = PromptParams {
            resume_content: "Developer".to_string(),
            job_content: "Developer position".to_string(),
            missing_keywords: vec!["Python".to_string()],
            exact_matches: vec!["Developer".to_string()],
            section_scores: HashMap::new(),
            overall_score: 0.8,
            focus_area: None,
        };
        
        let prompt = templates.render_combined_analysis(&params);
        
        // Should handle single keywords without trailing commas
        assert!(prompt.contains("Missing: Python |"));
        assert!(prompt.contains("Found: Developer |"));
    }
    
    #[test]
    fn test_smart_truncation_sentence_boundary() {
        let content = "First sentence. Second sentence. Third sentence that goes on for a while.";
        let truncated = PromptTemplates::smart_truncate(content, 30);
        
        // Should truncate at sentence boundary
        assert_eq!(truncated, "First sentence. Second sentence.");
        assert!(truncated.len() <= 30);
    }
    
    #[test]
    fn test_smart_truncation_no_sentence_boundary() {
        let content = "This is a very long sentence without any periods and it should be truncated";
        let truncated = PromptTemplates::smart_truncate(content, 20);
        
        // Should fall back to character limit with ellipsis
        assert!(truncated.ends_with("..."));
        assert_eq!(truncated.len(), 20);
        assert_eq!(truncated, "This is a very lo...");
    }
    
    #[test]
    fn test_smart_truncation_short_content() {
        let content = "Short text";
        let truncated = PromptTemplates::smart_truncate(content, 100);
        
        // Should return unchanged if under limit
        assert_eq!(truncated, content);
    }
    
    #[test]
    fn test_smart_truncation_sentence_too_early() {
        let content = "Short. This is a much longer sentence that goes on and on.";
        let truncated = PromptTemplates::smart_truncate(content, 30);
        
        // Should not break at very early sentence if it's less than half the limit
        assert!(truncated.ends_with("..."));
        assert_eq!(truncated.len(), 30);
    }
    
    #[test]
    fn test_prompt_template_creation() {
        let templates = PromptTemplates::default();
        assert!(!templates.combined_analysis.is_empty());
        assert!(templates.combined_analysis.contains("## STRATEGIC"));
        assert!(templates.combined_analysis.contains("## ACHIEVEMENTS"));
        assert!(templates.combined_analysis.contains("## ATS"));
    }
    
    #[test]
    fn test_score_formatting() {
        let templates = PromptTemplates::default();
        let params = PromptParams {
            resume_content: "Test".to_string(),
            job_content: "Test".to_string(),
            missing_keywords: vec![],
            exact_matches: vec![],
            section_scores: HashMap::new(),
            overall_score: 0.876, // Test decimal rounding
            focus_area: None,
        };
        
        let prompt = templates.render_combined_analysis(&params);
        assert!(prompt.contains("88/100")); // Should round 87.6 to 88
    }
    
    #[test]
    fn test_large_keyword_lists() {
        let templates = PromptTemplates::default();
        let params = PromptParams {
            resume_content: "Test".to_string(),
            job_content: "Test".to_string(),
            missing_keywords: vec![
                "Python".to_string(), "React".to_string(), "Node.js".to_string(),
                "TypeScript".to_string(), "MongoDB".to_string(), "AWS".to_string()
            ],
            exact_matches: vec![
                "JavaScript".to_string(), "HTML".to_string(), "CSS".to_string(),
                "Git".to_string(), "Docker".to_string()
            ],
            section_scores: HashMap::new(),
            overall_score: 0.65,
            focus_area: None,
        };
        
        let prompt = templates.render_combined_analysis(&params);
        
        // Should limit to first 3 missing and first 2 matches
        assert!(prompt.contains("Python, React, Node.js"));
        assert!(prompt.contains("JavaScript, HTML"));
        assert!(!prompt.contains("TypeScript")); // Should not include 4th missing keyword
        assert!(!prompt.contains("CSS")); // Should not include 3rd match
    }
    
    #[test]
    fn test_content_truncation_limits() {
        let templates = PromptTemplates::default();
        let long_resume = "A".repeat(1000); // 1000 characters
        let long_job = "B".repeat(800); // 800 characters
        
        let params = PromptParams {
            resume_content: long_resume,
            job_content: long_job,
            missing_keywords: vec!["test".to_string()],
            exact_matches: vec!["match".to_string()],
            section_scores: HashMap::new(),
            overall_score: 0.5,
            focus_area: None,
        };
        
        let prompt = templates.render_combined_analysis(&params);
        
        // Resume should be truncated to ~400 chars, job to ~300 chars
        let resume_section = prompt.find("Resume: ").unwrap();
        let job_section = prompt.find("Job: ").unwrap();
        let resume_content = &prompt[resume_section + 8..job_section].trim();
        
        assert!(resume_content.len() <= 403); // 400 + "..." = 403
        
        // Verify structure is maintained
        assert!(prompt.contains("Missing: test"));
        assert!(prompt.contains("Found: match"));
    }
}
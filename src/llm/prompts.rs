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
}

impl PromptTemplates {
    /// Generate single combined analysis with enhanced content selection
    pub fn render_combined_analysis(&self, params: &PromptParams) -> String {
        // Debug logging to see what we're getting
        println!("🔍 DEBUG - Resume content length: {}", params.resume_content.len());
        println!("🔍 DEBUG - Job content length: {}", params.job_content.len());
        println!("🔍 DEBUG - Resume preview (first 200 chars): {}", 
            params.resume_content.chars().take(200).collect::<String>());
        println!("🔍 DEBUG - Job preview (first 200 chars): {}", 
            params.job_content.chars().take(200).collect::<String>());
        
        let final_prompt = self.combined_analysis
            .replace("{resume}", &params.resume_content)
            .replace("{job}", &params.job_content);
        
        println!("🔍 DEBUG - Final prompt length: {}", final_prompt.len());
        println!("🔍 DEBUG - Final prompt preview (first 500 chars):\n{}", 
            final_prompt.chars().take(500).collect::<String>());
        println!("🔍 DEBUG - Final prompt end (last 200 chars):\n{}", 
            final_prompt.chars().skip(final_prompt.len().saturating_sub(200)).collect::<String>());
        
        final_prompt
    }
}

const COMBINED_ANALYSIS_TEMPLATE: &str = r#"TASK: Analyze the provided resume for the target job and give specific improvement recommendations.

<RESUME>
{resume}
</RESUME>

<JOB POSTING>
{job}
</JOB POSTING>

Provide your analysis in the following format:

## STRATEGIC ASSESSMENT
Based on the resume above, what are the 3 biggest gaps for this specific job?

## ACHIEVEMENT TRANSFORMATION
Take one bullet point from the resume above and rewrite it with better metrics.

## ATS OPTIMIZATION
What keywords should be added to improve ATS matching?

## PRIORITY ACTIONS
What are the top 3 changes to make to this specific resume?

IMPORTANT: Reference the actual resume content above, not generic advice."#;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_combined_analysis_rendering() {
        let templates = PromptTemplates::default();
        let params = PromptParams {
            resume_content: "Software Engineer with Python experience at Tech Corp.".to_string(),
            job_content: "Senior Software Engineer role requiring React and Python.".to_string(),
            missing_keywords: vec![], 
            exact_matches: vec![],     
            section_scores: HashMap::new(), 
            overall_score: 0.75,      
            focus_area: None,         
        };
        
        let prompt = templates.render_combined_analysis(&params);
        
        // Test only resume and job content inclusion
        assert!(prompt.contains("Software Engineer with Python experience at Tech Corp"));
        assert!(prompt.contains("Senior Software Engineer role requiring React and Python"));
        assert!(prompt.contains("## STRATEGIC"));
        assert!(prompt.contains("## ACHIEVEMENT")); // Fixed: singular, not plural
    }
    
    #[test]
    fn test_prompt_template_creation() {
        let templates = PromptTemplates::default();
        assert!(!templates.combined_analysis.is_empty());
        assert!(templates.combined_analysis.contains("## STRATEGIC"));
        assert!(templates.combined_analysis.contains("## ACHIEVEMENT")); // Fixed: singular
        assert!(templates.combined_analysis.contains("## ATS"));
    }
    
    #[test]
    fn test_basic_replacement() {
        let templates = PromptTemplates::default();
        let params = PromptParams {
            resume_content: "Test Resume Content".to_string(),
            job_content: "Test Job Content".to_string(),
            missing_keywords: vec![],
            exact_matches: vec![],
            section_scores: HashMap::new(),
            overall_score: 0.0,
            focus_area: None,
        };
        
        let prompt = templates.render_combined_analysis(&params);
        
        // Verify basic template replacement works
        assert!(prompt.contains("Test Resume Content"));
        assert!(prompt.contains("Test Job Content"));
        assert!(prompt.contains("<RESUME>"));
        assert!(prompt.contains("</RESUME>"));
        assert!(prompt.contains("<JOB POSTING>"));
        assert!(prompt.contains("</JOB POSTING>"));
    }
}
//! LLM prompt templates for resume analysis and recommendations

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Collection of prompt templates for different analysis tasks
#[derive(Debug, Clone)]
pub struct PromptTemplates {
    pub gap_analysis: String,
    pub skill_extraction: String,
    pub recommendations: String,
    pub resume_improvement: String,
    pub section_analysis: String,
    pub keyword_optimization: String,
}

impl Default for PromptTemplates {
    fn default() -> Self {
        Self {
            gap_analysis: GAP_ANALYSIS_TEMPLATE.to_string(),
            skill_extraction: SKILL_EXTRACTION_TEMPLATE.to_string(),
            recommendations: RECOMMENDATIONS_TEMPLATE.to_string(),
            resume_improvement: RESUME_IMPROVEMENT_TEMPLATE.to_string(),
            section_analysis: SECTION_ANALYSIS_TEMPLATE.to_string(),
            keyword_optimization: KEYWORD_OPTIMIZATION_TEMPLATE.to_string(),
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
    pub specific_instruction: Option<String>,
}

impl PromptTemplates {
    /// Generate a gap analysis prompt
    pub fn render_gap_analysis(&self, params: &PromptParams) -> String {
        let missing_keywords_str = if params.missing_keywords.is_empty() {
            "No significant missing keywords identified.".to_string()
        } else {
            format!(
                "Missing keywords: {}",
                params.missing_keywords.join(", ")
            )
        };
        
        let exact_matches_str = if params.exact_matches.is_empty() {
            "No exact keyword matches found.".to_string()
        } else {
            format!(
                "Found keywords: {}",
                params.exact_matches.join(", ")
            )
        };
        
        self.gap_analysis
            .replace("{resume_content}", &Self::truncate_content(&params.resume_content, 2000))
            .replace("{job_content}", &Self::truncate_content(&params.job_content, 2000))
            .replace("{missing_keywords}", &missing_keywords_str)
            .replace("{exact_matches}", &exact_matches_str)
            .replace("{overall_score}", &format!("{:.1}%", params.overall_score * 100.0))
    }
    
    /// Generate a recommendations prompt
    pub fn render_recommendations(&self, params: &PromptParams) -> String {
        let section_scores_str = params.section_scores
            .iter()
            .map(|(name, score)| format!("{}: {:.1}%", name, score * 100.0))
            .collect::<Vec<_>>()
            .join("\n");
        
        self.recommendations
            .replace("{resume_content}", &Self::truncate_content(&params.resume_content, 1500))
            .replace("{job_content}", &Self::truncate_content(&params.job_content, 1500))
            .replace("{section_scores}", &section_scores_str)
            .replace("{missing_keywords}", &params.missing_keywords.join(", "))
            .replace("{overall_score}", &format!("{:.1}%", params.overall_score * 100.0))
    }
    
    /// Generate a skill extraction prompt
    pub fn render_skill_extraction(&self, params: &PromptParams) -> String {
        self.skill_extraction
            .replace("{resume_content}", &Self::truncate_content(&params.resume_content, 2500))
            .replace("{job_content}", &Self::truncate_content(&params.job_content, 2500))
    }
    
    /// Generate a resume improvement prompt
    pub fn render_resume_improvement(&self, params: &PromptParams) -> String {
        let default_instruction = "Provide general improvement suggestions.".to_string();
        let instruction = params.specific_instruction
            .as_ref()
            .unwrap_or(&default_instruction);
        
        self.resume_improvement
            .replace("{resume_content}", &Self::truncate_content(&params.resume_content, 2000))
            .replace("{job_content}", &Self::truncate_content(&params.job_content, 2000))
            .replace("{specific_instruction}", instruction)
            .replace("{missing_keywords}", &params.missing_keywords.join(", "))
    }
    
    /// Generate a section analysis prompt
    pub fn render_section_analysis(&self, params: &PromptParams) -> String {
        self.section_analysis
            .replace("{resume_content}", &Self::truncate_content(&params.resume_content, 2500))
            .replace("{job_content}", &Self::truncate_content(&params.job_content, 1500))
    }
    
    /// Generate a keyword optimization prompt
    pub fn render_keyword_optimization(&self, params: &PromptParams) -> String {
        self.keyword_optimization
            .replace("{resume_content}", &Self::truncate_content(&params.resume_content, 2000))
            .replace("{job_content}", &Self::truncate_content(&params.job_content, 2000))
            .replace("{missing_keywords}", &params.missing_keywords.join(", "))
            .replace("{exact_matches}", &params.exact_matches.join(", "))
    }
    
    /// Truncate content to fit within token limits
    fn truncate_content(content: &str, max_chars: usize) -> String {
        if content.len() <= max_chars {
            content.to_string()
        } else {
            format!("{}...[content truncated]", &content[..max_chars])
        }
    }
}

/// Template for gap analysis between resume and job description
const GAP_ANALYSIS_TEMPLATE: &str = r#"
You are an expert resume analyst. Analyze the gap between this resume and job description to identify missing elements and opportunities for improvement.

RESUME:
{resume_content}

JOB DESCRIPTION:
{job_content}

CURRENT ANALYSIS:
- Overall Match Score: {overall_score}
- {missing_keywords}
- {exact_matches}

PROVIDE:
1. **Key Gaps**: List 3-5 most important missing elements
2. **Skills Assessment**: Which required skills are missing or understated?
3. **Experience Alignment**: How well does experience match requirements?
4. **Impact Analysis**: Rate each gap's impact on candidacy (High/Medium/Low)

Keep your analysis concise, specific, and actionable. Focus on the most impactful improvements.
"#;

/// Template for skill extraction and comparison
const SKILL_EXTRACTION_TEMPLATE: &str = r#"
You are a technical recruiter analyzing skills alignment. Extract and compare skills between the resume and job requirements.

RESUME:
{resume_content}

JOB REQUIREMENTS:
{job_content}

EXTRACT AND CATEGORIZE:
1. **Technical Skills**: Programming languages, frameworks, tools
2. **Soft Skills**: Leadership, communication, teamwork
3. **Domain Skills**: Industry-specific knowledge
4. **Required vs. Present**: For each job requirement, indicate if it's present in the resume

FORMAT:
- ✅ Skill Name (Strong match)
- ⚠️ Skill Name (Partial match/needs strengthening)
- ❌ Skill Name (Missing from resume)

Be specific about technical versions, experience levels, and context when relevant.
"#;

/// Template for generating actionable recommendations
const RECOMMENDATIONS_TEMPLATE: &str = r#"
You are a career coach providing specific, actionable recommendations to improve resume alignment with this job opportunity.

RESUME:
{resume_content}

JOB OPPORTUNITY:
{job_content}

CURRENT STATUS:
- Overall Score: {overall_score}
- Section Scores:
{section_scores}
- Missing Keywords: {missing_keywords}

PROVIDE 5 SPECIFIC RECOMMENDATIONS:
For each recommendation, include:
1. **Action**: What to change/add
2. **Location**: Which resume section
3. **Priority**: High/Medium/Low
4. **Impact**: Expected improvement
5. **Example**: Specific language or phrasing

Focus on high-impact changes that directly address job requirements. Be specific about wording and placement.
"#;

/// Template for resume improvement suggestions
const RESUME_IMPROVEMENT_TEMPLATE: &str = r#"
You are a professional resume writer. Provide specific improvements to better align this resume with the target position.

CURRENT RESUME:
{resume_content}

TARGET POSITION:
{job_content}

SPECIAL FOCUS:
{specific_instruction}

MISSING ELEMENTS:
{missing_keywords}

PROVIDE:
1. **Content Improvements**: What information to add, remove, or modify
2. **Keyword Integration**: How to naturally incorporate missing keywords
3. **Structure Suggestions**: Section organization and formatting
4. **Quantification Opportunities**: Where to add metrics and achievements
5. **ATS Optimization**: Specific formatting and keyword placement tips

Make suggestions concrete and implementable. Include before/after examples where helpful.
"#;

/// Template for section-by-section analysis
const SECTION_ANALYSIS_TEMPLATE: &str = r#"
You are a resume expert conducting a detailed section-by-section analysis to optimize alignment with job requirements.

RESUME:
{resume_content}

JOB REQUIREMENTS:
{job_content}

ANALYZE EACH SECTION:
1. **Professional Summary/Objective**
   - Alignment with job requirements
   - Key messages and positioning
   - Suggestions for improvement

2. **Experience Section**
   - Relevance of roles and responsibilities
   - Achievement quantification
   - Keyword optimization opportunities

3. **Skills Section**
   - Technical skills coverage
   - Missing required skills
   - Organization and presentation

4. **Education/Certifications**
   - Relevance to position
   - Additional certifications needed

5. **Other Sections**
   - Value and relevance
   - Potential additions or removals

Provide specific, actionable feedback for each section with priority ratings.
"#;

/// Template for keyword optimization
const KEYWORD_OPTIMIZATION_TEMPLATE: &str = r#"
You are an ATS (Applicant Tracking System) optimization expert. Help improve keyword alignment between this resume and job posting.

RESUME:
{resume_content}

JOB POSTING:
{job_content}

KEYWORD ANALYSIS:
- Missing Keywords: {missing_keywords}
- Found Keywords: {exact_matches}

PROVIDE:
1. **Keyword Integration Strategy**:
   - Where to place each missing keyword naturally
   - Context and phrasing suggestions
   - Section-specific recommendations

2. **Natural Language Integration**:
   - How to incorporate keywords without keyword stuffing
   - Synonym and variation strategies
   - Industry-appropriate terminology

3. **ATS Optimization Tips**:
   - Formatting considerations
   - Keyword density and placement
   - Common ATS pitfalls to avoid

4. **Priority Keywords**:
   - Rank missing keywords by importance
   - Essential vs. nice-to-have keywords

Focus on maintaining readability while optimizing for ATS systems.
"#;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_prompt_templates_creation() {
        let templates = PromptTemplates::default();
        assert!(!templates.gap_analysis.is_empty());
        assert!(!templates.recommendations.is_empty());
        assert!(!templates.skill_extraction.is_empty());
    }
    
    #[test]
    fn test_gap_analysis_rendering() {
        let templates = PromptTemplates::default();
        let params = PromptParams {
            resume_content: "Software Engineer with 5 years experience".to_string(),
            job_content: "Looking for Senior Software Engineer".to_string(),
            missing_keywords: vec!["Python".to_string(), "React".to_string()],
            exact_matches: vec!["Software".to_string(), "Engineer".to_string()],
            section_scores: HashMap::new(),
            overall_score: 0.75,
            specific_instruction: None,
        };
        
        let prompt = templates.render_gap_analysis(&params);
        assert!(prompt.contains("Software Engineer with 5 years experience"));
        assert!(prompt.contains("Looking for Senior Software Engineer"));
        assert!(prompt.contains("Python, React"));
        assert!(prompt.contains("75.0%"));
    }
    
    #[test]
    fn test_content_truncation() {
        let long_content = "a".repeat(3000);
        let truncated = PromptTemplates::truncate_content(&long_content, 100);
        // Should be exactly 100 chars + "...[content truncated]" = 122 chars
        assert_eq!(truncated.len(), 122);
        assert!(truncated.ends_with("...[content truncated]"));
        assert!(truncated.starts_with("aaaa")); // Should start with the original content
    }
}

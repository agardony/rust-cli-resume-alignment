//! Output formatters for Phase 6 - Multiple format support with rich presentation

use crate::config::OutputFormat;
use crate::error::Result;
use crate::output::report::*;
use askama::Template;
use colored::{Colorize, Color};
use serde_json;
use std::path::Path;

/// Trait for formatting comprehensive reports
pub trait OutputFormatter {
    fn format_report(&self, report: &ComprehensiveReport) -> Result<String>;
    fn supports_format(&self) -> OutputFormat;
}

/// Enhanced console formatter with colors and rich presentation
pub struct ConsoleFormatter {
    use_colors: bool,
    detailed: bool,
}

/// JSON formatter for API integration and structured data
pub struct JsonFormatter {
    pretty: bool,
}

/// Markdown formatter for documentation and reports
pub struct MarkdownFormatter {
    include_metadata: bool,
}

/// HTML formatter with professional styling
pub struct HtmlFormatter {
    include_styles: bool,
    _template: HtmlTemplate,
}

/// PDF formatter for professional reports
pub struct PdfFormatter {
    _include_charts: bool,
}

/// Report generator that coordinates different formatters
pub struct ReportGenerator {
    console_formatter: ConsoleFormatter,
    json_formatter: JsonFormatter,
    markdown_formatter: MarkdownFormatter,
    html_formatter: HtmlFormatter,
    pdf_formatter: PdfFormatter,
}

/// Askama template for HTML output
#[derive(Template)]
#[template(source = r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Resume Alignment Analysis Report</title>
    {% if include_styles %}
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: #f8f9fa;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 3px solid #007acc;
            padding-bottom: 20px;
        }
        .score-badge {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            color: white;
            margin-left: 10px;
        }
        .score-excellent { background: #28a745; }
        .score-good { background: #17a2b8; }
        .score-fair { background: #ffc107; color: #000; }
        .score-poor { background: #dc3545; }
        .section {
            margin: 25px 0;
        }
        .section h2 {
            color: #007acc;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 10px;
        }
        .section h3 {
            color: #495057;
            margin-top: 20px;
        }
        .score-breakdown {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .score-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #007acc;
        }
        .recommendations {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 6px;
            margin: 15px 0;
        }
        .recommendation {
            background: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 6px;
            border-left: 4px solid #17a2b8;
        }
        .priority-critical { border-left-color: #dc3545; }
        .priority-high { border-left-color: #ffc107; }
        .priority-medium { border-left-color: #17a2b8; }
        .priority-low { border-left-color: #28a745; }
        .strengths, .improvements {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            margin: 10px 0;
        }
        .strengths { border-left: 4px solid #28a745; }
        .improvements { border-left: 4px solid #ffc107; }
        ul { margin: 10px 0; }
        li { margin: 5px 0; }
        .metadata {
            background: #e9ecef;
            padding: 15px;
            border-radius: 6px;
            margin-top: 30px;
            font-size: 0.9em;
            color: #6c757d;
        }
        .improvement-plan {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .plan-section {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
        }
        .plan-section h4 {
            margin-top: 0;
            color: #007acc;
        }
    </style>
    {% endif %}
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Resume Alignment Analysis Report</h1>
            <p>Generated: {{ generated_at }} | Processing time: {{ processing_time }}ms</p>
        </div>
        
        <div class="section">
            <h2>Executive Summary</h2>
            <h3>Overall Score: {{ overall_score }}% <span class="score-badge {{ score_class }}">{{ score_label }}</span></h3>
            <p><strong>Verdict:</strong> {{ verdict }}</p>
            
            <div class="score-breakdown">
                <div class="score-item">
                    <h4>🎯 Semantic Alignment</h4>
                    <p><strong>{{ embedding_score }}%</strong> (weight: {{ embedding_weight }}%)</p>
                </div>
                <div class="score-item">
                    <h4>🔍 Keyword Matching</h4>
                    <p><strong>{{ keyword_score }}%</strong> (weight: {{ keyword_weight }}%)</p>
                </div>
                {% if has_llm_score %}
                <div class="score-item">
                    <h4>🤖 AI Analysis</h4>
                    <p><strong>{{ llm_score }}%</strong> (weight: {{ llm_weight }}%)</p>
                </div>
                {% endif %}
            </div>
        </div>
        
        <div class="section">
            <h2>✅ Key Strengths</h2>
            <div class="strengths">
                {{ strengths_html | safe }}
            </div>
        </div>
        
        <div class="section">
            <h2>🎯 Areas for Improvement</h2>
            <div class="improvements">
                {{ improvement_areas_html | safe }}
            </div>
        </div>
        
        {% if has_llm_summary %}
        <div class="section">
            <h2>🤖 AI Analysis Insights</h2>
            <p><strong>Model:</strong> {{ llm_model }} | <strong>Confidence:</strong> {{ llm_confidence }}%</p>
            <blockquote>{{ llm_summary }}</blockquote>
        </div>
        {% endif %}
        
        <div class="section">
            <h2>📋 Top Recommendations</h2>
            <div class="recommendations">
                {{ top_recommendations_html | safe }}
            </div>
        </div>
        
        <div class="section">
            <h2>🚀 Improvement Plan</h2>
            <div class="improvement-plan">
                {% if has_immediate_actions %}
                <div class="plan-section">
                    <h4>Quick Wins (< 1 hour)</h4>
                    {{ immediate_actions_html | safe }}
                </div>
                {% endif %}
                
                {% if has_short_term_improvements %}
                <div class="plan-section">
                    <h4>Short-term (1-3 days)</h4>
                    {{ short_term_improvements_html | safe }}
                </div>
                {% endif %}
                
                {% if has_cover_letter_points %}
                <div class="plan-section">
                    <h4>💌 Cover Letter Points</h4>
                    {{ cover_letter_points_html | safe }}
                </div>
                {% endif %}
            </div>
        </div>
        
        <div class="metadata">
            <p><strong>ℹ️ Generated by Resume Aligner v{{ version }}</strong></p>
            <p><strong>Models:</strong> {{ embedding_model }} + {{ llm_model_name }}</p>
            <p><strong>Resume:</strong> {{ resume_file }} | <strong>Job:</strong> {{ job_file }}</p>
        </div>
    </div>
</body>
</html>"#, ext = "html")]
struct HtmlTemplate {
    include_styles: bool,
    generated_at: String,
    processing_time: u64,
    overall_score: u8,
    score_class: String,
    score_label: String,
    verdict: String,
    embedding_score: u8,
    embedding_weight: String,
    keyword_score: u8,
    keyword_weight: String,
    llm_score: u8,  // Changed from Option<u8> to u8 with 0 as default
    has_llm_score: bool,  // Added flag to check if LLM score exists
    llm_weight: String,
    strengths_html: String,  // Changed from Vec<String> to pre-formatted HTML string
    improvement_areas_html: String,  // Changed from Vec<String> to pre-formatted HTML string
    llm_summary: String,  // Changed from Option<String> to String (empty if none)
    has_llm_summary: bool,  // Added flag to check if LLM summary exists
    llm_model: String,
    llm_confidence: String,
    top_recommendations_html: String,  // Changed from Vec<HtmlRecommendation> to pre-formatted HTML
    immediate_actions_html: String,  // Changed from Vec<HtmlAction> to pre-formatted HTML
    short_term_improvements_html: String,  // Changed from Vec<HtmlAction> to pre-formatted HTML
    cover_letter_points_html: String,  // Changed from Vec<String> to pre-formatted HTML
    has_immediate_actions: bool,
    has_short_term_improvements: bool,
    has_cover_letter_points: bool,
    version: String,
    embedding_model: String,
    llm_model_name: String,
    resume_file: String,
    job_file: String,
}

#[derive(Debug, Clone)]
struct HtmlRecommendation {
    priority_icon: String,
    priority_class: String,
    title: String,
    section: String,
    impact: String,
    description: String,
    example: Option<String>,
}

#[derive(Debug, Clone)]
struct HtmlAction {
    title: String,
    time: String,
    description: String,
}

impl ConsoleFormatter {
    pub fn new(use_colors: bool, detailed: bool) -> Self {
        Self { use_colors, detailed }
    }
    
    fn colorize(&self, text: &str, color: Color) -> String {
        if self.use_colors {
            text.color(color).to_string()
        } else {
            text.to_string()
        }
    }
    
    fn format_header(&self, title: &str, level: u8) -> String {
        let prefix = match level {
            1 => "█",
            2 => "▓",
            3 => "▒",
            _ => "░",
        };
        
        let color = match level {
            1 => Color::Blue,
            2 => Color::Green,
            3 => Color::Yellow,
            _ => Color::White,
        };
        
        if self.use_colors {
            format!("\n{} {}\n", prefix.color(color).bold(), title.color(color).bold())
        } else {
            format!("\n{} {}\n", prefix, title)
        }
    }
    
    fn format_score_badge(&self, score: u8) -> String {
        let (badge, color) = match score {
            90..=100 => ("EXCELLENT", Color::Green),
            80..=89 => ("VERY GOOD", Color::BrightGreen),
            70..=79 => ("GOOD", Color::Yellow),
            60..=69 => ("FAIR", Color::BrightYellow),
            50..=59 => ("BELOW AVG", Color::Red),
            _ => ("POOR", Color::BrightRed),
        };
        
        if self.use_colors {
            format!("[{}]", badge.color(color).bold())
        } else {
            format!("[{}]", badge)
        }
    }
    
    fn format_priority_icon(&self, priority: &RecommendationPriority) -> String {
        let (icon, _color) = match priority {
            RecommendationPriority::Critical => ("🚨", Color::Red),
            RecommendationPriority::High => ("⚠️", Color::Yellow),
            RecommendationPriority::Medium => ("📋", Color::Blue),
            RecommendationPriority::Low => ("💡", Color::Green),
        };
        
        if self.use_colors {
            format!("{} ", icon)
        } else {
            let text_icon = match priority {
                RecommendationPriority::Critical => "[!]",
                RecommendationPriority::High => "[*]",
                RecommendationPriority::Medium => "[-]",
                RecommendationPriority::Low => "[+]",
            };
            format!("{} ", text_icon)
        }
    }
}

impl OutputFormatter for ConsoleFormatter {
    fn format_report(&self, report: &ComprehensiveReport) -> Result<String> {
        let mut output = String::new();
        
        // Header
        output.push_str(&self.format_header("📊 RESUME ALIGNMENT ANALYSIS", 1));
        output.push_str(&format!("Generated: {} | Processing time: {}ms\n", 
            chrono::DateTime::<chrono::Utc>::from(report.metadata.generated_at)
                .format("%Y-%m-%d %H:%M:%S UTC"),
            report.metadata.processing_time_ms
        ));
        
        // Executive Summary
        output.push_str(&self.format_header("Executive Summary", 2));
        let score_badge = self.format_score_badge(report.analysis_summary.overall_score_percentage);
        output.push_str(&format!("Overall Score: {}% {}\n", 
            report.analysis_summary.overall_score_percentage, 
            score_badge
        ));
        output.push_str(&format!("Verdict: {}\n\n", 
            self.colorize(&report.analysis_summary.verdict, Color::Cyan)
        ));
        
        // Score Breakdown
        output.push_str(&self.format_header("Score Breakdown", 3));
        output.push_str(&format!("🎯 Embedding Score: {}% (weight: {:.1}%)\n", 
            report.analysis_summary.score_breakdown.embedding_score,
            report.analysis_summary.score_breakdown.component_weights.embedding_weight * 100.0
        ));
        output.push_str(&format!("🔍 Keyword Score: {}% (weight: {:.1}%)\n", 
            report.analysis_summary.score_breakdown.keyword_score,
            report.analysis_summary.score_breakdown.component_weights.keyword_weight * 100.0
        ));
        if let Some(llm_score) = report.analysis_summary.score_breakdown.llm_score {
            output.push_str(&format!("🤖 LLM Score: {}% (weight: {:.1}%)\n", 
                llm_score,
                report.analysis_summary.score_breakdown.component_weights.llm_weight * 100.0
            ));
        }
        output.push('\n');
        
        // Strengths
        if !report.analysis_summary.strengths.is_empty() {
            output.push_str(&self.format_header("✅ Key Strengths", 3));
            for strength in &report.analysis_summary.strengths {
                output.push_str(&format!("  • {}\n", self.colorize(strength, Color::Green)));
            }
            output.push('\n');
        }
        
        // Improvement Areas
        if !report.analysis_summary.improvement_areas.is_empty() {
            output.push_str(&self.format_header("🎯 Improvement Areas", 3));
            for area in &report.analysis_summary.improvement_areas {
                output.push_str(&format!("  • {}\n", self.colorize(area, Color::Yellow)));
            }
            output.push('\n');
        }
        
        // LLM Insights
        if !report.llm_insights.natural_language_summary.is_empty() && report.llm_insights.model_used != "None" {
            output.push_str(&self.format_header("🤖 AI Analysis Insights", 2));
            output.push_str(&format!("Model: {}\n", report.llm_insights.model_used));
            output.push_str(&format!("Confidence: {:.1}%\n\n", report.llm_insights.llm_confidence * 100.0));
            output.push_str(&format!("{} {}\n\n", 
                self.colorize("Summary:", Color::Cyan),
                report.llm_insights.natural_language_summary
            ));
        }
        
        // Top Recommendations
        output.push_str(&self.format_header("📋 Top Recommendations", 2));
        let top_recommendations = report.recommendations
            .iter()
            .filter(|r| matches!(r.priority, RecommendationPriority::Critical | RecommendationPriority::High))
            .take(5);
        
        for (i, rec) in top_recommendations.enumerate() {
            output.push_str(&format!("{}. {}{} {}\n", 
                i + 1,
                self.format_priority_icon(&rec.priority),
                self.colorize(&rec.title, Color::White),
                self.colorize(&format!("({})", Self::format_section(&rec.section)), Color::BrightBlack)
            ));
            output.push_str(&format!("   {}\n", rec.description));
            if let Some(example) = &rec.after_example {
                output.push_str(&format!("   💡 Example: {}\n", self.colorize(example, Color::Green)));
            }
            output.push('\n');
        }
        
        // Critical Gaps (if any)
        if !report.gap_analysis.critical_gaps.is_empty() {
            output.push_str(&self.format_header("🚨 Critical Gaps", 2));
            for gap in &report.gap_analysis.critical_gaps {
                output.push_str(&format!("• {} {}\n", 
                    self.colorize(&gap.title, Color::Red),
                    self.colorize(&format!("({})", gap.description), Color::BrightBlack)
                ));
                output.push_str(&format!("  Fix: {}\n\n", gap.fix_suggestion));
            }
        }
        
        // Improvement Plan
        output.push_str(&self.format_header("🚀 Immediate Action Plan", 2));
        if !report.improvement_plan.immediate_actions.is_empty() {
            output.push_str(&self.colorize("Quick Wins (< 1 hour):\n", Color::Green));
            for (i, action) in report.improvement_plan.immediate_actions.iter().take(3).enumerate() {
                output.push_str(&format!("  {}. {} ({})\n", 
                    i + 1, 
                    action.title, 
                    action.estimated_time
                ));
            }
            output.push('\n');
        }
        
        if self.detailed {
            // Detailed Analysis (only in detailed mode)
            output.push_str(&self.format_header("📊 Detailed Analysis", 2));
            
            // All Recommendations
            if report.recommendations.len() > 5 {
                output.push_str(&self.format_header("All Recommendations", 3));
                for (i, rec) in report.recommendations.iter().enumerate() {
                    output.push_str(&format!("{}. {}{} [{}]\n", 
                        i + 1,
                        self.format_priority_icon(&rec.priority),
                        rec.title,
                        Self::format_effort(&rec.effort_level)
                    ));
                    output.push_str(&format!("   Section: {} | Impact: {}\n", 
                        Self::format_section(&rec.section),
                        rec.expected_impact
                    ));
                    output.push_str(&format!("   Action: {}\n", rec.specific_action));
                    if !rec.addresses_keywords.is_empty() {
                        output.push_str(&format!("   Keywords: {}\n", rec.addresses_keywords.join(", ")));
                    }
                    output.push('\n');
                }
            }
            
            // Gap Analysis Details
            output.push_str(&self.format_header("Gap Analysis Details", 3));
            if !report.gap_analysis.skills_analysis.missing_technical_skills.is_empty() {
                output.push_str(&format!("Missing Technical Skills: {}\n", 
                    report.gap_analysis.skills_analysis.missing_technical_skills.join(", ")
                ));
            }
            if !report.gap_analysis.content_analysis.missing_sections.is_empty() {
                output.push_str(&format!("Missing Sections: {}\n", 
                    report.gap_analysis.content_analysis.missing_sections.join(", ")
                ));
            }
        }
        
        // Footer
        output.push_str(&format!("\n{} Generated by Resume Aligner v{} | Models: {} + {}\n", 
            self.colorize("ℹ️", Color::Blue),
            report.metadata.aligner_version,
            report.metadata.models_used.embedding_model,
            report.metadata.models_used.llm_model.as_deref().unwrap_or("ATS-only")
        ));
        
        Ok(output)
    }
    
    fn supports_format(&self) -> OutputFormat {
        OutputFormat::Console
    }
}

impl ConsoleFormatter {
    fn format_section(section: &ResumeSection) -> String {
        match section {
            ResumeSection::Summary => "Summary".to_string(),
            ResumeSection::Experience => "Experience".to_string(),
            ResumeSection::Skills => "Skills".to_string(),
            ResumeSection::Education => "Education".to_string(),
            ResumeSection::Projects => "Projects".to_string(),
            ResumeSection::Certifications => "Certifications".to_string(),
            ResumeSection::Other(name) => name.clone(),
        }
    }
    
    fn format_effort(effort: &EffortLevel) -> String {
        match effort {
            EffortLevel::Minimal => "Quick",
            EffortLevel::Low => "Easy",
            EffortLevel::Medium => "Moderate",
            EffortLevel::High => "Significant",
            EffortLevel::Significant => "Major",
        }.to_string()
    }
}

impl JsonFormatter {
    pub fn new(pretty: bool) -> Self {
        Self { pretty }
    }
}

impl OutputFormatter for JsonFormatter {
    fn format_report(&self, report: &ComprehensiveReport) -> Result<String> {
        if self.pretty {
            Ok(serde_json::to_string_pretty(report)?)
        } else {
            Ok(serde_json::to_string(report)?)
        }
    }
    
    fn supports_format(&self) -> OutputFormat {
        OutputFormat::Json
    }
}

impl HtmlFormatter {
    pub fn new(include_styles: bool) -> Self {
        // We'll create a dummy template here since we'll populate it in format_report
        let template = HtmlTemplate {
            include_styles,
            generated_at: String::new(),
            processing_time: 0,
            overall_score: 0,
            score_class: String::new(),
            score_label: String::new(),
            verdict: String::new(),
            embedding_score: 0,
            embedding_weight: String::new(),
            keyword_score: 0,
            keyword_weight: String::new(),
            llm_score: 0,
            has_llm_score: false,
            llm_weight: String::new(),
            strengths_html: String::new(),
            improvement_areas_html: String::new(),
            llm_summary: String::new(),
            has_llm_summary: false,
            llm_model: String::new(),
            llm_confidence: String::new(),
            top_recommendations_html: String::new(),
            immediate_actions_html: String::new(),
            short_term_improvements_html: String::new(),
            cover_letter_points_html: String::new(),
            has_immediate_actions: false,
            has_short_term_improvements: false,
            has_cover_letter_points: false,
            version: String::new(),
            embedding_model: String::new(),
            llm_model_name: String::new(),
            resume_file: String::new(),
            job_file: String::new(),
        };
        
        Self {
            include_styles,
            _template: template,
        }
    }
    
    fn create_template_data(&self, report: &ComprehensiveReport) -> HtmlTemplate {
        let (score_class, score_label) = match report.analysis_summary.overall_score_percentage {
            90..=100 => ("score-excellent", "Excellent"),
            80..=89 => ("score-good", "Very Good"),
            70..=79 => ("score-good", "Good"),
            60..=69 => ("score-fair", "Fair"),
            _ => ("score-poor", "Poor"),
        };
        
        let top_recommendations: Vec<HtmlRecommendation> = report.recommendations
            .iter()
            .filter(|r| matches!(r.priority, RecommendationPriority::Critical | RecommendationPriority::High))
            .take(5)
            .map(|rec| {
                let (priority_icon, priority_class) = match rec.priority {
                    RecommendationPriority::Critical => ("🚨", "critical"),
                    RecommendationPriority::High => ("⚠️", "high"),
                    RecommendationPriority::Medium => ("📋", "medium"),
                    RecommendationPriority::Low => ("💡", "low"),
                };
                
                HtmlRecommendation {
                    priority_icon: priority_icon.to_string(),
                    priority_class: priority_class.to_string(),
                    title: rec.title.clone(),
                    section: ConsoleFormatter::format_section(&rec.section),
                    impact: rec.expected_impact.clone(),
                    description: rec.description.clone(),
                    example: rec.after_example.clone(),
                }
            })
            .collect();
        
        let immediate_actions: Vec<HtmlAction> = report.improvement_plan.immediate_actions
            .iter()
            .take(3)
            .map(|action| HtmlAction {
                title: action.title.clone(),
                time: action.estimated_time.clone(),
                description: action.description.clone(),
            })
            .collect();
        
        let short_term_improvements: Vec<HtmlAction> = report.improvement_plan.short_term_improvements
            .iter()
            .take(3)
            .map(|action| HtmlAction {
                title: action.title.clone(),
                time: action.estimated_time.clone(),
                description: action.description.clone(),
            })
            .collect();
        
        // Convert data to HTML strings
        let strengths_html = if report.analysis_summary.strengths.is_empty() {
            "".to_string()
        } else {
            format!("<ul>\n{}\n</ul>", 
                report.analysis_summary.strengths.iter()
                    .map(|s| format!("  <li>{}</li>", s))
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        };
        
        let improvement_areas_html = if report.analysis_summary.improvement_areas.is_empty() {
            "".to_string()
        } else {
            format!("<ul>\n{}\n</ul>", 
                report.analysis_summary.improvement_areas.iter()
                    .map(|s| format!("  <li>{}</li>", s))
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        };
        
        let top_recommendations_html = top_recommendations.iter()
            .map(|rec| {
                let example_html = rec.example.as_ref()
                    .map(|e| format!("<p><strong>💡 Example:</strong> <em>{}</em></p>", e))
                    .unwrap_or_default();
                format!(
                    "<div class=\"recommendation priority-{}\">\n    <h4>{} {}</h4>\n    <p><strong>Section:</strong> {} | <strong>Impact:</strong> {}</p>\n    <p>{}</p>\n    {}\n</div>",
                    rec.priority_class, rec.priority_icon, rec.title, rec.section, rec.impact, rec.description, example_html
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        
        let immediate_actions_html = if immediate_actions.is_empty() {
            "".to_string()
        } else {
            format!("<ul>\n{}\n</ul>", 
                immediate_actions.iter()
                    .map(|action| format!("  <li><strong>{}</strong> ({})\n      <br><small>{}</small></li>", action.title, action.time, action.description))
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        };
        
        let short_term_improvements_html = if short_term_improvements.is_empty() {
            "".to_string()
        } else {
            format!("<ul>\n{}\n</ul>", 
                short_term_improvements.iter()
                    .map(|action| format!("  <li><strong>{}</strong> ({})\n      <br><small>{}</small></li>", action.title, action.time, action.description))
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        };
        
        let cover_letter_points_html = if report.improvement_plan.cover_letter_points.is_empty() {
            "".to_string()
        } else {
            format!("<ul>\n{}\n</ul>", 
                report.improvement_plan.cover_letter_points.iter()
                    .map(|point| format!("  <li>{}</li>", point))
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        };
        
        HtmlTemplate {
            include_styles: self.include_styles,
            generated_at: chrono::DateTime::<chrono::Utc>::from(report.metadata.generated_at)
                .format("%Y-%m-%d %H:%M:%S UTC").to_string(),
            processing_time: report.metadata.processing_time_ms,
            overall_score: report.analysis_summary.overall_score_percentage,
            score_class: score_class.to_string(),
            score_label: score_label.to_string(),
            verdict: report.analysis_summary.verdict.clone(),
            embedding_score: report.analysis_summary.score_breakdown.embedding_score,
            embedding_weight: format!("{:.1}", report.analysis_summary.score_breakdown.component_weights.embedding_weight * 100.0),
            keyword_score: report.analysis_summary.score_breakdown.keyword_score,
            keyword_weight: format!("{:.1}", report.analysis_summary.score_breakdown.component_weights.keyword_weight * 100.0),
            llm_score: report.analysis_summary.score_breakdown.llm_score.unwrap_or(0),
            has_llm_score: report.analysis_summary.score_breakdown.llm_score.is_some(),
            llm_weight: format!("{:.1}", report.analysis_summary.score_breakdown.component_weights.llm_weight * 100.0),
            strengths_html,
            improvement_areas_html,
            llm_summary: if report.llm_insights.model_used != "None" {
                report.llm_insights.natural_language_summary.clone()
            } else {
                "".to_string()
            },
            has_llm_summary: !report.llm_insights.natural_language_summary.is_empty() && report.llm_insights.model_used != "None",
            llm_model: report.llm_insights.model_used.clone(),
            llm_confidence: format!("{:.1}", report.llm_insights.llm_confidence * 100.0),
            top_recommendations_html,
            immediate_actions_html,
            short_term_improvements_html,
            cover_letter_points_html,
            has_immediate_actions: !immediate_actions.is_empty(),
            has_short_term_improvements: !short_term_improvements.is_empty(),
            has_cover_letter_points: !report.improvement_plan.cover_letter_points.is_empty(),
            version: report.metadata.aligner_version.clone(),
            embedding_model: report.metadata.models_used.embedding_model.clone(),
            llm_model_name: report.metadata.models_used.llm_model.as_deref().unwrap_or("ATS-only").to_string(),
            resume_file: Path::new(&report.metadata.resume_file).file_name().unwrap().to_string_lossy().to_string(),
            job_file: Path::new(&report.metadata.job_file).file_name().unwrap().to_string_lossy().to_string(),
        }
    }
}

impl OutputFormatter for HtmlFormatter {
    fn format_report(&self, report: &ComprehensiveReport) -> Result<String> {
        let template_data = self.create_template_data(report);
        Ok(template_data.render().map_err(|e| crate::error::ResumeAlignerError::OutputFormatting(e.to_string()))?)
    }
    
    fn supports_format(&self) -> OutputFormat {
        OutputFormat::Html
    }
}

impl MarkdownFormatter {
    pub fn new(include_metadata: bool) -> Self {
        Self { include_metadata }
    }
}

impl OutputFormatter for MarkdownFormatter {
    fn format_report(&self, report: &ComprehensiveReport) -> Result<String> {
        let mut output = String::new();
        
        // Title
        output.push_str("# 📊 Resume Alignment Analysis Report\n\n");
        
        if self.include_metadata {
            output.push_str(&format!("**Generated:** {} | **Processing Time:** {}ms\n", 
                chrono::DateTime::<chrono::Utc>::from(report.metadata.generated_at)
                    .format("%Y-%m-%d %H:%M:%S UTC"),
                report.metadata.processing_time_ms
            ));
            output.push_str(&format!("**Resume:** `{}` | **Job:** `{}`\n\n", 
                Path::new(&report.metadata.resume_file).file_name().unwrap().to_string_lossy(),
                Path::new(&report.metadata.job_file).file_name().unwrap().to_string_lossy()
            ));
        }
        
        // Executive Summary
        output.push_str("## Executive Summary\n\n");
        output.push_str(&format!("**Overall Alignment Score:** {}% {}\n\n", 
            report.analysis_summary.overall_score_percentage,
            Self::markdown_score_badge(report.analysis_summary.overall_score_percentage)
        ));
        output.push_str(&format!("**Verdict:** {}\n\n", report.analysis_summary.verdict));
        
        // Score Breakdown
        output.push_str("### Score Breakdown\n\n");
        output.push_str("| Component | Score | Weight |\n");
        output.push_str("|-----------|-------|--------|\n");
        output.push_str(&format!("| 🎯 Semantic Alignment | {}% | {:.1}% |\n", 
            report.analysis_summary.score_breakdown.embedding_score,
            report.analysis_summary.score_breakdown.component_weights.embedding_weight * 100.0
        ));
        output.push_str(&format!("| 🔍 Keyword Matching | {}% | {:.1}% |\n", 
            report.analysis_summary.score_breakdown.keyword_score,
            report.analysis_summary.score_breakdown.component_weights.keyword_weight * 100.0
        ));
        if let Some(llm_score) = report.analysis_summary.score_breakdown.llm_score {
            output.push_str(&format!("| 🤖 AI Analysis | {}% | {:.1}% |\n", 
                llm_score,
                report.analysis_summary.score_breakdown.component_weights.llm_weight * 100.0
            ));
        }
        output.push_str("\n");
        
        // Strengths
        if !report.analysis_summary.strengths.is_empty() {
            output.push_str("### ✅ Key Strengths\n\n");
            for strength in &report.analysis_summary.strengths {
                output.push_str(&format!("- {}\n", strength));
            }
            output.push_str("\n");
        }
        
        // Improvement Areas
        if !report.analysis_summary.improvement_areas.is_empty() {
            output.push_str("### 🎯 Areas for Improvement\n\n");
            for area in &report.analysis_summary.improvement_areas {
                output.push_str(&format!("- {}\n", area));
            }
            output.push_str("\n");
        }
        
        // LLM Insights
        if !report.llm_insights.natural_language_summary.is_empty() && report.llm_insights.model_used != "None" {
            output.push_str("## 🤖 AI Analysis Insights\n\n");
            output.push_str(&format!("**Model:** {} | **Confidence:** {:.1}%\n\n", 
                report.llm_insights.model_used,
                report.llm_insights.llm_confidence * 100.0
            ));
            output.push_str(&format!("> {}\n\n", report.llm_insights.natural_language_summary));
        }
        
        // Recommendations
        output.push_str("## 📋 Recommendations\n\n");
        
        // Group recommendations by priority
        let critical_recs: Vec<_> = report.recommendations.iter()
            .filter(|r| matches!(r.priority, RecommendationPriority::Critical))
            .collect();
        let high_recs: Vec<_> = report.recommendations.iter()
            .filter(|r| matches!(r.priority, RecommendationPriority::High))
            .collect();
        let medium_recs: Vec<_> = report.recommendations.iter()
            .filter(|r| matches!(r.priority, RecommendationPriority::Medium))
            .collect();
        
        if !critical_recs.is_empty() {
            output.push_str("### 🚨 Critical Priority\n\n");
            for (i, rec) in critical_recs.iter().enumerate() {
                output.push_str(&Self::format_markdown_recommendation(i + 1, rec));
            }
        }
        
        if !high_recs.is_empty() {
            output.push_str("### ⚠️ High Priority\n\n");
            for (i, rec) in high_recs.iter().enumerate() {
                output.push_str(&Self::format_markdown_recommendation(i + 1, rec));
            }
        }
        
        if !medium_recs.is_empty() {
            output.push_str("### 📋 Medium Priority\n\n");
            for (i, rec) in medium_recs.iter().enumerate() {
                output.push_str(&Self::format_markdown_recommendation(i + 1, rec));
            }
        }
        
        // Improvement Plan
        output.push_str("## 🚀 Improvement Plan\n\n");
        
        if !report.improvement_plan.immediate_actions.is_empty() {
            output.push_str("### Quick Wins (< 1 hour)\n\n");
            for (i, action) in report.improvement_plan.immediate_actions.iter().enumerate() {
                output.push_str(&format!("{}. **{}** ({})\n   {}\n\n", 
                    i + 1, 
                    action.title, 
                    action.estimated_time,
                    action.description
                ));
            }
        }
        
        if !report.improvement_plan.short_term_improvements.is_empty() {
            output.push_str("### Short-term Improvements (1-3 days)\n\n");
            for (i, action) in report.improvement_plan.short_term_improvements.iter().enumerate() {
                output.push_str(&format!("{}. **{}** ({})\n   {}\n\n", 
                    i + 1, 
                    action.title, 
                    action.estimated_time,
                    action.description
                ));
            }
        }
        
        // Cover Letter Points
        if !report.improvement_plan.cover_letter_points.is_empty() {
            output.push_str("### 💌 Cover Letter Talking Points\n\n");
            for point in &report.improvement_plan.cover_letter_points {
                output.push_str(&format!("- {}\n", point));
            }
            output.push_str("\n");
        }
        
        // Gap Analysis Summary
        output.push_str("## 📊 Gap Analysis Summary\n\n");
        if !report.gap_analysis.critical_gaps.is_empty() {
            output.push_str(&format!("**Critical Gaps:** {}\n", report.gap_analysis.critical_gaps.len()));
        }
        if !report.gap_analysis.important_gaps.is_empty() {
            output.push_str(&format!("**Important Gaps:** {}\n", report.gap_analysis.important_gaps.len()));
        }
        if !report.gap_analysis.skills_analysis.missing_technical_skills.is_empty() {
            output.push_str(&format!("**Missing Technical Skills:** {}\n\n", 
                report.gap_analysis.skills_analysis.missing_technical_skills.join(", ")
            ));
        }
        
        // Footer
        if self.include_metadata {
            output.push_str("---\n\n");
            output.push_str(&format!("*Generated by Resume Aligner v{} using {} + {}*\n", 
                report.metadata.aligner_version,
                report.metadata.models_used.embedding_model,
                report.metadata.models_used.llm_model.as_deref().unwrap_or("ATS-only")
            ));
        }
        
        Ok(output)
    }
    
    fn supports_format(&self) -> OutputFormat {
        OutputFormat::Markdown
    }
}

impl MarkdownFormatter {
    fn markdown_score_badge(score: u8) -> &'static str {
        match score {
            90..=100 => "🟢 Excellent",
            80..=89 => "🟡 Very Good",
            70..=79 => "🟠 Good",
            60..=69 => "🔴 Fair",
            50..=59 => "🔴 Below Average",
            _ => "🔴 Poor",
        }
    }
    
    fn format_markdown_recommendation(index: usize, rec: &ActionableRecommendation) -> String {
        let mut output = format!("#### {}. {}\n\n", index, rec.title);
        output.push_str(&format!("**Section:** {} | **Effort:** {} | **Impact:** {}\n\n", 
            ConsoleFormatter::format_section(&rec.section),
            ConsoleFormatter::format_effort(&rec.effort_level),
            rec.expected_impact
        ));
        output.push_str(&format!("{}\n\n", rec.description));
        
        if let Some(example) = &rec.after_example {
            output.push_str(&format!("**Example:**\n```\n{}\n```\n\n", example));
        }
        
        if !rec.addresses_keywords.is_empty() {
            output.push_str(&format!("**Keywords addressed:** `{}`\n\n", rec.addresses_keywords.join("`, `")));
        }
        
        output
    }
}

// Basic PDF formatter - for now we'll generate a simple text-based PDF
// In the future, this could be enhanced with charts, graphs, and rich formatting
impl PdfFormatter {
    pub fn new(include_charts: bool) -> Self {
        Self { _include_charts: include_charts }
    }
}

impl OutputFormatter for PdfFormatter {
    fn format_report(&self, report: &ComprehensiveReport) -> Result<String> {
        // For now, we'll return formatted text that could be used to generate a PDF
        // In a production implementation, you'd use printpdf to create an actual PDF
        
        let mut content = String::new();
        
        content.push_str("RESUME ALIGNMENT ANALYSIS REPORT\n");
        content.push_str(&"=".repeat(50));
        content.push_str("\n\n");
        
        content.push_str(&format!("Generated: {}\n", 
            chrono::DateTime::<chrono::Utc>::from(report.metadata.generated_at)
                .format("%Y-%m-%d %H:%M:%S UTC")
        ));
        content.push_str(&format!("Processing Time: {}ms\n\n", report.metadata.processing_time_ms));
        
        // Executive Summary
        content.push_str("EXECUTIVE SUMMARY\n");
        content.push_str(&"-".repeat(20));
        content.push_str("\n\n");
        
        content.push_str(&format!("Overall Score: {}%\n", report.analysis_summary.overall_score_percentage));
        content.push_str(&format!("Verdict: {}\n\n", report.analysis_summary.verdict));
        
        // Score Breakdown
        content.push_str("Score Breakdown:\n");
        content.push_str(&format!("  • Semantic Alignment: {}% (weight: {:.1}%)\n", 
            report.analysis_summary.score_breakdown.embedding_score,
            report.analysis_summary.score_breakdown.component_weights.embedding_weight * 100.0
        ));
        content.push_str(&format!("  • Keyword Matching: {}% (weight: {:.1}%)\n", 
            report.analysis_summary.score_breakdown.keyword_score,
            report.analysis_summary.score_breakdown.component_weights.keyword_weight * 100.0
        ));
        if let Some(llm_score) = report.analysis_summary.score_breakdown.llm_score {
            content.push_str(&format!("  • AI Analysis: {}% (weight: {:.1}%)\n", 
                llm_score,
                report.analysis_summary.score_breakdown.component_weights.llm_weight * 100.0
            ));
        }
        content.push_str("\n");
        
        // Key Strengths
        if !report.analysis_summary.strengths.is_empty() {
            content.push_str("KEY STRENGTHS\n");
            content.push_str(&"-".repeat(20));
            content.push_str("\n");
            for strength in &report.analysis_summary.strengths {
                content.push_str(&format!("• {}\n", strength));
            }
            content.push_str("\n");
        }
        
        // Improvement Areas
        if !report.analysis_summary.improvement_areas.is_empty() {
            content.push_str("AREAS FOR IMPROVEMENT\n");
            content.push_str(&"-".repeat(20));
            content.push_str("\n");
            for area in &report.analysis_summary.improvement_areas {
                content.push_str(&format!("• {}\n", area));
            }
            content.push_str("\n");
        }
        
        // AI Insights
        if !report.llm_insights.natural_language_summary.is_empty() && report.llm_insights.model_used != "None" {
            content.push_str("AI ANALYSIS INSIGHTS\n");
            content.push_str(&"-".repeat(20));
            content.push_str("\n");
            content.push_str(&format!("Model: {}\n", report.llm_insights.model_used));
            content.push_str(&format!("Confidence: {:.1}%\n\n", report.llm_insights.llm_confidence * 100.0));
            content.push_str(&format!("{} \n\n", report.llm_insights.natural_language_summary));
        }
        
        // Top Recommendations
        content.push_str("TOP RECOMMENDATIONS\n");
        content.push_str(&"-".repeat(20));
        content.push_str("\n");
        
        let top_recommendations = report.recommendations
            .iter()
            .filter(|r| matches!(r.priority, RecommendationPriority::Critical | RecommendationPriority::High))
            .take(5);
        
        for (i, rec) in top_recommendations.enumerate() {
            content.push_str(&format!("{}. {} ({})", 
                i + 1, 
                rec.title,
                ConsoleFormatter::format_section(&rec.section)
            ));
            
            let priority_text = match rec.priority {
                RecommendationPriority::Critical => "[CRITICAL]",
                RecommendationPriority::High => "[HIGH]",
                RecommendationPriority::Medium => "[MEDIUM]",
                RecommendationPriority::Low => "[LOW]",
            };
            content.push_str(&format!(" {}\n", priority_text));
            content.push_str(&format!("   {}\n", rec.description));
            
            if let Some(example) = &rec.after_example {
                content.push_str(&format!("   Example: {}\n", example));
            }
            content.push_str("\n");
        }
        
        // Improvement Plan
        content.push_str("IMMEDIATE ACTION PLAN\n");
        content.push_str(&"-".repeat(20));
        content.push_str("\n");
        
        if !report.improvement_plan.immediate_actions.is_empty() {
            content.push_str("Quick Wins (< 1 hour):\n");
            for (i, action) in report.improvement_plan.immediate_actions.iter().take(3).enumerate() {
                content.push_str(&format!("  {}. {} ({})\n", 
                    i + 1, 
                    action.title, 
                    action.estimated_time
                ));
            }
            content.push_str("\n");
        }
        
        // Footer
        content.push_str(&"=".repeat(50));
        content.push_str("\n");
        content.push_str(&format!("Generated by Resume Aligner v{}\n", report.metadata.aligner_version));
        content.push_str(&format!("Models: {} + {}\n", 
            report.metadata.models_used.embedding_model,
            report.metadata.models_used.llm_model.as_deref().unwrap_or("ATS-only")
        ));
        
        // Note: In a real implementation, you would use the printpdf crate here
        // to generate an actual PDF from this content with proper formatting,
        // fonts, charts, and visual elements
        
        Ok(content)
    }
    
    fn supports_format(&self) -> OutputFormat {
        OutputFormat::Pdf
    }
}

impl ReportGenerator {
    pub fn new() -> Self {
        Self {
            console_formatter: ConsoleFormatter::new(true, false),
            json_formatter: JsonFormatter::new(true),
            markdown_formatter: MarkdownFormatter::new(true),
            html_formatter: HtmlFormatter::new(true),
            pdf_formatter: PdfFormatter::new(false),
        }
    }
    
    pub fn with_options(
        use_colors: bool, 
        detailed: bool, 
        pretty_json: bool, 
        include_metadata: bool,
        include_html_styles: bool,
        include_pdf_charts: bool
    ) -> Self {
        Self {
            console_formatter: ConsoleFormatter::new(use_colors, detailed),
            json_formatter: JsonFormatter::new(pretty_json),
            markdown_formatter: MarkdownFormatter::new(include_metadata),
            html_formatter: HtmlFormatter::new(include_html_styles),
            pdf_formatter: PdfFormatter::new(include_pdf_charts),
        }
    }
    
    pub fn generate_report(&self, report: &ComprehensiveReport, format: &OutputFormat) -> Result<String> {
        match format {
            OutputFormat::Console => self.console_formatter.format_report(report),
            OutputFormat::Json => self.json_formatter.format_report(report),
            OutputFormat::Markdown => self.markdown_formatter.format_report(report),
            OutputFormat::Html => self.html_formatter.format_report(report),
            OutputFormat::Pdf => self.pdf_formatter.format_report(report),
        }
    }
    
    pub fn generate_detailed_console(&self, report: &ComprehensiveReport) -> Result<String> {
        let detailed_formatter = ConsoleFormatter::new(true, true);
        detailed_formatter.format_report(report)
    }
}

impl Default for ReportGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// Utility functions for saving reports
pub fn save_report_to_file(content: &str, file_path: &Path) -> Result<()> {
    use std::fs;
    if let Some(parent) = file_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(file_path, content)?;
    Ok(())
}

pub fn suggest_filename(format: &OutputFormat, resume_name: &str, timestamp: bool) -> String {
    let base_name = Path::new(resume_name)
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy();
    
    let timestamp_suffix = if timestamp {
        format!("_{}", chrono::Utc::now().format("%Y%m%d_%H%M%S"))
    } else {
        String::new()
    };
    
    match format {
        OutputFormat::Console => format!("{}_analysis{}.txt", base_name, timestamp_suffix),
        OutputFormat::Json => format!("{}_analysis{}.json", base_name, timestamp_suffix),
        OutputFormat::Markdown => format!("{}_analysis{}.md", base_name, timestamp_suffix),
        OutputFormat::Html => format!("{}_analysis{}.html", base_name, timestamp_suffix),
        OutputFormat::Pdf => format!("{}_analysis{}.pdf", base_name, timestamp_suffix),
    }
}


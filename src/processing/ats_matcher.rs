//! ATS (Applicant Tracking System) keyword matching and scoring

use crate::error::{Result, ResumeAlignerError};
use crate::processing::document::{ProcessedDocument, SectionType};
use aho_corasick::AhoCorasick;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use strsim::{jaro_winkler, levenshtein};

/// ATS matcher for exact and fuzzy keyword matching
pub struct ATSMatcher {
    exact_matcher: AhoCorasick,
    skill_database: Vec<String>,
    fuzzy_threshold: f32,
    tech_skills: HashSet<String>,
    soft_skills: HashSet<String>,
    role_keywords: HashSet<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeywordMatch {
    pub keyword: String,
    pub positions: Vec<usize>,
    pub count: usize,
    pub section_type: Option<SectionType>,
    pub match_type: MatchType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyMatch {
    pub original_keyword: String,
    pub matched_text: String,
    pub similarity_score: f32,
    pub position: usize,
    pub section_type: Option<SectionType>,
    pub algorithm: FuzzyAlgorithm,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MatchType {
    Exact,
    CaseInsensitive,
    Partial,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FuzzyAlgorithm {
    JaroWinkler,
    Levenshtein,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ATSScore {
    pub overall_score: f32,
    pub exact_matches: Vec<KeywordMatch>,
    pub fuzzy_matches: Vec<FuzzyMatch>,
    pub section_scores: HashMap<String, SectionScore>,
    pub keyword_coverage: f32,
    pub missing_keywords: Vec<String>,
    pub skill_category_scores: SkillCategoryScores,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectionScore {
    pub section_type: SectionType,
    pub score: f32,
    pub matched_keywords: Vec<String>,
    pub missing_keywords: Vec<String>,
    pub keyword_density: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillCategoryScores {
    pub technical_skills: f32,
    pub soft_skills: f32,
    pub role_specific: f32,
    pub overall: f32,
}

impl ATSMatcher {
    /// Create a new ATS matcher with default skill databases
    pub fn new() -> Result<Self> {
        Self::with_custom_skills(Vec::new())
    }
    
    /// Create ATS matcher with custom additional skills
    pub fn with_custom_skills(additional_skills: Vec<String>) -> Result<Self> {
        let mut skill_database = Self::default_skill_database();
        skill_database.extend(additional_skills);
        
        // Sort patterns by length (longest first) to prioritize longer matches
        skill_database.sort_by(|a, b| b.len().cmp(&a.len()));
        
        // Build exact matcher for case-insensitive matching
        let patterns: Vec<&str> = skill_database.iter().map(|s| s.as_str()).collect();
        let exact_matcher = AhoCorasick::builder()
            .ascii_case_insensitive(true)
            .match_kind(aho_corasick::MatchKind::LeftmostLongest) // Prefer longer matches
            .build(&patterns)
            .map_err(|e| ResumeAlignerError::Processing(format!("Failed to build ATS matcher: {}", e)))?;
        
        let tech_skills = Self::default_tech_skills();
        let soft_skills = Self::default_soft_skills();
        let role_keywords = Self::default_role_keywords();
        
        Ok(Self {
            exact_matcher,
            skill_database,
            fuzzy_threshold: 0.8, // 80% similarity threshold
            tech_skills,
            soft_skills,
            role_keywords,
        })
    }
    
    /// Find exact keyword matches in text
    pub fn find_exact_matches(&self, text: &str, section_type: Option<SectionType>) -> Vec<KeywordMatch> {
        let mut matches = HashMap::new();
        
        // Find all matches using Aho-Corasick
        for mat in self.exact_matcher.find_iter(text) {
            let pattern_id = mat.pattern();
            let keyword = &self.skill_database[pattern_id];
            let position = mat.start();
            
            let entry = matches.entry(keyword.clone()).or_insert_with(|| KeywordMatch {
                keyword: keyword.clone(),
                positions: Vec::new(),
                count: 0,
                section_type: section_type.clone(),
                match_type: MatchType::CaseInsensitive,
            });
            
            entry.positions.push(position);
            entry.count += 1;
        }
        
        matches.into_values().collect()
    }
    
    /// Find fuzzy keyword matches using string similarity
    pub fn find_fuzzy_matches(&self, text: &str, section_type: Option<SectionType>) -> Vec<FuzzyMatch> {
        let mut fuzzy_matches = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();
        
        for (word_idx, word) in words.iter().enumerate() {
            let clean_word = self.clean_word(word);
            if clean_word.len() < 3 {
                continue; // Skip very short words
            }
            
            for skill in &self.skill_database {
                // Skip if we already have an exact match
                if clean_word.to_lowercase() == skill.to_lowercase() {
                    continue;
                }
                
                // Try Jaro-Winkler similarity
                let jw_similarity = jaro_winkler(&clean_word.to_lowercase(), &skill.to_lowercase()) as f32;
                if jw_similarity >= self.fuzzy_threshold {
                    let position = text.split_whitespace()
                        .take(word_idx)
                        .map(|w| w.len() + 1)
                        .sum::<usize>();
                    
                    fuzzy_matches.push(FuzzyMatch {
                        original_keyword: skill.clone(),
                        matched_text: clean_word.clone(),
                        similarity_score: jw_similarity,
                        position,
                        section_type: section_type.clone(),
                        algorithm: FuzzyAlgorithm::JaroWinkler,
                    });
                    continue;
                }
                
                // Try Levenshtein distance for shorter words
                if clean_word.len() <= 8 && skill.len() <= 8 {
                    let distance = levenshtein(&clean_word.to_lowercase(), &skill.to_lowercase());
                    let max_len = clean_word.len().max(skill.len());
                    let similarity = 1.0 - (distance as f32 / max_len as f32);
                    
                    if similarity >= self.fuzzy_threshold {
                        let position = text.split_whitespace()
                            .take(word_idx)
                            .map(|w| w.len() + 1)
                            .sum::<usize>();
                        
                        fuzzy_matches.push(FuzzyMatch {
                            original_keyword: skill.clone(),
                            matched_text: clean_word.clone(),
                            similarity_score: similarity,
                            position,
                            section_type: section_type.clone(),
                            algorithm: FuzzyAlgorithm::Levenshtein,
                        });
                    }
                }
            }
        }
        
        // Remove duplicates and keep highest similarity
        fuzzy_matches.sort_by(|a, b| b.similarity_score.partial_cmp(&a.similarity_score).unwrap());
        fuzzy_matches.dedup_by(|a, b| a.original_keyword == b.original_keyword && a.matched_text == b.matched_text);
        
        fuzzy_matches
    }
    
    /// Calculate comprehensive ATS score for resume against job description
    pub fn calculate_ats_score(&self, resume: &ProcessedDocument, job: &ProcessedDocument) -> Result<ATSScore> {
        // Extract keywords from job description
        let job_keywords = self.extract_job_keywords(job)?;
        
        // Find matches in resume
        let resume_exact_matches = self.find_document_exact_matches(resume);
        let resume_fuzzy_matches = self.find_document_fuzzy_matches(resume);
        
        // Calculate section-wise scores
        let section_scores = self.calculate_section_scores(resume, &job_keywords)?;
        
        // Calculate keyword coverage
        let matched_keywords: HashSet<String> = resume_exact_matches.iter()
            .map(|m| m.keyword.to_lowercase())
            .chain(resume_fuzzy_matches.iter().map(|m| m.original_keyword.to_lowercase()))
            .collect();
        
        let job_keywords_lower: HashSet<String> = job_keywords.iter()
            .map(|k| k.to_lowercase())
            .collect();
        
        let keyword_coverage = matched_keywords.intersection(&job_keywords_lower).count() as f32 / job_keywords_lower.len() as f32;
        
        // Find missing keywords
        let missing_keywords: Vec<String> = job_keywords_lower.difference(&matched_keywords)
            .cloned()
            .collect();
        
        // Calculate skill category scores
        let skill_category_scores = self.calculate_skill_category_scores(&resume_exact_matches, &resume_fuzzy_matches);
        
        // Calculate overall score (weighted average)
        let overall_score = self.calculate_overall_ats_score(
            keyword_coverage,
            &section_scores,
            &skill_category_scores,
        );
        
        Ok(ATSScore {
            overall_score,
            exact_matches: resume_exact_matches,
            fuzzy_matches: resume_fuzzy_matches,
            section_scores,
            keyword_coverage,
            missing_keywords,
            skill_category_scores,
        })
    }
    
    /// Extract important keywords from job description
    fn extract_job_keywords(&self, job: &ProcessedDocument) -> Result<Vec<String>> {
        let mut keywords = HashSet::new();
        
        // Extract from skills section if available
        for section in &job.sections {
            if matches!(section.section_type, SectionType::Skills) {
                keywords.extend(section.keywords.iter().cloned());
            }
        }
        
        // Extract from requirements using pattern matching
        let requirements_patterns = vec![
            Regex::new(r"(?i)(?:required|must have|should have|experience with|knowledge of|proficient in):?\s*([^.!?\n]+)").unwrap(),
            Regex::new(r"(?i)(?:minimum|preferred)\s+(?:\d+\+?)\s+years?\s+(?:of\s+)?(?:experience\s+)?(?:with|in)\s+([^.!?\n]+)").unwrap(),
        ];
        
        for pattern in &requirements_patterns {
            for cap in pattern.captures_iter(&job.original.content) {
                if let Some(requirements_text) = cap.get(1) {
                    let extracted = self.extract_skills_from_text(requirements_text.as_str());
                    keywords.extend(extracted);
                }
            }
        }
        
        // Also add keywords from document-level keywords
        keywords.extend(job.keywords.iter().cloned());
        
        Ok(keywords.into_iter().collect())
    }
    
    /// Find exact matches in entire document
    fn find_document_exact_matches(&self, doc: &ProcessedDocument) -> Vec<KeywordMatch> {
        let mut all_matches = Vec::new();
        
        // Match in sections
        for section in &doc.sections {
            let mut section_matches = self.find_exact_matches(&section.content, Some(section.section_type.clone()));
            all_matches.append(&mut section_matches);
        }
        
        // Match in full document (for keywords not in specific sections)
        let mut doc_matches = self.find_exact_matches(&doc.original.content, None);
        all_matches.append(&mut doc_matches);
        
        // Deduplicate
        all_matches.sort_by(|a, b| a.keyword.cmp(&b.keyword));
        all_matches.dedup_by(|a, b| a.keyword == b.keyword);
        
        all_matches
    }
    
    /// Find fuzzy matches in entire document
    fn find_document_fuzzy_matches(&self, doc: &ProcessedDocument) -> Vec<FuzzyMatch> {
        let mut all_matches = Vec::new();
        
        // Match in sections
        for section in &doc.sections {
            let mut section_matches = self.find_fuzzy_matches(&section.content, Some(section.section_type.clone()));
            all_matches.append(&mut section_matches);
        }
        
        // Remove duplicates
        all_matches.sort_by(|a, b| {
            a.original_keyword.cmp(&b.original_keyword)
                .then_with(|| b.similarity_score.partial_cmp(&a.similarity_score).unwrap())
        });
        all_matches.dedup_by(|a, b| a.original_keyword == b.original_keyword && a.matched_text == b.matched_text);
        
        all_matches
    }
    
    /// Calculate scores for each section
    fn calculate_section_scores(&self, resume: &ProcessedDocument, job_keywords: &[String]) -> Result<HashMap<String, SectionScore>> {
        let mut section_scores = HashMap::new();
        
        for section in &resume.sections {
            let section_matches = self.find_exact_matches(&section.content, Some(section.section_type.clone()));
            let matched_keywords: Vec<String> = section_matches.iter().map(|m| m.keyword.clone()).collect();
            
            let job_keywords_in_section: Vec<String> = job_keywords.iter()
                .filter(|k| matched_keywords.iter().any(|m| m.to_lowercase() == k.to_lowercase()))
                .cloned()
                .collect();
                
            let missing_keywords: Vec<String> = job_keywords.iter()
                .filter(|k| !matched_keywords.iter().any(|m| m.to_lowercase() == k.to_lowercase()))
                .cloned()
                .collect();
            
            let score = if job_keywords.is_empty() {
                0.0
            } else {
                job_keywords_in_section.len() as f32 / job_keywords.len() as f32
            };
            
            let keyword_density = if section.content.split_whitespace().count() == 0 {
                0.0
            } else {
                matched_keywords.len() as f32 / section.content.split_whitespace().count() as f32
            };
            
            let section_name = format!("{}", section.section_type);
            section_scores.insert(section_name, SectionScore {
                section_type: section.section_type.clone(),
                score,
                matched_keywords,
                missing_keywords,
                keyword_density,
            });
        }
        
        Ok(section_scores)
    }
    
    /// Calculate skill category scores
    fn calculate_skill_category_scores(&self, exact_matches: &[KeywordMatch], fuzzy_matches: &[FuzzyMatch]) -> SkillCategoryScores {
        let all_matched_skills: HashSet<String> = exact_matches.iter()
            .map(|m| m.keyword.to_lowercase())
            .chain(fuzzy_matches.iter().map(|m| m.original_keyword.to_lowercase()))
            .collect();
        
        let tech_matched = all_matched_skills.intersection(&self.tech_skills.iter().map(|s| s.to_lowercase()).collect()).count();
        let soft_matched = all_matched_skills.intersection(&self.soft_skills.iter().map(|s| s.to_lowercase()).collect()).count();
        let role_matched = all_matched_skills.intersection(&self.role_keywords.iter().map(|s| s.to_lowercase()).collect()).count();
        
        let technical_skills = tech_matched as f32 / self.tech_skills.len() as f32;
        let soft_skills = soft_matched as f32 / self.soft_skills.len() as f32;
        let role_specific = role_matched as f32 / self.role_keywords.len() as f32;
        
        let overall = (technical_skills * 0.5) + (soft_skills * 0.2) + (role_specific * 0.3);
        
        SkillCategoryScores {
            technical_skills,
            soft_skills,
            role_specific,
            overall,
        }
    }
    
    /// Calculate overall ATS score
    fn calculate_overall_ats_score(
        &self,
        keyword_coverage: f32,
        section_scores: &HashMap<String, SectionScore>,
        skill_scores: &SkillCategoryScores,
    ) -> f32 {
        // Weighted scoring:
        // - Keyword coverage: 40%
        // - Skill categories: 35%
        // - Section quality: 25%
        
        let section_avg = if section_scores.is_empty() {
            0.0
        } else {
            section_scores.values().map(|s| s.score).sum::<f32>() / section_scores.len() as f32
        };
        
        (keyword_coverage * 0.4) + (skill_scores.overall * 0.35) + (section_avg * 0.25)
    }
    
    /// Clean word for better matching
    fn clean_word(&self, word: &str) -> String {
        word.chars()
            .filter(|c| c.is_alphanumeric() || *c == '+' || *c == '#')
            .collect::<String>()
            .trim()
            .to_string()
    }
    
    /// Extract skills from arbitrary text
    fn extract_skills_from_text(&self, text: &str) -> Vec<String> {
        let mut skills = Vec::new();
        
        // Split by common delimiters
        let delimiters = [',', ';', '/', '|', '\n', '•', '-'];
        let mut parts = vec![text];
        
        for delimiter in &delimiters {
            parts = parts.into_iter()
                .flat_map(|part| part.split(*delimiter))
                .collect();
        }
        
        for part in parts {
            let cleaned = part.trim().to_string();
            if cleaned.len() > 2 && cleaned.len() < 50 {
                // Check if it matches any of our known skills
                for skill in &self.skill_database {
                    if cleaned.to_lowercase().contains(&skill.to_lowercase()) ||
                       skill.to_lowercase().contains(&cleaned.to_lowercase()) {
                        skills.push(cleaned.clone());
                        break;
                    }
                }
            }
        }
        
        skills
    }
    
    /// Default technical skills database
    fn default_tech_skills() -> HashSet<String> {
        vec![
            // Programming Languages
            "rust", "python", "javascript", "typescript", "java", "c++", "c#", "go", "ruby",
            "php", "swift", "kotlin", "scala", "haskell", "clojure", "r", "matlab",
            
            // Web Technologies
            "react", "vue", "angular", "svelte", "html", "css", "sass", "less", "tailwind",
            "bootstrap", "jquery", "webpack", "vite", "babel", "node.js", "express",
            "nextjs", "nuxt", "gatsby", "remix",
            
            // Backend/Infrastructure
            "docker", "kubernetes", "aws", "azure", "gcp", "terraform", "ansible",
            "jenkins", "gitlab", "github", "cicd", "devops", "microservices", "api",
            "rest", "graphql", "grpc", "redis", "elasticsearch", "nginx",
            
            // Databases
            "postgresql", "mysql", "mongodb", "cassandra", "dynamodb", "sqlite",
            "oracle", "sql server", "neo4j", "influxdb",
            
            // Data Science/ML
            "machine learning", "deep learning", "tensorflow", "pytorch", "sklearn",
            "pandas", "numpy", "jupyter", "spark", "hadoop", "kafka", "airflow",
            
            // Testing
            "jest", "pytest", "junit", "selenium", "cypress", "testing", "tdd", "bdd",
        ].iter().map(|s| s.to_string()).collect()
    }
    
    /// Default soft skills database
    fn default_soft_skills() -> HashSet<String> {
        vec![
            "leadership", "communication", "teamwork", "problem solving", "critical thinking",
            "creativity", "adaptability", "time management", "project management",
            "collaboration", "mentoring", "coaching", "presentation", "negotiation",
            "customer service", "analytical", "detail oriented", "organized",
        ].iter().map(|s| s.to_string()).collect()
    }
    
    /// Default role-specific keywords
    fn default_role_keywords() -> HashSet<String> {
        vec![
            "software engineer", "developer", "architect", "senior", "lead", "principal",
            "manager", "director", "cto", "full stack", "frontend", "backend",
            "devops", "sre", "data scientist", "ml engineer", "product manager",
            "designer", "analyst", "consultant", "specialist",
        ].iter().map(|s| s.to_string()).collect()
    }
    
    /// Default comprehensive skill database
    fn default_skill_database() -> Vec<String> {
        let mut skills = Vec::new();
        
        skills.extend(Self::default_tech_skills());
        skills.extend(Self::default_soft_skills());
        skills.extend(Self::default_role_keywords());
        
        // Additional specific skills
        skills.extend(vec![
            "agile", "scrum", "kanban", "jira", "confluence", "slack", "git",
            "linux", "unix", "windows", "macos", "bash", "powershell",
            "vim", "emacs", "vscode", "intellij", "eclipse",
        ].iter().map(|s| s.to_string()));
        
        skills.sort();
        skills.dedup();
        skills
    }
    
    /// Set fuzzy matching threshold (0.0 to 1.0)
    pub fn set_fuzzy_threshold(&mut self, threshold: f32) {
        self.fuzzy_threshold = threshold.clamp(0.0, 1.0);
    }
    
    /// Get current fuzzy threshold
    pub fn fuzzy_threshold(&self) -> f32 {
        self.fuzzy_threshold
    }
    
    /// Get skill database size
    pub fn skill_count(&self) -> usize {
        self.skill_database.len()
    }
}

impl Default for ATSMatcher {
    fn default() -> Self {
        Self::new().expect("Failed to create default ATS matcher")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processing::document::{Document, DocumentType};

    #[test]
    fn test_ats_matcher_creation() {
        let matcher = ATSMatcher::new().unwrap();
        assert!(matcher.skill_count() > 0);
        assert_eq!(matcher.fuzzy_threshold(), 0.8);
    }

    #[test]
    fn test_exact_matching() {
        let matcher = ATSMatcher::new().unwrap();
        let text = "I have experience with Python, JavaScript, and React development.";
        let matches = matcher.find_exact_matches(text, None);
        
        assert!(matches.len() > 0);
        assert!(matches.iter().any(|m| m.keyword.to_lowercase() == "python"));
        assert!(matches.iter().any(|m| m.keyword.to_lowercase() == "javascript"));
        assert!(matches.iter().any(|m| m.keyword.to_lowercase() == "react"));
    }

    #[test]
    fn test_fuzzy_matching() {
        let matcher = ATSMatcher::new().unwrap();
        let text = "I know Pythong and Reactjs very well.";
        let matches = matcher.find_fuzzy_matches(text, None);
        
        // Should find fuzzy matches for misspelled skills
        assert!(matches.len() > 0);
    }

    #[test]
    fn test_ats_scoring() {
        let matcher = ATSMatcher::new().unwrap();
        
        let resume_content = "John Doe\n\nSkills:\nPython, JavaScript, React, Node.js\n\nExperience:\nSoftware Engineer with 5 years experience";
        let mut resume_doc = Document::new(resume_content.to_string(), "resume.txt".to_string(), DocumentType::Resume);
        let resume = resume_doc.process(512, 50).unwrap();
        
        let job_content = "We need a developer with Python and React experience. JavaScript knowledge required.";
        let mut job_doc = Document::new(job_content.to_string(), "job.txt".to_string(), DocumentType::JobDescription);
        let job = job_doc.process(512, 50).unwrap();
        
        let score = matcher.calculate_ats_score(&resume, &job).unwrap();
        
        assert!(score.overall_score > 0.0);
        assert!(score.keyword_coverage > 0.0);
        assert!(score.exact_matches.len() > 0);
    }
}

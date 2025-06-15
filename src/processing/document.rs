//! Document structures and processing

use crate::error::{Result, ResumeAlignerError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Document {
    pub content: String,
    pub file_path: String,
    pub document_type: DocumentType,
    pub metadata: DocumentMetadata,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DocumentType {
    Resume,
    JobDescription,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocumentMetadata {
    pub title: Option<String>,
    pub sections: HashMap<String, SectionInfo>,
    pub word_count: usize,
    pub character_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SectionInfo {
    pub start_index: usize,
    pub end_index: usize,
    pub section_type: SectionType,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SectionType {
    Skills,
    Experience,
    Education,
    Summary,
    Projects,
    Certifications,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub content: String,
    pub start_index: usize,
    pub end_index: usize,
    pub section_type: Option<SectionType>,
    pub chunk_id: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProcessedDocument {
    pub original: Document,
    pub chunks: Vec<DocumentChunk>,
    pub sections: Vec<DocumentSection>,
    pub keywords: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocumentSection {
    pub section_type: SectionType,
    pub content: String,
    pub start_index: usize,
    pub end_index: usize,
    pub keywords: Vec<String>,
}

impl Document {
    pub fn new(
        content: String,
        file_path: String,
        document_type: DocumentType,
    ) -> Self {
        let word_count = content.split_whitespace().count();
        let character_count = content.chars().count();
        
        Self {
            content,
            file_path,
            document_type,
            metadata: DocumentMetadata {
                title: None,
                sections: HashMap::new(),
                word_count,
                character_count,
            },
        }
    }

    /// Extract the document title from the first few lines
    pub fn extract_title(&mut self) {
        let lines: Vec<&str> = self.content.lines().take(5).collect();
        
        for line in lines {
            let trimmed = line.trim();
            if !trimmed.is_empty() && trimmed.len() > 5 && trimmed.len() < 100 {
                // Likely a title if it's not too short or too long
                if !trimmed.contains('@') && !trimmed.starts_with('-') {
                    self.metadata.title = Some(trimmed.to_string());
                    break;
                }
            }
        }
    }

    /// Detect sections in the document
    pub fn detect_sections(&mut self) -> Result<()> {
        let lines: Vec<&str> = self.content.lines().collect();
        
        let section_patterns = vec![
            ("skills", vec!["skills", "technical skills", "core competencies", "expertise"]),
            ("experience", vec!["experience", "work experience", "professional experience", "employment", "career"]),
            ("education", vec!["education", "academic background", "qualifications", "degree"]),
            ("summary", vec!["summary", "profile", "objective", "about", "overview"]),
            ("projects", vec!["projects", "portfolio", "notable projects"]),
            ("certifications", vec!["certifications", "certificates", "licenses"]),
        ];

        for (section_name, patterns) in section_patterns {
            for (line_idx, line) in lines.iter().enumerate() {
                let line_lower = line.to_lowercase();
                let line_lower_trimmed = line_lower.trim();
                
                for pattern in &patterns {
                    if line_lower_trimmed.contains(pattern) && 
                       (line_lower_trimmed.starts_with(pattern) || 
                        line_lower_trimmed == *pattern ||
                        line.trim().ends_with(':')) {
                        
                        let start_index = self.content
                            .lines()
                            .take(line_idx)
                            .map(|l| l.len() + 1) // +1 for newline
                            .sum::<usize>();
                        
                        // Find end of section (next section or end of document)
                        let end_index = self.find_section_end(line_idx, &lines);
                        
                        let section_type = match section_name {
                            "skills" => SectionType::Skills,
                            "experience" => SectionType::Experience,
                            "education" => SectionType::Education,
                            "summary" => SectionType::Summary,
                            "projects" => SectionType::Projects,
                            "certifications" => SectionType::Certifications,
                            _ => SectionType::Other(section_name.to_string()),
                        };
                        
                        self.metadata.sections.insert(
                            section_name.to_string(),
                            SectionInfo {
                                start_index,
                                end_index,
                                section_type,
                            },
                        );
                        break;
                    }
                }
            }
        }
        
        Ok(())
    }

    fn find_section_end(&self, start_line: usize, lines: &[&str]) -> usize {
        // Look for the next section header or end of document
        for (idx, line) in lines.iter().enumerate().skip(start_line + 1) {
            let line_lower = line.to_lowercase();
            let line_lower_trimmed = line_lower.trim();
            
            // Check if this looks like a section header
            if (line_lower_trimmed.contains("experience") ||
                line_lower_trimmed.contains("education") ||
                line_lower_trimmed.contains("skills") ||
                line_lower_trimmed.contains("summary") ||
                line_lower_trimmed.contains("projects") ||
                line_lower_trimmed.contains("certifications")) &&
               (line.trim().ends_with(':') || line_lower_trimmed == line_lower_trimmed) {
                
                return self.content
                    .lines()
                    .take(idx)
                    .map(|l| l.len() + 1)
                    .sum::<usize>();
            }
        }
        
        // If no next section found, return end of document
        self.content.len()
    }

    /// Create chunks from the document
    pub fn create_chunks(&self, chunk_size: usize, overlap: usize) -> Result<Vec<DocumentChunk>> {
        if chunk_size <= overlap {
            return Err(ResumeAlignerError::Processing(
                "Chunk size must be greater than overlap".to_string(),
            ));
        }

        let mut chunks = Vec::new();
        let content_chars: Vec<char> = self.content.chars().collect();
        let total_length = content_chars.len();
        
        if total_length == 0 {
            return Ok(chunks);
        }

        let step_size = chunk_size - overlap;
        let mut start = 0;
        let mut chunk_id = 0;

        while start < total_length {
            let end = std::cmp::min(start + chunk_size, total_length);
            
            // Try to break at word boundaries
            let mut actual_end = end;
            if end < total_length {
                // Look backwards for a space or punctuation
                for i in (start..end).rev() {
                    if content_chars[i].is_whitespace() || 
                       content_chars[i] == '.' || 
                       content_chars[i] == '!' || 
                       content_chars[i] == '?' {
                        actual_end = i + 1;
                        break;
                    }
                }
            }

            let chunk_content: String = content_chars[start..actual_end].iter().collect();
            
            // Determine section type for this chunk
            let section_type = self.get_section_type_for_position(start);
            
            chunks.push(DocumentChunk {
                content: chunk_content.trim().to_string(),
                start_index: start,
                end_index: actual_end,
                section_type,
                chunk_id,
            });

            chunk_id += 1;
            start += step_size;
            
            // Prevent infinite loop
            if step_size == 0 {
                break;
            }
        }

        Ok(chunks)
    }

    fn get_section_type_for_position(&self, position: usize) -> Option<SectionType> {
        for section_info in self.metadata.sections.values() {
            if position >= section_info.start_index && position < section_info.end_index {
                return Some(section_info.section_type.clone());
            }
        }
        None
    }

    /// Convert to processed document with chunks and sections
    pub fn process(&mut self, chunk_size: usize, overlap: usize) -> Result<ProcessedDocument> {
        // Extract metadata
        self.extract_title();
        self.detect_sections()?;
        
        // Create chunks
        let chunks = self.create_chunks(chunk_size, overlap)?;
        
        // Extract sections
        let sections = self.extract_sections();
        
        // Extract keywords (basic implementation)
        let keywords = self.extract_keywords();
        
        Ok(ProcessedDocument {
            original: self.clone(),
            chunks,
            sections,
            keywords,
        })
    }

    fn extract_sections(&self) -> Vec<DocumentSection> {
        let mut sections = Vec::new();
        
        for (_name, section_info) in &self.metadata.sections {
            let content = if section_info.end_index <= self.content.len() {
                self.content[section_info.start_index..section_info.end_index].to_string()
            } else {
                self.content[section_info.start_index..].to_string()
            };
            
            let keywords = self.extract_section_keywords(&content);
            
            sections.push(DocumentSection {
                section_type: section_info.section_type.clone(),
                content: content.trim().to_string(),
                start_index: section_info.start_index,
                end_index: section_info.end_index,
                keywords,
            });
        }
        
        sections
    }

    fn extract_keywords(&self) -> Vec<String> {
        // Basic keyword extraction - can be enhanced later
        let mut keywords = Vec::new();
        let words: Vec<&str> = self.content
            .split_whitespace()
            .filter(|w| w.len() > 3) // Filter short words
            .collect();
        
        // Simple frequency-based keyword extraction
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        for word in words {
            let clean_word = word.to_lowercase()
                .chars()
                .filter(|c| c.is_alphabetic())
                .collect::<String>();
            
            if clean_word.len() > 3 {
                *word_counts.entry(clean_word).or_insert(0) += 1;
            }
        }
        
        // Get top keywords
        let mut sorted_words: Vec<(String, usize)> = word_counts.into_iter().collect();
        sorted_words.sort_by(|a, b| b.1.cmp(&a.1));
        
        keywords.extend(
            sorted_words
                .into_iter()
                .take(20) // Top 20 keywords
                .map(|(word, _)| word)
        );
        
        keywords
    }

    fn extract_section_keywords(&self, content: &str) -> Vec<String> {
        let mut keywords = Vec::new();
        let words: Vec<&str> = content
            .split_whitespace()
            .filter(|w| w.len() > 2)
            .collect();
        
        for word in words {
            let clean_word = word.to_lowercase()
                .chars()
                .filter(|c| c.is_alphabetic())
                .collect::<String>();
            
            if clean_word.len() > 2 && !keywords.contains(&clean_word) {
                keywords.push(clean_word);
            }
        }
        
        keywords.truncate(10); // Limit to 10 keywords per section
        keywords
    }
}

impl std::fmt::Display for SectionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SectionType::Skills => write!(f, "Skills"),
            SectionType::Experience => write!(f, "Experience"),
            SectionType::Education => write!(f, "Education"),
            SectionType::Summary => write!(f, "Summary"),
            SectionType::Projects => write!(f, "Projects"),
            SectionType::Certifications => write!(f, "Certifications"),
            SectionType::Other(name) => write!(f, "{}", name),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_creation() {
        let content = "John Doe\nSoftware Engineer\n\nSkills:\nRust, Python, JavaScript".to_string();
        let doc = Document::new(content.clone(), "test.txt".to_string(), DocumentType::Resume);
        
        assert_eq!(doc.content, content);
        assert_eq!(doc.document_type, DocumentType::Resume);
        assert!(doc.metadata.word_count > 0);
    }

    #[test]
    fn test_chunking() {
        let content = "This is a test document with enough content to create multiple chunks when we set a small chunk size.".to_string();
        let doc = Document::new(content, "test.txt".to_string(), DocumentType::Resume);
        
        let chunks = doc.create_chunks(50, 10).unwrap();
        assert!(chunks.len() > 1);
        assert!(chunks[0].content.len() <= 50);
    }

    #[test]
    fn test_section_detection() {
        let content = "John Doe\n\nSummary:\nExperienced developer\n\nExperience:\nSoftware Engineer at Company\n\nSkills:\nRust, Python".to_string();
        let mut doc = Document::new(content, "test.txt".to_string(), DocumentType::Resume);
        
        doc.detect_sections().unwrap();
        assert!(doc.metadata.sections.contains_key("summary"));
        assert!(doc.metadata.sections.contains_key("experience"));
        assert!(doc.metadata.sections.contains_key("skills"));
    }
}

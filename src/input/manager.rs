//! Input manager for handling different file types

use crate::error::{Result, ResumeAlignerError};
use crate::input::file_detector::FileType;
use crate::input::text_extractor::{TextExtractor, PdfExtractor, PlainTextExtractor, MarkdownExtractor};
use std::path::Path;
use std::collections::HashMap;
use log::info;

pub struct InputManager {
    cache: HashMap<String, String>,
    enable_cache: bool,
}

impl InputManager {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            enable_cache: true,
        }
    }
    
    pub fn with_cache(mut self, enable: bool) -> Self {
        self.enable_cache = enable;
        self
    }
    
    pub async fn extract_text(&mut self, path: &Path) -> Result<String> {
        let path_str = path.to_string_lossy().to_string();
        
        // Check cache first
        if self.enable_cache {
            if let Some(cached_text) = self.cache.get(&path_str) {
                info!("Using cached text for: {}", path.display());
                return Ok(cached_text.clone());
            }
        }
        
        // Validate file exists
        if !path.exists() {
            return Err(ResumeAlignerError::InvalidInput(
                format!("File does not exist: {}", path.display())
            ));
        }
        
        // Detect file type
        let file_type = self.detect_file_type(path)?;
        
        // Route to appropriate extractor
        let text = match file_type {
            FileType::Pdf => {
                info!("Extracting text from PDF: {}", path.display());
                PdfExtractor.extract(path).await?
            },
            FileType::Text => {
                info!("Reading plain text file: {}", path.display());
                PlainTextExtractor.extract(path).await?
            },
            FileType::Markdown => {
                info!("Processing markdown file: {}", path.display());
                MarkdownExtractor.extract(path).await?
            },
            FileType::Unknown => {
                return Err(ResumeAlignerError::UnsupportedFormat(
                    format!("Unsupported file type for: {}", path.display())
                ));
            }
        };
        
        // Cache the result
        if self.enable_cache {
            self.cache.insert(path_str, text.clone());
        }
        
        Ok(text)
    }
    
    fn detect_file_type(&self, path: &Path) -> Result<FileType> {
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .ok_or_else(|| ResumeAlignerError::InvalidInput(
                format!("File has no extension: {}", path.display())
            ))?;
        
        Ok(FileType::from_extension(extension))
    }
    
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
    
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }
}


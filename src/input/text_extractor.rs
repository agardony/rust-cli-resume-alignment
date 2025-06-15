//! Text extraction from various file formats

use crate::error::{Result, ResumeAlignerError};
use pulldown_cmark::{html, Parser};
use std::path::Path;
use tokio::fs;

pub trait TextExtractor {
    fn extract(&self, path: &Path) -> impl std::future::Future<Output = Result<String>> + Send;
}

pub struct PdfExtractor;

impl TextExtractor for PdfExtractor {
    async fn extract(&self, path: &Path) -> Result<String> {
        let bytes = fs::read(path).await.map_err(|e| {
            ResumeAlignerError::Io(e)
        })?;

        let text = pdf_extract::extract_text_from_mem(&bytes).map_err(|e| {
            ResumeAlignerError::PdfExtraction(format!("Failed to extract text from PDF '{}': {}", path.display(), e))
        })?;
        Ok(text)
    }
}

pub struct PlainTextExtractor;

impl TextExtractor for PlainTextExtractor {
    async fn extract(&self, path: &Path) -> Result<String> {
        let content = fs::read_to_string(path).await.map_err(|e| {
            ResumeAlignerError::Io(e)
        })?;
        Ok(content)
    }
}

pub struct MarkdownExtractor;

impl TextExtractor for MarkdownExtractor {
    async fn extract(&self, path: &Path) -> Result<String> {
        let markdown_content = fs::read_to_string(path).await.map_err(|e| {
            ResumeAlignerError::Io(e)
        })?;

        let parser = Parser::new(&markdown_content);
        let mut html_output = String::new();
        html::push_html(&mut html_output, parser);

        let text = self.html_to_text(&html_output);
        Ok(text)
    }
}

impl MarkdownExtractor {
    fn html_to_text(&self, html: &str) -> String {
        let text = html
            .replace("<br>", "\n")
            .replace("</p>", "\n\n")
            .replace("&nbsp;", " ")
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&#39;", "'");

        let re = regex::Regex::new(r"<[^>]*>").unwrap();
        let clean_text = re.replace_all(&text, "");

        let lines: Vec<String> = clean_text
            .lines()
            .map(|line| line.trim().to_string())
            .filter(|line| !line.is_empty())
            .collect();

        lines.join("\n")
    }
}


//! Integration tests for the resume aligner

use resume_aligner::input::manager::InputManager;
use std::path::Path;

#[tokio::test]
async fn test_text_extraction_from_txt() {
    let mut manager = InputManager::new();
    let path = Path::new("tests/fixtures/sample_resume.txt");
    
    let result = manager.extract_text(path).await;
    assert!(result.is_ok());
    
    let text = result.unwrap();
    assert!(text.contains("John Doe"));
    assert!(text.contains("Software Engineer"));
    assert!(text.contains("React"));
    assert!(text.contains("Node.js"));
}

#[tokio::test]
async fn test_text_extraction_from_markdown() {
    let mut manager = InputManager::new();
    let path = Path::new("tests/fixtures/sample_resume.md");
    
    let result = manager.extract_text(path).await;
    assert!(result.is_ok());
    
    let text = result.unwrap();
    assert!(text.contains("John Doe"));
    assert!(text.contains("Software Engineer"));
    assert!(text.contains("React"));
    assert!(text.contains("Node.js"));
    // Should not contain markdown formatting
    assert!(!text.contains("**"));
    assert!(!text.contains("##"));
}

#[tokio::test]
async fn test_caching_functionality() {
    let mut manager = InputManager::new();
    let path = Path::new("tests/fixtures/sample_resume.txt");
    
    // First extraction
    let text1 = manager.extract_text(path).await.unwrap();
    assert_eq!(manager.cache_size(), 1);
    
    // Second extraction should use cache
    let text2 = manager.extract_text(path).await.unwrap();
    assert_eq!(text1, text2);
    assert_eq!(manager.cache_size(), 1);
}

#[tokio::test]
async fn test_unsupported_file_type() {
    let mut manager = InputManager::new();
    let path = Path::new("tests/fixtures/unsupported.xyz");
    
    let result = manager.extract_text(path).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_nonexistent_file() {
    let mut manager = InputManager::new();
    let path = Path::new("tests/fixtures/nonexistent.txt");
    
    let result = manager.extract_text(path).await;
    assert!(result.is_err());
}


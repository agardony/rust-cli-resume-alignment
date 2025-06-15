//! Error handling for the resume aligner application

use thiserror::Error;

#[derive(Error, Debug)]
pub enum ResumeAlignerError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("PDF extraction error: {0}")]
    PdfExtraction(String),

    #[error("Text processing error: {0}")]
    TextProcessing(String),

    #[error("Embedding generation error: {0}")]
    Embedding(String),

    #[error("LLM inference error: {0}")]
    LlmInference(String),

    #[error("Model loading error: {0}")]
    ModelLoading(String),

    #[error("Model error: {0}")]
    ModelError(String),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("File format not supported: {0}")]
    UnsupportedFormat(String),

    #[error("Network error: {0}")]
    Network(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Analysis failed: {0}")]
    AnalysisFailed(String),

    #[error("Processing error: {0}")]
    Processing(String),
    
    #[error("Output formatting error: {0}")]
    OutputFormatting(String),
}

pub type Result<T> = std::result::Result<T, ResumeAlignerError>;

/// Convert anyhow errors to our custom error type
impl From<anyhow::Error> for ResumeAlignerError {
    fn from(err: anyhow::Error) -> Self {
        ResumeAlignerError::AnalysisFailed(err.to_string())
    }
}

/// Convert candle core errors to our custom error type
impl From<candle_core::Error> for ResumeAlignerError {
    fn from(err: candle_core::Error) -> Self {
        ResumeAlignerError::ModelError(err.to_string())
    }
}


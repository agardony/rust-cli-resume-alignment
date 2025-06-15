//! CLI interface for the resume aligner

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "resume-aligner")]
#[command(about = "AI-powered resume and job description alignment tool")]
#[command(long_about = "Analyze resume compatibility with job descriptions using embeddings, ATS keywords, and local LLM insights")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
    
    /// Enable verbose logging
    #[arg(short, long, global = true)]
    pub verbose: bool,
    
    /// Configuration file path
    #[arg(short, long, global = true)]
    pub config: Option<PathBuf>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Align resume with job description
    Align {
        /// Path to resume file (PDF, TXT, MD)
        #[arg(short, long)]
        resume: PathBuf,
        
        /// Path to job description file (TXT, MD)
        #[arg(short, long)]
        job: PathBuf,
        
        /// LLM model to use for analysis
        #[arg(short, long)]
        llm: Option<String>,
        
        /// Embedding model to use
        #[arg(short, long)]
        embedding: Option<String>,
        
        /// Output detailed analysis
        #[arg(short, long)]
        detailed: bool,
        
        /// Output format: console, json, markdown, html, pdf
        #[arg(short, long, default_value = "console")]
        output: String,
        
        /// Save output to file
        #[arg(short, long)]
        save: Option<PathBuf>,
        
        /// Skip LLM analysis (embeddings + keywords only)
        #[arg(long)]
        no_llm: bool,
    },
    
    /// Model management commands
    Models {
        #[command(subcommand)]
        action: ModelAction,
    },
    
    /// Show configuration
    Config {
        #[command(subcommand)]
        action: Option<ConfigAction>,
    },
}

#[derive(Subcommand)]
pub enum ModelAction {
    /// List available models
    List {
        /// Show only embedding models
        #[arg(long)]
        embeddings: bool,
        
        /// Show only LLM models
        #[arg(long)]
        llms: bool,
    },
    
    /// Download a model
    Download {
        /// Model name or HuggingFace repo ID
        model: String,
        
        /// Force re-download if model exists
        #[arg(short, long)]
        force: bool,
    },
    
    /// Remove a downloaded model
    Remove {
        /// Model name to remove
        model: String,
    },
    
    /// Show model information
    Info {
        /// Model name
        model: String,
    },
}

#[derive(Subcommand)]
pub enum ConfigAction {
    /// Show current configuration
    Show,
    
    /// Edit configuration file
    Edit,
    
    /// Reset configuration to defaults
    Reset,
    
    /// Set a configuration value
    Set {
        /// Configuration key (e.g., "scoring.embedding_weight")
        key: String,
        
        /// Configuration value
        value: String,
    },
}

/// Parse and validate output format
pub fn parse_output_format(format: &str) -> Result<crate::config::OutputFormat, String> {
    match format.to_lowercase().as_str() {
        "console" => Ok(crate::config::OutputFormat::Console),
        "json" => Ok(crate::config::OutputFormat::Json),
        "markdown" | "md" => Ok(crate::config::OutputFormat::Markdown),
        "html" => Ok(crate::config::OutputFormat::Html),
        "pdf" => Ok(crate::config::OutputFormat::Pdf),
        _ => Err(format!("Invalid output format: {}. Supported: console, json, markdown, html, pdf", format)),
    }
}

/// Validate file extension
pub fn validate_file_extension(path: &PathBuf, allowed_extensions: &[&str]) -> Result<(), String> {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) => {
            if allowed_extensions.contains(&ext.to_lowercase().as_str()) {
                Ok(())
            } else {
                Err(format!(
                    "Unsupported file extension: .{}. Allowed: {}",
                    ext,
                    allowed_extensions.join(", ")
                ))
            }
        }
        None => Err("File has no extension".to_string()),
    }
}


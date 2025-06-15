//! Resume aligner library

pub mod cli;
pub mod config;
pub mod error;
pub mod input;
pub mod processing;
pub mod llm;
pub mod output;

pub use error::{Result, ResumeAlignerError};
pub use config::Config;


//! Resume aligner: AI-powered resume and job description alignment tool

mod cli;
mod config;
mod error;
mod input;
mod processing;
mod llm;
mod output;

use clap::Parser;
use cli::{Cli, Commands, ModelAction, ConfigAction};
use config::Config;
use error::{Result, ResumeAlignerError};
use input::manager::InputManager;
use log::{info, error};
use std::process;
use processing::{document::{Document, DocumentType}, text_processor::TextProcessor, analyzer::AnalysisEngine};
use llm::model_manager::ModelManager;

#[tokio::main]
async fn main() {
    // Parse CLI arguments
    let cli = Cli::parse();
    
    // Initialize logging
    let log_level = if cli.verbose {
        "debug"
    } else {
        "info"
    };
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or(log_level)
    ).init();
    
    // Load configuration
    let config = match Config::load() {
        Ok(config) => config,
        Err(e) => {
            error!("Failed to load configuration: {}", e);
            process::exit(1);
        }
    };
    
    // Execute command
    if let Err(e) = run_command(cli.command, config).await {
        error!("Command failed: {}", e);
        process::exit(1);
    }
}

async fn run_command(command: Commands, config: Config) -> Result<()> {
    match command {
        Commands::Align {
            resume,
            job,
            llm,
            embedding,
            detailed,
            output,
            save,
            no_llm,
        } => {
            info!("Starting resume alignment analysis");
            
            // Validate input files
            cli::validate_file_extension(&resume, &["pdf", "txt", "md"])
                .map_err(|e| ResumeAlignerError::InvalidInput(format!("Resume file: {}", e)))?;
            
            cli::validate_file_extension(&job, &["txt", "md"])
                .map_err(|e| ResumeAlignerError::InvalidInput(format!("Job description file: {}", e)))?;
            
            // Parse output format
            let output_format = cli::parse_output_format(&output)
                .map_err(|e| ResumeAlignerError::InvalidInput(e))?;
            
            println!("🚀 Resume alignment analysis");
            println!("📄 Resume: {}", resume.display());
            println!("💼 Job Description: {}", job.display());
            println!("🔧 Output Format: {:?}", output_format);
            
            if let Some(embedding_model) = &embedding {
                println!("🧠 Embedding Model: {}", embedding_model);
            }
            
            if !no_llm {
                if let Some(llm_model) = &llm {
                    println!("🤖 LLM Model: {}", llm_model);
                }
            } else {
                println!("⚠️  LLM analysis disabled");
            }
            
            if detailed {
                println!("📊 Detailed analysis enabled");
            }
            
            println!("\n📂 Extracting text from files...");
            
            // Initialize input manager
            let mut input_manager = InputManager::new();
            
            // Extract resume text
            println!("📄 Processing resume...");
            let resume_text = input_manager.extract_text(&resume).await?;
            
            // Extract job description text
            println!("💼 Processing job description...");
            let job_text = input_manager.extract_text(&job).await?;
            
            println!("\n📊 Text extraction completed!");
            println!("Resume text length: {} characters", resume_text.len());
            println!("Job description length: {} characters", job_text.len());
            
            if detailed {
                println!("\n📄 Resume Content Preview:");
                println!("{}", truncate_text(&resume_text, 300));
                
                println!("\n💼 Job Description Preview:");
                println!("{}", truncate_text(&job_text, 300));
            }
            
            println!("\n✅ Phase 2 (Input Pipeline) completed successfully!");
            
            // Phase 3: Text Processing and Document Analysis
            println!("\n🔄 Phase 3: Starting text processing and document analysis...");
            
            // Initialize text processor
            let text_processor = TextProcessor::new();
            
            // Create document structures
            let mut resume_doc = Document::new(
                resume_text.clone(),
                resume.to_string_lossy().to_string(),
                DocumentType::Resume,
            );
            
            let mut job_doc = Document::new(
                job_text.clone(),
                job.to_string_lossy().to_string(),
                DocumentType::JobDescription,
            );
            
            // Process documents with chunking and section detection
            println!("📄 Processing resume document...");
            let processed_resume = resume_doc.process(
                config.processing.chunk_size,
                config.processing.chunk_overlap,
            )?;
            
            println!("💼 Processing job description document...");
            let processed_job = job_doc.process(
                config.processing.chunk_size,
                config.processing.chunk_overlap,
            )?;
            
            // Text similarity analysis
            println!("\n🔍 Analyzing text similarity...");
            let text_similarity = text_processor.text_similarity(&resume_text, &job_text)?;
            
            // Display results
            println!("\n📊 Processing Results:");
            println!("\n📄 Resume Analysis:");
            println!("  • Chunks created: {}", processed_resume.chunks.len());
            println!("  • Sections detected: {}", processed_resume.sections.len());
            println!("  • Keywords extracted: {}", processed_resume.keywords.len());
            
            if let Some(title) = &processed_resume.original.metadata.title {
                println!("  • Detected title: {}", title);
            }
            
            println!("\n💼 Job Description Analysis:");
            println!("  • Chunks created: {}", processed_job.chunks.len());
            println!("  • Sections detected: {}", processed_job.sections.len());
            println!("  • Keywords extracted: {}", processed_job.keywords.len());
            
            if let Some(title) = &processed_job.original.metadata.title {
                println!("  • Detected title: {}", title);
            }
            
            println!("\n📈 Similarity Analysis:");
            println!("  • Text similarity (Jaccard): {:.1}%", text_similarity * 100.0);
            
            if detailed {
                println!("\n📄 Resume Sections Detected:");
                for section in &processed_resume.sections {
                    println!("  • {}: {} characters", section.section_type, section.content.len());
                    if !section.keywords.is_empty() {
                        println!("    Keywords: {}", section.keywords.join(", "));
                    }
                }
                
                println!("\n💼 Job Description Sections Detected:");
                for section in &processed_job.sections {
                    println!("  • {}: {} characters", section.section_type, section.content.len());
                    if !section.keywords.is_empty() {
                        println!("    Keywords: {}", section.keywords.join(", "));
                    }
                }
                
                println!("\n🔤 Top Resume Keywords:");
                for (i, keyword) in processed_resume.keywords.iter().take(10).enumerate() {
                    println!("  {}. {}", i + 1, keyword);
                }
                
                println!("\n🔤 Top Job Keywords:");
                for (i, keyword) in processed_job.keywords.iter().take(10).enumerate() {
                    println!("  {}. {}", i + 1, keyword);
                }
                
                println!("\n📊 Chunk Analysis:");
                println!("  Resume chunks:");
                for (i, chunk) in processed_resume.chunks.iter().take(3).enumerate() {
                    println!("    Chunk {}: {} chars ({})", 
                        i + 1, 
                        chunk.content.len(),
                        chunk.section_type.as_ref().map(|s| s.to_string()).unwrap_or("General".to_string())
                    );
                }
                if processed_resume.chunks.len() > 3 {
                    println!("    ... and {} more chunks", processed_resume.chunks.len() - 3);
                }
                
                println!("  Job description chunks:");
                for (i, chunk) in processed_job.chunks.iter().take(3).enumerate() {
                    println!("    Chunk {}: {} chars ({})", 
                        i + 1, 
                        chunk.content.len(),
                        chunk.section_type.as_ref().map(|s| s.to_string()).unwrap_or("General".to_string())
                    );
                }
                if processed_job.chunks.len() > 3 {
                    println!("    ... and {} more chunks", processed_job.chunks.len() - 3);
                }
            }
            
            println!("\n✅ Phase 3 (Text Processing) completed successfully!");
            
            // Phase 4: Comprehensive Analysis Engine
            println!("\n🔄 Phase 4: Starting comprehensive alignment analysis...");
            
            // Initialize analysis engine
            println!("🧠 Initializing analysis engine (embeddings + ATS matching)...");
            let mut analysis_engine = match AnalysisEngine::new(&config).await {
                Ok(engine) => {
                    println!("✅ Analysis engine initialized successfully!");
                    engine
                }
                Err(e) => {
                    println!("⚠️  Warning: Could not initialize full analysis engine: {}", e);
                    println!("📊 Proceeding with basic ATS analysis only...");
                    
                    // Fall back to basic ATS analysis
                    let ats_matcher = processing::ats_matcher::ATSMatcher::new()?;
                    let ats_score = ats_matcher.calculate_ats_score(&processed_resume, &processed_job)?;
                    
                    println!("\n📊 ATS Analysis Results:");
                    println!("  • Overall ATS Score: {:.1}%", ats_score.overall_score * 100.0);
                    println!("  • Keyword Coverage: {:.1}%", ats_score.keyword_coverage * 100.0);
                    println!("  • Exact Matches: {}", ats_score.exact_matches.len());
                    println!("  • Fuzzy Matches: {}", ats_score.fuzzy_matches.len());
                    
                    if detailed {
                        println!("\n🎯 Exact Keyword Matches:");
                        for (i, keyword_match) in ats_score.exact_matches.iter().take(10).enumerate() {
                            println!("  {}. {} (found {} times)", i + 1, keyword_match.keyword, keyword_match.count);
                        }
                        
                        if !ats_score.missing_keywords.is_empty() {
                            println!("\n⚠️  Missing Keywords (first 10):");
                            for (i, missing) in ats_score.missing_keywords.iter().take(10).enumerate() {
                                println!("  {}. {}", i + 1, missing);
                            }
                        }
                        
                        println!("\n📈 Skill Category Breakdown:");
                        println!("  • Technical Skills: {:.1}%", ats_score.skill_category_scores.technical_skills * 100.0);
                        println!("  • Soft Skills: {:.1}%", ats_score.skill_category_scores.soft_skills * 100.0);
                        println!("  • Role-Specific: {:.1}%", ats_score.skill_category_scores.role_specific * 100.0);
                    }
                    
                    println!("\n✅ Phase 4 (Basic ATS Analysis) completed!");
                    return Ok(());
                }
            };
            
            // Perform comprehensive analysis
            println!("🔍 Performing comprehensive alignment analysis...");
            let alignment_report = if no_llm {
                analysis_engine.analyze_alignment(&processed_resume, &processed_job).await?
            } else {
                analysis_engine.analyze_alignment_with_llm(&processed_resume, &processed_job, llm.clone()).await?
            };
            
            println!("\n✅ Phase 4 (Comprehensive Analysis) completed successfully!");

            // Phase 5: LLM Analysis (if enabled)
            let llm_analysis = if !no_llm {
                println!("\n🔄 Phase 5: Starting LLM analysis...");
                
                // Initialize LLM analyzer
                match llm::analyzer::LLMAnalyzer::new(config.clone()).await {
                    Ok(mut llm_analyzer) => {
                        // Try to initialize the LLM engine
                        if let Err(e) = llm_analyzer.initialize_engine(llm.clone()).await {
                            println!("⚠️  Warning: Could not initialize LLM engine: {}", e);
                            println!("📊 Proceeding without LLM analysis...");
                            None
                        } else {
                            println!("🤖 LLM engine initialized successfully!");
                            
                            // Prepare existing analysis data for LLM
                            let existing_analysis = llm::analyzer::ExistingAnalysis {
                                overall_score: alignment_report.overall_score,
                                embedding_score: alignment_report.embedding_score,
                                keyword_score: alignment_report.ats_score,
                                missing_keywords: alignment_report.gap_analysis.missing_keywords.clone(),
                                exact_matches: alignment_report.keyword_analysis.exact_matches.iter()
                                    .map(|m| llm::analyzer::ExactMatch {
                                        keyword: m.keyword.clone(),
                                        count: m.count,
                                    })
                                    .collect(),
                                section_scores: alignment_report.section_analysis.section_scores.iter()
                                    .map(|(name, score)| (name.clone(), llm::analyzer::SectionScore {
                                        combined_score: score.combined_score,
                                        embedding_score: score.embedding_score,
                                        keyword_score: score.keyword_score,
                                    }))
                                    .collect(),
                            };
                            
                            // Perform LLM analysis
                            match llm_analyzer.analyze(&processed_resume, &processed_job, &existing_analysis).await {
                                Ok(llm_result) => {
                                    println!("✅ LLM analysis completed!");
                                    println!("🎯 LLM Score: {:.1}% (confidence: {:.1}%)", 
                                        llm_result.overall_score * 100.0, 
                                        llm_result.confidence * 100.0
                                    );
                                    println!("📊 Token usage: {} total ({} in, {} out)", 
                                        llm_result.token_usage.total_tokens,
                                        llm_result.token_usage.input_tokens,
                                        llm_result.token_usage.output_tokens
                                    );
                                    Some(llm_result)
                                }
                                Err(e) => {
                                    println!("⚠️  LLM analysis failed: {}", e);
                                    println!("📊 Proceeding with embeddings + ATS analysis only...");
                                    None
                                }
                            }
                        }
                    }
                    Err(e) => {
                        println!("⚠️  Could not initialize LLM analyzer: {}", e);
                        println!("📊 Proceeding with embeddings + ATS analysis only...");
                        None
                    }
                }
            } else {
                println!("\n⚠️  LLM analysis disabled (--no-llm flag)");
                None
            };

            // Phase 6: Generate Comprehensive Report
            println!("\n🔄 Phase 6: Generating comprehensive report...");

            // Create component weights from config
            let component_weights = output::ComponentWeights {
                embedding_weight: config.scoring.embedding_weight,
                keyword_weight: config.scoring.keyword_weight,
                llm_weight: config.scoring.llm_weight,
            };

            // Generate comprehensive report
            let comprehensive_report = output::ComprehensiveReport::from_alignment_analysis(
                alignment_report.clone(),
                llm_analysis,
                component_weights,
            );

            // Initialize report generator
            let report_generator = if detailed {
                output::ReportGenerator::with_options(
                    true,   // use_colors
                    true,   // detailed
                    true,   // pretty_json
                    true,   // include_metadata
                    true,   // include_html_styles
                    false,  // include_pdf_charts
                )
            } else {
                output::ReportGenerator::new()
            };

            // Generate the formatted output
            let formatted_output = report_generator.generate_report(&comprehensive_report, &output_format)?;

            // Display the report
            match output_format {
                config::OutputFormat::Console => {
                    println!("\n{}", formatted_output);
                }
                _ => {
                    // For non-console formats, show a brief summary and save info
                    println!("\n📊 Final Analysis Summary:");
                    println!("Overall Score: {}% - {}", 
                        comprehensive_report.analysis_summary.overall_score_percentage,
                        comprehensive_report.analysis_summary.verdict
                    );
                    
                    if let Some(save_path) = save {
                        // Save to specified file
                        output::save_report_to_file(&formatted_output, &save_path)?;
                        println!("💾 Report saved to: {}", save_path.display());
                    } else {
                        // Auto-generate filename and save
                        let filename = output::suggest_filename(
                            &output_format, 
                            &resume.to_string_lossy(), 
                            true
                        );
                        let save_path = std::env::current_dir()?.join(filename);
                        output::save_report_to_file(&formatted_output, &save_path)?;
                        println!("💾 Report saved to: {}", save_path.display());
                    }
                }
            }

            // Show quick recommendations preview for all formats
            if !comprehensive_report.recommendations.is_empty() {
                println!("\n💡 Top {} Recommendations:", 
                    comprehensive_report.recommendations.len().min(3)
                );
                for (i, rec) in comprehensive_report.recommendations.iter().take(3).enumerate() {
                    let priority_icon = match rec.priority {
                        output::RecommendationPriority::Critical => "🚨",
                        output::RecommendationPriority::High => "⚠️",
                        output::RecommendationPriority::Medium => "📋",
                        output::RecommendationPriority::Low => "💡",
                    };
                    println!("  {}. {} {}", i + 1, priority_icon, rec.title);
                }
            }

            println!("\n✅ Phase 6 (Comprehensive Reporting) completed successfully!");
            println!("🎉 Analysis complete! Overall alignment score: {:.1}%", 
                comprehensive_report.analysis_summary.overall_score_percentage
            );
        }
        
        Commands::Models { action } => {
            match action {
                ModelAction::List { embeddings, llms } => {
                    println!("📚 Available Models\n");
                    
                    // Initialize model manager to check download status
                    let model_manager = ModelManager::new(config.get_models_dir()).await?;
                    
                    if !llms {
                        println!("🧠 Embedding Models:");
                        for model in config.list_embedding_models() {
                            println!("  • {} ({}) - {} MB", 
                                model.name, 
                                model.repo_id, 
                                model.size_mb
                            );
                            println!("    {}", model.description);
                        }
                        println!();
                    }
                    
                    if !embeddings {
                        println!("🤖 LLM Models:");
                        let downloaded_models = model_manager.list_downloaded_models();
                        
                        for model_info in model_manager.list_available_models() {
                            // Get the model key/ID for matching
                            let model_key = if model_info.repo_id.contains("Phi-4-mini") {
                                "phi-4-mini"
                            } else if model_info.repo_id.contains("Llama-3.2") {
                                "llama-3.2-3b"
                            } else if model_info.repo_id.contains("Llama-3.1") {
                                "llama-3.1-8b"
                            } else {
                                "unknown"
                            };
                            
                            let is_downloaded = model_manager.is_model_downloaded(model_key);
                            
                            let status = if is_downloaded { "✅ Downloaded" } else { "⬇️  Available" };
                            
                            println!("  • {} ({}) - {:.1} GB [{}]", 
                                model_info.name, 
                                model_info.repo_id, 
                                model_info.size_mb as f64 / 1024.0,
                                status
                            );
                            println!("    {}", model_info.description);
                            
                            // Show model ID for download commands
                            let model_id = if model_info.repo_id.contains("Phi-4-mini") {
                                "phi-4-mini"
                            } else if model_info.repo_id.contains("Llama-3.2") {
                                "llama-3.2-3b"
                            } else if model_info.repo_id.contains("Llama-3.1") {
                                "llama-3.1-8b"
                            } else {
                                "unknown"
                            };
                            
                            if !is_downloaded {
                                println!("    💡 Download: resume-aligner models download {}", model_id);
                            }
                            println!();
                        }
                        
                        if downloaded_models.is_empty() {
                            println!("\n💡 No models downloaded yet. Get started with:");
                            println!("   resume-aligner models download phi-4-mini");
                            println!("\n🎯 Recommended for beginners: phi-4-mini (2.8 GB)");
                            println!("🚀 Best quality: llama-3.1-8b (8 GB)");
                            println!("⚡ Optimal balance: llama-3.2-3b (3.1 GB)");
                        }
                    }
                }
                
                ModelAction::Download { model, force } => {
                    println!("⬇️  Downloading model: {}", model);
                    if force {
                        println!("🔄 Force download enabled");
                    }
                    
                    // Initialize model manager
                    let mut model_manager = ModelManager::new(config.get_models_dir()).await?;
                    
                    // Check if model is already downloaded
                    if !force && model_manager.is_model_downloaded(&model) {
                        println!("✅ Model '{}' is already downloaded!", model);
                        println!("💡 Use --force to re-download");
                        return Ok(());
                    }
                    
                    // Download the model
                    match model_manager.download_model(&model).await {
                        Ok(model_path) => {
                            println!("✅ Model '{}' downloaded successfully!", model);
                            println!("📁 Location: {}", model_path.display());
                        }
                        Err(e) => {
                            println!("❌ Failed to download model '{}': {}", model, e);
                            return Err(e);
                        }
                    }
                }
                
                ModelAction::Remove { model } => {
                    println!("🗑️  Removing model: {}", model);
                    
                    // Initialize model manager
                    let model_manager = ModelManager::new(config.get_models_dir()).await?;
                    
                    // Check if model is downloaded
                    if !model_manager.is_model_downloaded(&model) {
                        println!("⚠️  Model '{}' is not downloaded", model);
                        return Ok(());
                    }
                    
                    // Get model path and remove
                    let model_path = config.get_models_dir().join(&model);
                    if model_path.exists() {
                        match std::fs::remove_dir_all(&model_path) {
                            Ok(()) => {
                                println!("✅ Model '{}' removed successfully!", model);
                                println!("📁 Removed directory: {}", model_path.display());
                            }
                            Err(e) => {
                                println!("❌ Failed to remove model '{}': {}", model, e);
                                return Err(ResumeAlignerError::ModelError(format!("Failed to remove model: {}", e)));
                            }
                        }
                    } else {
                        println!("⚠️  Model directory not found: {}", model_path.display());
                    }
                }
                
                ModelAction::Info { model } => {
                    println!("📋 Model Information for '{}'\n", model);
                    
                    // Initialize model manager to check download status
                    let model_manager = ModelManager::new(config.get_models_dir()).await?;
                    
                    if let Some(model_info) = model_manager.get_model_info(&model) {
                        println!("Name: {}", model_info.name);
                        println!("Repository: {}", model_info.repo_id);
                        println!("Type: {:?}", model_info.model_type);
                        println!("Size: {} MB ({} GB)", model_info.size_mb, model_info.size_mb as f64 / 1024.0);
                        println!("Description: {}", model_info.description);
                        
                        // Check if downloaded
                        let is_downloaded = model_manager.is_model_downloaded(&model);
                        println!("Status: {}", if is_downloaded { "✅ Downloaded" } else { "⬇️  Available for download" });
                        
                        if is_downloaded {
                            if let Some(model_path) = model_manager.get_model_path(&model) {
                                println!("Location: {}", model_path.display());
                            }
                        }
                        
                        if !model_info.capabilities.is_empty() {
                            println!("\nCapabilities:");
                            for capability in &model_info.capabilities {
                                println!("  • {}", capability);
                            }
                        }
                        
                        if !is_downloaded {
                            println!("\n💡 To download this model, run:");
                            println!("   resume-aligner models download {}", model);
                        }
                    } else {
                        return Err(ResumeAlignerError::ModelNotFound(model));
                    }
                }
            }
        }
        
        Commands::Config { action } => {
            match action {
                Some(ConfigAction::Show) | None => {
                    println!("⚙️  Current Configuration\n");
                    println!("Models Directory: {}", config.models_dir().display());
                    println!("Default Embedding Model: {}", config.models.default_embedding_model);
                    println!("Default LLM Model: {}", config.models.default_llm_model);
                    println!("\nScoring Weights:");
                    println!("  Embeddings: {:.1}%", config.scoring.embedding_weight * 100.0);
                    println!("  Keywords: {:.1}%", config.scoring.keyword_weight * 100.0);
                    println!("  LLM: {:.1}%", config.scoring.llm_weight * 100.0);
                }
                
                Some(ConfigAction::Edit) => {
                    println!("📝 Opening configuration editor...");
                    // TODO: Implement config editor
                    println!("💡 Feature coming soon!");
                }
                
                Some(ConfigAction::Reset) => {
                    println!("🔄 Resetting configuration to defaults...");
                    let default_config = Config::default();
                    default_config.save()?;
                    println!("✅ Configuration reset successfully!");
                }
                
                Some(ConfigAction::Set { key, value }) => {
                    println!("🔧 Setting {}: {}", key, value);
                    // TODO: Implement config setting
                    println!("💡 Feature coming soon!");
                }
            }
        }
    }
    
    Ok(())
}

/// Truncate text to a maximum length with ellipsis
fn truncate_text(text: &str, max_length: usize) -> String {
    if text.len() <= max_length {
        text.to_string()
    } else {
        let truncated = &text[..max_length.min(text.len())];
        // Find the last word boundary to avoid cutting words
        let last_space = truncated.rfind(' ').unwrap_or(max_length);
        format!("{}...", &text[..last_space])
    }
}

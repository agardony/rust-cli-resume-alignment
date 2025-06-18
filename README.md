# 🎯 Resume Aligner

[![Rust](https://img.shields.io/badge/rust-stable-brightgreen.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

**AI-powered resume and job description alignment tool** that helps job seekers optimize their resumes for specific positions using advanced natural language processing, ATS keyword matching, and local LLM analysis.

## ✨ Features

### 🔍 **Multi-Modal Analysis**
- **Semantic Analysis**: Model2Vec embeddings for deep text understanding
- **ATS Keyword Matching**: Exact and fuzzy matching with 80+ skill database
- **LLM Insights**: Local AI analysis with Llama 3.2, Phi-4, or Llama 3.1 models
- **Weighted Scoring**: Configurable 30/40/30 split (embeddings/keywords/LLM)

### 📄 **File Format Support**
- **Resume**: PDF, TXT, Markdown
- **Job Descriptions**: TXT, Markdown
- **Output**: Console, JSON, Markdown, HTML, PDF

### 🤖 **Local AI Models**
- **Llama 3.2 3B**: 3.1GB - **DEFAULT**
- **Phi-4-mini**: 2.8GB - Latest lightweight instruction-tuned model
- **Llama 3.1 8B**: 8GB - High-quality production model
- **Automatic Model Management**: Seamless download and caching

### 🎨 **Rich Output Formats**
- **Console**: Color-coded, emoji-rich terminal output
- **HTML**: Professional, styled reports for sharing
- **PDF**: Print-ready reports (perfect for novice users)
- **JSON**: Structured data for developers
- **Markdown**: Documentation-friendly format

### 🚀 **Performance & Privacy**
- **100% Local**: No data sent to external services
- **Fast Processing**: Optimized embeddings and caching
- **Memory Efficient**: Quantized model support
- **Graceful Degradation**: Works without LLM if needed

## 🚀 Quick Start

### Prerequisites

- **Rust** 1.70+ ([Install Rust](https://rustup.rs/))
- **macOS**, **Linux**, or **Windows**
- **4GB+ RAM** (8GB+ recommended for Llama models)
- **Internet connection** (for initial model download)

### Installation

```bash
# Clone the repository
git clone https://github.com/username/resume-aligner.git
cd resume-aligner

# Build and install
cargo install --path .
```

### First Run - Automatic Model Setup

```bash
# The tool will automatically download a model on first use
resume-aligner align --resume resume.pdf --job job.txt

# Or explicitly download your preferred model
resume-aligner models download phi-4-mini   # Recommended for beginners
resume-aligner models download llama-3.1-8b  # Best quality
resume-aligner models download llama-3.2-3b  # Optimal balance
```

### Basic Usage

```bash
# Quick analysis
resume-aligner align --resume resume.pdf --job job.txt

# Detailed analysis with specific model
resume-aligner align --resume resume.pdf --job job.txt --llm phi-4-mini --detailed

# Generate HTML report (perfect for sharing)
resume-aligner align --resume resume.pdf --job job.txt --output html --save report.html

# JSON output for developers
resume-aligner align --resume resume.pdf --job job.txt --output json --save analysis.json
```

## 📚 Model Management

### Available Models

| Model | Size | Best For | Download Command |
|-------|------|----------|------------------|
| **Llama 3.2 3B** | 3.1GB | **DEFAULT** | `models download llama-3.2-3b` |
| **Phi-4-mini** | 2.8GB | Latest lightweight, instruction-tuned | `models download phi-4-mini` |
| **Llama 3.1 8B** | 8GB | Production, highest quality | `models download llama-3.1-8b` |

### Model Commands

```bash
# List all available models
resume-aligner models list

# Download a specific model
resume-aligner models download phi-4-mini

# Get model information
resume-aligner models info phi-4-mini

# Remove a downloaded model
resume-aligner models remove phi-4-mini

# Force re-download
resume-aligner models download phi-4-mini --force
```

### Automatic Model Download

The tool features **seamless automatic model downloading**:

1. **First Run**: Automatically downloads `llama-3.2-3b` (default)
2. **Smart Selection**: Chooses best available model based on system resources
3. **Cached Storage**: Models stored in `~/.resume-aligner/models/`
4. **Progress Tracking**: Real-time download progress with file-by-file status
5. **Validation**: Automatic model integrity verification

## 📖 Usage Guide

### Command Line Interface

```bash
resume-aligner [OPTIONS] <COMMAND>

Commands:
  align     Align resume with job description
  models    Model management commands
  config    Show or modify configuration
  help      Print this message or the help of the given subcommand(s)

Options:
  -v, --verbose           Enable verbose logging
  -c, --config <CONFIG>   Configuration file path
  -h, --help              Print help
```

### Align Command Options

```bash
resume-aligner align [OPTIONS] --resume <RESUME> --job <JOB>

Options:
  -r, --resume <RESUME>         Path to resume file (PDF, TXT, MD)
  -j, --job <JOB>              Path to job description file (TXT, MD)
  -l, --llm <LLM>              LLM model to use for analysis
  -e, --embedding <EMBEDDING>   Embedding model to use
  -d, --detailed               Output detailed analysis
  -o, --output <OUTPUT>        Output format [default: console]
                               [possible values: console, json, markdown, html, pdf]
  -s, --save <SAVE>            Save output to file
      --no-llm                 Skip LLM analysis (embeddings + keywords only)
```

### Output Formats

#### Console Output (Default)
```bash
resume-aligner align --resume resume.pdf --job job.txt
```
Color-coded terminal output with emojis, progress bars, and structured sections.

#### HTML Report (Recommended for Sharing)
```bash
resume-aligner align --resume resume.pdf --job job.txt --output html --save report.html
```
Professional, styled HTML report perfect for sharing with recruiters or saving for later.

#### JSON Data (Developer-Friendly)
```bash
resume-aligner align --resume resume.pdf --job job.txt --output json --save data.json
```
Structured JSON with all analysis data for integration with other tools.

#### Markdown Documentation
```bash
resume-aligner align --resume resume.pdf --job job.txt --output markdown --save report.md
```
Clean markdown format perfect for documentation and version control.

#### PDF Report (Coming Soon)
```bash
resume-aligner align --resume resume.pdf --job job.txt --output pdf --save report.pdf
```
Print-ready PDF report with professional formatting.

## ⚙️ Configuration

### Hugging Face Token Setup (Recommended)

To avoid rate limiting when downloading models, set up a **free** Hugging Face API token:

1. **Create an account** at [huggingface.co](https://huggingface.co/join) (free)
2. **Generate a token** at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. **Set the token** in your environment:

```bash
# Option 1: Environment variable (recommended)
export HUGGING_FACE_HUB_TOKEN="your_token_here"

# Option 2: Save to file
echo "your_token_here" > ~/.cache/huggingface/token

# Option 3: Use huggingface-cli (if installed)
huggingface-cli login
```

**Benefits of using a token:**
- ✅ **Higher download limits** - No more rate limiting errors
- ✅ **Faster downloads** - Priority access to model files
- ✅ **Access to gated models** - Some models require acceptance of terms
- ✅ **100% Free** - No cost for downloading open-source models

**Note**: The token is only used for downloading models and does not incur any charges.

### Configuration File Location

- **macOS**: `~/.config/resume-aligner/config.toml`
- **Linux**: `~/.config/resume-aligner/config.toml`
- **Windows**: `%APPDATA%\resume-aligner\config.toml`

### Default Configuration

```toml
[models]
models_dir = "~/.resume-aligner/models"
default_embedding_model = "minishlab/M2V_base_output"
default_llm_model = "microsoft/Phi-4-mini-instruct"

[processing]
chunk_size = 512
chunk_overlap = 50
max_tokens = 4096
enable_caching = true
batch_size = 32

[scoring]
embedding_weight = 0.3
keyword_weight = 0.4
llm_weight = 0.3
min_similarity_threshold = 0.1

[output]
format = "console"
detailed = false
include_recommendations = true
color_output = true
```

### Configuration Commands

```bash
# Show current configuration
resume-aligner config show

# Reset to defaults
resume-aligner config reset

# Edit configuration (coming soon)
resume-aligner config edit
```

## 🏗️ Architecture

### Project Structure

```
resume-aligner/
├── src/
│   ├── main.rs              # CLI entry point
│   ├── cli.rs               # Command-line interface
│   ├── config.rs            # Configuration management
│   ├── error.rs             # Error handling
│   ├── input/               # File processing pipeline
│   │   ├── file_detector.rs # Auto file type detection
│   │   ├── text_extractor.rs# PDF/TXT/MD extraction
│   │   └── manager.rs       # Input coordination
│   ├── processing/          # Text analysis engines
│   │   ├── document.rs      # Document structures
│   │   ├── text_processor.rs# Text preprocessing
│   │   ├── embeddings.rs    # Model2Vec integration
│   │   ├── ats_matcher.rs   # Keyword matching
│   │   └── analyzer.rs      # Main analysis engine
│   ├── llm/                 # Local LLM integration
│   │   ├── model_manager.rs # Model download/management
│   │   ├── inference.rs     # Candle-based inference
│   │   ├── prompts.rs       # LLM prompt templates
│   │   └── analyzer.rs      # LLM analysis logic
│   └── output/              # Report generation
│       ├── report.rs        # Report structures
│       └── formatter.rs     # Multiple format support
├── tests/
│   ├── integration_tests.rs
│   └── fixtures/            # Test data
├── models/                  # Downloaded LLM models
└── Cargo.toml
```

### Implementation Phases

- ✅ **Phase 1**: Foundation (CLI, config, error handling)
- ✅ **Phase 2**: Input Pipeline (PDF/TXT/MD extraction)
- ✅ **Phase 3**: Text Processing (chunking, preprocessing)
- ✅ **Phase 4**: Analysis Engines (embeddings, ATS matching)
- ✅ **Phase 5**: LLM Integration (local model inference)
- ✅ **Phase 6**: Output System (multiple formats, rich reports)

## 📊 Analysis Components

### 1. Semantic Analysis (30% weight)
- **Model2Vec Embeddings**: Fast, efficient text embeddings
- **Cosine Similarity**: Document-level and chunk-level comparison
- **Section Matching**: Skills, experience, education analysis

### 2. ATS Keyword Matching (40% weight)
- **Exact Matching**: Aho-Corasick algorithm for precise matches
- **Fuzzy Matching**: Levenshtein distance for variations
- **Skill Database**: 80+ technical and soft skills
- **Category Scoring**: Technical, soft, and role-specific skills

### 3. LLM Analysis (30% weight)
- **Gap Analysis**: AI-identified missing qualifications
- **Recommendations**: Specific, actionable improvements
- **Natural Language**: Human-readable summaries
- **Interview Prep**: Talking points and potential questions

## 🧪 Testing

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Test specific module
cargo test processing::

# Integration tests
cargo test --test integration_tests
```

### Test Coverage
- **Unit Tests**: 26 tests covering all modules
- **Integration Tests**: End-to-end file processing
- **Fixtures**: Sample resume and job description files
- **Error Handling**: Comprehensive error scenario testing

## 🔧 Troubleshooting

### Model Download Issues

#### Rate Limiting (HTTP 429 Error)
```
Failed to download model: request error: HTTP status client error (429 Too Many Requests)
```
**Solution**: Set up a Hugging Face token (see [Configuration section](#%EF%B8%8F-configuration)) or wait a few minutes and retry.

#### Network Connection Issues
```
Failed to download model: Network error
```
**Solutions**:
- Check your internet connection
- Verify firewall settings allow HTTPS connections
- Try using a VPN if in a restricted network
- Ensure sufficient disk space (models are 1-8GB)

#### Corrupted Downloads
```
Model validation failed
```
**Solution**: Remove the model and re-download:
```bash
resume-aligner models remove phi-4-mini
resume-aligner models download phi-4-mini
```

### General Issues

#### PDF Processing Errors
- Ensure PDF is not password-protected
- Try converting to TXT format as workaround
- Check file permissions

#### Memory Issues
- Use `phi-4-mini` model for low-memory systems
- Close other applications during analysis
- Consider increasing system swap space

#### Performance Issues
- Enable caching in configuration
- Use smaller chunk sizes for large documents
- Consider using `--no-llm` flag for faster analysis

## 🚧 Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/username/resume-aligner.git
cd resume-aligner

# Install dependencies
cargo build

# Run in development mode
cargo run -- align --resume tests/fixtures/sample_resume.txt --job tests/fixtures/sample_job.txt

# Run with debug logging
RUST_LOG=debug cargo run -- align --resume tests/fixtures/sample_resume.txt --job tests/fixtures/sample_job.txt --verbose
```

### Key Dependencies

```toml
# Core ML/AI
model2vec-rs = { git = "https://github.com/MinishLab/model2vec-rs.git" }
candle-core = "0.9.1"
candle-transformers = "0.9.1"
hf-hub = { version = "0.3", features = ["tokio"] }

# Text Processing
aho-corasick = "1.0"          # Exact keyword matching
strsim = "0.10"               # Fuzzy string matching
pdf-extract = "0.6"           # PDF text extraction
pulldown-cmark = "0.9"        # Markdown parsing

# CLI & Config
clap = { version = "4.0", features = ["derive"] }
toml = "0.8"
serde = { version = "1.0", features = ["derive"] }

# Output Formats
askama = { version = "0.14", features = ["serde_json"] }  # HTML templates
printpdf = "0.7"              # PDF generation
colored = "2.0"               # Terminal colors
chrono = { version = "0.4", features = ["serde"] }        # Timestamps
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Areas for Contribution

1. **New Model Support**: Add support for more LLM models
2. **Output Formats**: Implement additional export formats
3. **ATS Database**: Expand keyword and skill databases
4. **Performance**: Optimize processing and inference
5. **UI/UX**: Improve CLI experience and output formatting

### Development Guidelines

1. **Follow Rust idioms** and use `clippy` for linting
2. **Add tests** for new functionality
3. **Update documentation** for any API changes
4. **Use conventional commits** for clear history
5. **Ensure backwards compatibility** when possible

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📈 Performance

### Benchmarks

- **PDF Processing**: ~500ms for typical resume
- **Embedding Generation**: ~200ms for document pair
- **ATS Analysis**: ~50ms for keyword matching
- **LLM Inference**: ~2-5s depending on model size
- **Total Analysis**: ~3-6s end-to-end

### Optimization Features

- **Caching**: Embeddings and LLM responses cached
- **Batching**: Efficient batch processing for chunks
- **Quantization**: Reduced memory usage for models
- **Streaming**: Progressive output for large documents

## 🔒 Privacy & Security

- **100% Local Processing**: No data sent to external services
- **No Telemetry**: No usage tracking or analytics
- **Secure Model Downloads**: Verified downloads from Hugging Face
- **File Handling**: Safe PDF parsing with bounds checking
- **Memory Safety**: Rust's memory safety guarantees

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Model2Vec](https://github.com/MinishLab/model2vec-rs)**: Fast, efficient embeddings
- **[Candle](https://github.com/huggingface/candle)**: Rust-native ML framework
- **[Hugging Face](https://huggingface.co/)**: Model hosting and distribution
- **[Microsoft](https://huggingface.co/microsoft/Phi-4-mini-instruct)**: Phi-4 model
- **[Meta](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)**: Llama 3.1 model

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/username/resume-aligner/issues)
- **Discussions**: [GitHub Discussions](https://github.com/username/resume-aligner/discussions)
- **Documentation**: This README and inline code documentation

---

**Made with ❤️ for job seekers everywhere. Good luck with your applications!** 🚀


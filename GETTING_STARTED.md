# 🚀 Getting Started with Resume Aligner

> **Quick guide to try out the Resume Aligner project**

Welcome! This guide will help you build and test the Resume Aligner to see its current capabilities.

---

## ⚡ **Quick Start (5 minutes)**

### 1. **Build the Project**
```bash
# Clone and build (if you haven't already)
cd /Users/agardony/Projects/rust-resume-alignment
cargo build --release
```

### 2. **Try Basic Analysis** 
```bash
# Analyze resume against job description (using built-in test files)
./target/release/resume-aligner align \
  --resume tests/fixtures/sample_resume.txt \
  --job tests/fixtures/sample_job.txt
```

**What you'll see:**
- ✅ Text extraction and processing
- 📊 Basic ATS keyword matching (21.5% match)
- 📈 Text similarity analysis (22.2%)
- 🔍 Keyword coverage analysis

---

## 📋 **What Currently Works**

### ✅ **Fully Functional Features**

1. **📄 File Processing**
   - PDF, TXT, and Markdown resume extraction
   - Text cleaning and preprocessing
   - Document chunking and section detection

2. **🔍 ATS Analysis**
   - 80+ keyword database for skill matching
   - Exact and fuzzy keyword matching
   - Industry-standard ATS scoring

3. **📊 Text Analytics**
   - Jaccard similarity scoring
   - Section-wise analysis
   - Keyword extraction and classification

4. **🖥️ CLI Interface**
   - Model management commands
   - Configuration viewing
   - Multiple output formats (console, JSON, etc.)

### ⚠️ **Partially Working Features**

1. **🧠 Embeddings** - Model download needed
2. **🤖 LLM Analysis** - Requires model download
3. **🎨 Rich Output** - HTML/PDF formatters implemented but not connected

---

## 🧪 **Testing Different Scenarios**

### **Basic Analysis (No Models Required)**
```bash
# Skip LLM analysis, use only ATS matching
./target/release/resume-aligner align \
  --resume tests/fixtures/sample_resume.txt \
  --job tests/fixtures/sample_job.txt \
  --no-llm
```

### **Detailed Analysis**
```bash
# Get more detailed breakdown
./target/release/resume-aligner align \
  --resume tests/fixtures/sample_resume.txt \
  --job tests/fixtures/sample_job.txt \
  --detailed
```

### **Different Output Formats**
```bash
# JSON output (structured data)
./target/release/resume-aligner align \
  --resume tests/fixtures/sample_resume.txt \
  --job tests/fixtures/sample_job.txt \
  --output json

# Save results to file
./target/release/resume-aligner align \
  --resume tests/fixtures/sample_resume.txt \
  --job tests/fixtures/sample_job.txt \
  --output json \
  --save analysis_results.json
```

---

## 🤖 **Advanced: Download AI Models** 

> **Note**: This requires internet connection and disk space

### **Download a Lightweight Model**
```bash
# Download newest lightweight model (2.8 GB)
./target/release/resume-aligner models download phi-4-mini

# Then run full analysis with AI
./target/release/resume-aligner align \
  --resume tests/fixtures/sample_resume.txt \
  --job tests/fixtures/sample_job.txt \
  --llm phi-4-mini
```

### **Check Available Models**
```bash
# See all available models
./target/release/resume-aligner models list

# Get detailed model info
./target/release/resume-aligner models info phi-4-mini
```

---

## 📁 **Using Your Own Files**

### **Test with Your Resume**
```bash
# Create your test files
mkdir -p my_test
echo "Your resume content here..." > my_test/my_resume.txt
echo "Job description here..." > my_test/job_posting.txt

# Analyze
./target/release/resume-aligner align \
  --resume my_test/my_resume.txt \
  --job my_test/job_posting.txt \
  --detailed
```

### **Supported File Formats**
- **Resume**: PDF, TXT, Markdown
- **Job Description**: TXT, Markdown

---

## 🔧 **Configuration & Settings**

### **View Current Configuration**
```bash
./target/release/resume-aligner config
```

### **Configuration Location**
- **File**: `~/.config/resume-aligner/config.toml`
- **Models**: `~/.resume-aligner/models/`

### **Customizable Settings**
- Scoring weights (embeddings 30%, keywords 40%, LLM 30%)
- Model selection
- Processing parameters

---

## 📊 **Sample Analysis Output**

```
🚀 Resume alignment analysis
📄 Resume: tests/fixtures/sample_resume.txt
💼 Job Description: tests/fixtures/sample_job.txt

📊 ATS Analysis Results:
  • Overall ATS Score: 21.5%
  • Keyword Coverage: 22.2%
  • Exact Matches: 23
  • Fuzzy Matches: 34

📈 Similarity Analysis:
  • Text similarity (Jaccard): 22.2%

📄 Resume Analysis:
  • Chunks created: 2
  • Sections detected: 3
  • Keywords extracted: 20
```

---

## 🎯 **What the Analysis Tells You**

### **ATS Score (21.5%)**
- How well your resume matches ATS keyword requirements
- Higher = better chance of passing automated screening

### **Text Similarity (22.2%)**
- Semantic overlap between resume and job description
- Indicates content relevance

### **Keyword Matches**
- **Exact**: Perfect keyword matches
- **Fuzzy**: Similar keywords (e.g., "JavaScript" vs "JS")

### **Section Analysis**
- Detected resume sections (Experience, Skills, Education)
- Content organization and structure

---

## 🚧 **Current Limitations**

1. **🧠 AI Features**: Require model downloads (1-8 GB)
2. **📊 Rich Reports**: HTML/PDF output not fully connected
3. **🎨 Recommendations**: LLM-powered suggestions need models
4. **⚡ Performance**: No caching yet (will re-process files)

---

## 🛠️ **Troubleshooting**

### **Build Issues**
```bash
# Make sure you have Rust 1.70+
rustc --version

# Clean build if needed
cargo clean
cargo build --release
```

### **Model Download Issues**
- Requires internet connection
- Check available disk space
- Models stored in `~/.resume-aligner/models/`

### **File Processing Issues**
- Ensure files exist and are readable
- PDF extraction requires valid PDF files
- Text encoding should be UTF-8

---

## 🚀 **Next Steps**

1. **Try the basic analysis** with sample files
2. **Test with your own resume** and job postings
3. **Download a model** (phi-4-mini) for AI features
4. **Explore different output formats** (JSON, detailed)
5. **Check the [TODO.md](TODO.md)** for upcoming features

---

## 💡 **Tips for Best Results**

- **Use relevant job descriptions** that match your career level
- **Try different resume formats** (TXT often works best)
- **Experiment with the `--detailed` flag** for more insights
- **Save results to JSON** for programmatic analysis

---

**Ready to analyze your resume? Start with the Quick Start section above!** 🎯


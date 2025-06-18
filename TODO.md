# 📋 Resume Aligner Development Roadmap

> **Based on Dead Code Analysis - Prioritized Implementation Plan**

This roadmap is derived from analyzing the "dead code" warnings in the current codebase. Each unused method, field, and feature represents a planned implementation that will complete the Resume Aligner's functionality.

---

## 🎯 **Phase 1: Core Infrastructure Completion** 
*Priority: HIGH - Foundation for all other features*

### 📁 **Configuration Management** (`src/config.rs`)
**Dead Code**: `ensure_models_dir()`, `get_model_by_name()`, `list_llm_models()`

**Implementation Tasks:**
- [ ] **Model Directory Management**
  - Implement `ensure_models_dir()` to create model storage directories
  - Add error handling for disk space and permissions
  - Integration with CLI model download commands

- [ ] **Model Lookup & Registry**
  - Complete `get_model_by_name()` for dynamic model selection
  - Implement `list_llm_models()` for CLI model listing
  - Add model validation and compatibility checks

- [ ] **Configuration CLI Commands**
  - Implement `resume-aligner config edit` command
  - Add `resume-aligner config set key=value` functionality
  - Configuration validation and schema checking

**Impact**: Enables proper model management and user configuration

---

## 🎯 **Phase 2: Advanced Text Processing** 
*Priority: HIGH - Core analysis capabilities*

### 📝 **Text Processor Enhancements** (`src/processing/text_processor.rs`)
**Dead Code**: `extract_keywords()`, `remove_pii()`, unused `ProcessedText` fields

**Implementation Tasks:**
- [ ] **Keyword Extraction System**
  - Complete frequency-based keyword extraction
  - Add TF-IDF scoring for keyword importance
  - Integration with ATS matching pipeline
  - Skills vs. general keyword classification

- [ ] **PII Detection & Removal**
  - Implement `remove_pii()` for privacy protection
  - Add phone, email, SSN, address detection
  - Integration with resume anonymization features
  - GDPR compliance utilities

- [ ] **Enhanced Text Analytics**
  - Utilize `original`, `cleaned`, `sentences` fields
  - Add `word_count`, `character_count` for analytics
  - Implement text quality scoring
  - Reading level and complexity analysis

**Impact**: Better content analysis and privacy protection

---

## 🎯 **Phase 3: Advanced Embeddings & Caching** 
*Priority: MEDIUM - Performance optimization*

### 🧠 **Embedding Engine Extensions** (`src/processing/embeddings.rs`)
**Dead Code**: `encode_texts()`, `clear_cache()`, performance tracking fields

**Implementation Tasks:**
- [ ] **Batch Processing**
  - Implement `encode_texts()` for efficient batch embeddings
  - Add `BatchEmbeddingResult` with timing metrics
  - Optimize memory usage for large document sets

- [ ] **Cache Management**
  - Complete `clear_cache()` functionality
  - Add cache statistics (`cache_hits`, `cache_misses`)
  - Implement cache size limits and LRU eviction
  - Cache persistence across sessions

- [ ] **Performance Monitoring**
  - Track `processing_time_ms` for optimization
  - Add `model_name` tracking for multi-model setups
  - Document similarity scoring improvements

**Impact**: Better performance and resource management

---

## 🎯 **Phase 4: Enhanced Input Management** 
*Priority: MEDIUM - User experience*

### 📂 **Input Manager Caching** (`src/input/manager.rs`)
**Dead Code**: `with_cache()`, `clear_cache()`, `cache_size()`

**Implementation Tasks:**
- [ ] **File Caching System**
  - Implement `with_cache()` for extracted text caching
  - Add PDF processing cache to avoid re-extraction
  - Cache invalidation based on file modification time

- [ ] **Cache Management UI**
  - CLI commands for cache inspection (`cache_size()`)
  - Cache clearing functionality (`clear_cache()`)
  - Storage usage reporting

**Impact**: Faster repeated analysis of same documents

---

## 🎯 **Phase 5: Advanced ATS & Analysis** 
*Priority: MEDIUM - Analysis quality*

### 🔍 **ATS Matcher Enhancements** (`src/processing/ats_matcher.rs`)
**Dead Code**: `set_fuzzy_threshold()`, `ats_skill_count` tracking

**Implementation Tasks:**
- [ ] **Dynamic Threshold Tuning**
  - Implement `set_fuzzy_threshold()` for configurable matching
  - A/B testing for optimal threshold values
  - Per-industry threshold optimization

- [ ] **Skills Database Expansion**
  - Track `ats_skill_count` for database size monitoring
  - Add industry-specific skill databases
  - Skills taxonomy and categorization

**Impact**: More accurate and configurable keyword matching

---

## 🎯 **Phase 6: Advanced LLM Features** 
*Priority: HIGH - AI capabilities*

### 🔧 **URGENT: LLM Model Compatibility** 
**Status**: ⚠️ **NEEDS IMMEDIATE ATTENTION**

**Current Issues:**
- [x] ✅ **LLM Parameter Passing**: Fixed - `--llm` parameter now properly passed through pipeline
- [x] ✅ **LLM Engine Initialization**: Fixed - Custom model parameter correctly reaches LLM analyzer
- [ ] ❌ **Model Format Support**: Only Phi-4 format supported, need Llama model loader
- [ ] ❌ **Phi-4-Mini Download Issue**: Missing `model-00001-of-00002.safetensors` file
- [ ] ❌ **Model Type Detection**: Works but needs proper model-specific loaders

**Critical Tasks:**
- [ ] **Add Llama Model Support**
  - Implement `LlamaModel` struct in `src/llm/inference.rs`
  - Add Llama-specific tensor loading (`model.embed_tokens.weight` vs Phi-3 format)
  - Handle different model architectures (`LlamaForCausalLM` vs `Phi3ForCausalLM`)
  - Test with TinyLlama model that's already downloaded

- [ ] **Fix Phi-3-Mini Download**
  - Debug why `model-00001-of-00002.safetensors` is not downloading
  - Check HuggingFace model repository structure
  - Fix model manager to handle multi-part downloads correctly

- [ ] **Model Auto-Detection**
  - Enhance model type detection from config.json
  - Add proper model loader selection (Phi-3 vs Llama vs others)
  - Implement fallback mechanisms for unsupported models

### 🤖 **LLM Inference Extensions** (`src/llm/inference.rs`)
**Dead Code**: `batch_generate()`, `get_config()`, model info methods

**Implementation Tasks:**
- [ ] **Batch Processing**
  - Implement `batch_generate()` for efficient multi-prompt processing
  - Parallel inference for multiple resume sections
  - Cost optimization for API-based models

- [ ] **Model Configuration**
  - Complete `get_config()` and `update_config()` for runtime tuning
  - Temperature, top-p, max_tokens adjustment
  - Model performance monitoring

- [ ] **Model Information System**
  - Implement `get_model_info()` for capability reporting
  - Vocabulary size tracking (`get_vocab_size()`)
  - Model compatibility checking

### 🎨 **Prompt Template System** (`src/llm/prompts.rs`)
**Dead Code**: Multiple specialized prompt templates

**Implementation Tasks:**
- [ ] **Specialized Prompts**
  - `render_resume_improvement()` for targeted suggestions
  - `render_section_analysis()` for detailed section review
  - `render_keyword_optimization()` for ATS improvements

- [ ] **Prompt Engineering**
  - Industry-specific prompt variants
  - Role-level prompt customization
  - A/B testing for prompt effectiveness

### 🧪 **LLM Analyzer Features** (`src/llm/analyzer.rs`)
**Dead Code**: Model management, readiness checking, detailed scoring

**Implementation Tasks:**
- [ ] **Model Management Integration**
  - Implement `get_available_models()` for dynamic model selection
  - `get_current_model()` for status reporting
  - `is_ready()` for health checking

- [ ] **Detailed Scoring System**
  - Utilize `ExactMatch.count` for frequency-based scoring
  - `SectionScore` breakdown (`embedding_score`, `keyword_score`)
  - Granular analysis reporting

**Impact**: Complete AI-powered analysis capabilities

---

## 🎯 **Phase 7: Comprehensive Output System** 
*Priority: HIGH - User experience*

### 📊 **Report Generation** (`src/output/report.rs` & `src/output/formatter.rs`)
**Dead Code**: Complete reporting pipeline, multiple formatters

**Implementation Tasks:**
- [ ] **Report Generation Pipeline**
  - Complete `from_alignment_analysis()` integration
  - All report creation methods (`create_analysis_summary()`, etc.)
  - Rich data structure population

- [ ] **Multi-Format Output**
  - Implement all `OutputFormatter` trait methods
  - `ReportGenerator` with all formatter integration
  - `generate_detailed_console()` for verbose output

- [ ] **File Management**
  - `save_report_to_file()` with proper error handling
  - `suggest_filename()` with intelligent naming
  - Batch export capabilities

**Impact**: Professional, multi-format reporting system

---

## 🎯 **Phase 8: Error Handling & Robustness** 
*Priority: MEDIUM - Production readiness*

### ⚠️ **Error System Completion** (`src/error.rs`)
**Dead Code**: Specialized error variants

**Implementation Tasks:**
- [ ] **Specialized Error Handling**
  - `TextProcessing` errors for content analysis failures
  - `LlmInference` errors for AI processing issues
  - `ModelLoading` errors for model initialization
  - `Network` errors for model download failures

- [ ] **Error Recovery**
  - Graceful degradation strategies
  - Fallback processing modes
  - User-friendly error messages

**Impact**: Robust, production-ready error handling

---

## 🚀 **Implementation Priority Matrix**

### **🔥 Critical Path (Implement First)**
1. **LLM Integration** - Core AI functionality
2. **Output System** - User-facing results
3. **Configuration Management** - Foundation features
4. **Text Processing** - Analysis quality

### **⚡ High Impact, Medium Effort**
1. **Advanced Embeddings** - Performance improvements
2. **Error Handling** - Production readiness
3. **ATS Enhancements** - Analysis accuracy

### **📈 Future Enhancements**
1. **Input Caching** - User experience
2. **Specialized Prompts** - AI fine-tuning
3. **Model Management** - Advanced features

---

## 📅 **Estimated Timeline**

- **Phase 1-2** (Core Infrastructure): 2-3 weeks
- **Phase 3-4** (Performance & UX): 2 weeks  
- **Phase 5-6** (Advanced AI): 3-4 weeks
- **Phase 7-8** (Output & Polish): 2 weeks

**Total Estimated Development**: ~10-12 weeks for complete implementation

---

## 🎁 **Expected Benefits After Completion**

- **🔄 Zero Dead Code**: Clean, fully-utilized codebase
- **🚀 Complete Feature Set**: All planned functionality implemented
- **📊 Rich Analytics**: Comprehensive analysis and reporting
- **🤖 Advanced AI**: Full LLM integration and optimization
- **💼 Production Ready**: Robust error handling and performance
- **👥 User Friendly**: Multiple output formats and easy configuration

---

*This roadmap ensures systematic implementation of all planned features while maintaining code quality and user experience focus.*


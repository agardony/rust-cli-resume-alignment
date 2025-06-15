//! Embeddings generation using Model2Vec

use crate::error::{Result, ResumeAlignerError};
use crate::config::Config;
use crate::processing::document::{DocumentChunk, ProcessedDocument};
use model2vec_rs::model::StaticModel;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub struct EmbeddingEngine {
    model: StaticModel,
    batch_size: usize,
    cache: HashMap<String, Vec<f32>>,
    model_name: String,
}

#[derive(Debug, Clone)]
pub struct EmbeddingResult {
    pub text: String,
    pub embedding: Vec<f32>,
    pub processing_time_ms: u64,
}

#[derive(Debug, Clone)]
pub struct BatchEmbeddingResult {
    pub results: Vec<EmbeddingResult>,
    pub total_processing_time_ms: u64,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

#[derive(Debug, Clone)]
pub struct SimilarityScore {
    pub score: f32,
    pub text1_length: usize,
    pub text2_length: usize,
    pub embedding_dim: usize,
}

impl EmbeddingEngine {
    pub async fn new(model_path: &Path, config: &Config) -> Result<Self> {
        let start_time = Instant::now();
        
        println!("Loading Model2Vec embedding model from: {}", model_path.display());
        
        let model = StaticModel::from_pretrained(
            model_path,
            None, // token
            None, // normalize
            None, // subfolder
        ).map_err(|e| ResumeAlignerError::Embedding(format!("Failed to load model: {}", e)))?;
        
        let load_time = start_time.elapsed();
        println!("Model loaded successfully in {:.2?}", load_time);
        
        Ok(Self {
            model,
            batch_size: config.processing.batch_size,
            cache: HashMap::new(),
            model_name: config.models.default_embedding_model.clone(),
        })
    }
    
    /// Create from default model in config
    pub async fn from_config(config: &Config) -> Result<Self> {
        let model_path = Self::get_model_path(config)?;
        Self::new(&model_path, config).await
    }
    
    /// Get the path to the default embedding model
    fn get_model_path(config: &Config) -> Result<PathBuf> {
        let model_name = &config.models.default_embedding_model;
        
        // Check if it's a local path first
        let local_path = config.models_dir().join(model_name);
        if local_path.exists() {
            return Ok(local_path);
        }
        
        // For now, assume it's a HuggingFace model ID and will be downloaded
        // This would be implemented in Phase 5 with model downloading
        Ok(local_path)
    }
    
    /// Encode multiple texts with batching and caching
    pub fn encode_texts_with_batching(&mut self, texts: &[String]) -> Result<BatchEmbeddingResult> {
        let start_time = Instant::now();
        let mut results = Vec::new();
        let mut cache_hits = 0;
        let mut cache_misses = 0;
        
        // Process texts in batches
        for batch in texts.chunks(self.batch_size) {
            let batch_results = self.process_batch(batch)?;
            for result in batch_results {
                if result.processing_time_ms == 0 {
                    cache_hits += 1;
                } else {
                    cache_misses += 1;
                }
                results.push(result);
            }
        }
        
        let total_time = start_time.elapsed();
        
        Ok(BatchEmbeddingResult {
            results,
            total_processing_time_ms: total_time.as_millis() as u64,
            cache_hits,
            cache_misses,
        })
    }
    
    /// Process a single batch of texts
    fn process_batch(&mut self, texts: &[String]) -> Result<Vec<EmbeddingResult>> {
        let mut results = Vec::new();
        let mut uncached_texts = Vec::new();
        let mut uncached_indices = Vec::new();
        
        // Check cache first
        for (i, text) in texts.iter().enumerate() {
            if let Some(cached_embedding) = self.cache.get(text) {
                results.push(EmbeddingResult {
                    text: text.clone(),
                    embedding: cached_embedding.clone(),
                    processing_time_ms: 0, // Cache hit
                });
            } else {
                uncached_texts.push(text.clone());
                uncached_indices.push(i);
            }
        }
        
        // Process uncached texts
        if !uncached_texts.is_empty() {
            let start_time = Instant::now();
            let embeddings = self.model.encode(&uncached_texts);
            let processing_time = start_time.elapsed().as_millis() as u64;
            
            for (text, embedding) in uncached_texts.iter().zip(embeddings.iter()) {
                // Cache the result
                self.cache.insert(text.clone(), embedding.clone());
                
                results.push(EmbeddingResult {
                    text: text.clone(),
                    embedding: embedding.clone(),
                    processing_time_ms: processing_time / uncached_texts.len() as u64,
                });
            }
        }
        
        // Sort results to maintain original order
        results.sort_by_key(|r| texts.iter().position(|t| t == &r.text).unwrap_or(0));
        
        Ok(results)
    }
    
    /// Encode texts without caching (for simple use cases)
    pub fn encode_texts(&self, texts: &[String]) -> Vec<Vec<f32>> {
        self.model.encode(texts)
    }
    
    /// Encode a single text with caching
    pub fn encode_single_cached(&mut self, text: &str) -> Result<EmbeddingResult> {
        if let Some(cached_embedding) = self.cache.get(text) {
            return Ok(EmbeddingResult {
                text: text.to_string(),
                embedding: cached_embedding.clone(),
                processing_time_ms: 0,
            });
        }
        
        let start_time = Instant::now();
        let embedding = self.model.encode_single(text);
        let processing_time = start_time.elapsed().as_millis() as u64;
        
        // Cache the result
        self.cache.insert(text.to_string(), embedding.clone());
        
        Ok(EmbeddingResult {
            text: text.to_string(),
            embedding,
            processing_time_ms: processing_time,
        })
    }
    
    /// Encode a single text without caching
    pub fn encode_single(&self, text: &str) -> Vec<f32> {
        self.model.encode_single(text)
    }
    
    /// Process document chunks and generate embeddings
    pub fn process_document_chunks(&mut self, chunks: &[DocumentChunk]) -> Result<Vec<EmbeddingResult>> {
        let texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
        let batch_result = self.encode_texts_with_batching(&texts)?;
        Ok(batch_result.results)
    }
    
    /// Process entire document with sections
    pub fn process_document(&mut self, doc: &ProcessedDocument) -> Result<DocumentEmbeddings> {
        let start_time = Instant::now();
        
        // Process chunks
        let chunk_embeddings = self.process_document_chunks(&doc.chunks)?;
        
        // Process sections separately
        let section_texts: Vec<String> = doc.sections.iter().map(|s| s.content.clone()).collect();
        let section_batch_result = self.encode_texts_with_batching(&section_texts)?;
        
        // Process full document
        let full_doc_result = self.encode_single_cached(&doc.original.content)?;
        
        let total_time = start_time.elapsed();
        
        Ok(DocumentEmbeddings {
            document_path: doc.original.file_path.clone(),
            full_document: full_doc_result,
            chunks: chunk_embeddings,
            sections: section_batch_result.results,
            processing_time_ms: total_time.as_millis() as u64,
            model_name: self.model_name.clone(),
        })
    }
    
    /// Calculate cosine similarity between two embeddings
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<SimilarityScore> {
        if a.len() != b.len() {
            return Err(ResumeAlignerError::Processing(
                format!("Embedding dimensions don't match: {} vs {}", a.len(), b.len())
            ));
        }
        
        if a.is_empty() || b.is_empty() {
            return Ok(SimilarityScore {
                score: 0.0,
                text1_length: 0,
                text2_length: 0,
                embedding_dim: 0,
            });
        }
        
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        let score = if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        };
        
        Ok(SimilarityScore {
            score,
            text1_length: a.len(),
            text2_length: b.len(),
            embedding_dim: a.len(),
        })
    }
    
    /// Calculate similarity between document sections
    pub fn calculate_document_similarity(
        &self,
        doc1_embeddings: &DocumentEmbeddings,
        doc2_embeddings: &DocumentEmbeddings,
    ) -> Result<DocumentSimilarity> {
        // Overall document similarity
        let overall_similarity = Self::cosine_similarity(
            &doc1_embeddings.full_document.embedding,
            &doc2_embeddings.full_document.embedding,
        )?;
        
        // Section-wise similarities
        let mut section_similarities = Vec::new();
        for section1 in &doc1_embeddings.sections {
            for section2 in &doc2_embeddings.sections {
                let similarity = Self::cosine_similarity(
                    &section1.embedding,
                    &section2.embedding,
                )?;
                
                section_similarities.push(SectionSimilarity {
                    section1_text: section1.text.clone(),
                    section2_text: section2.text.clone(),
                    similarity,
                });
            }
        }
        
        // Chunk-wise similarities (average)
        let chunk_similarities: Result<Vec<_>> = doc1_embeddings.chunks.iter()
            .zip(doc2_embeddings.chunks.iter())
            .map(|(c1, c2)| Self::cosine_similarity(&c1.embedding, &c2.embedding))
            .collect();
        
        let chunk_similarities = chunk_similarities?;
        let avg_chunk_similarity = if chunk_similarities.is_empty() {
            SimilarityScore {
                score: 0.0,
                text1_length: 0,
                text2_length: 0,
                embedding_dim: 0,
            }
        } else {
            let avg_score = chunk_similarities.iter().map(|s| s.score).sum::<f32>() / chunk_similarities.len() as f32;
            SimilarityScore {
                score: avg_score,
                text1_length: chunk_similarities[0].text1_length,
                text2_length: chunk_similarities[0].text2_length,
                embedding_dim: chunk_similarities[0].embedding_dim,
            }
        };
        
        Ok(DocumentSimilarity {
            overall_similarity,
            section_similarities,
            average_chunk_similarity: avg_chunk_similarity,
            doc1_path: doc1_embeddings.document_path.clone(),
            doc2_path: doc2_embeddings.document_path.clone(),
        })
    }
    
    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        CacheStats {
            cache_size: self.cache.len(),
            model_name: self.model_name.clone(),
            batch_size: self.batch_size,
        }
    }
    
    /// Clear the embedding cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

#[derive(Debug, Clone)]
pub struct DocumentEmbeddings {
    pub document_path: String,
    pub full_document: EmbeddingResult,
    pub chunks: Vec<EmbeddingResult>,
    pub sections: Vec<EmbeddingResult>,
    pub processing_time_ms: u64,
    pub model_name: String,
}

#[derive(Debug, Clone)]
pub struct SectionSimilarity {
    pub section1_text: String,
    pub section2_text: String,
    pub similarity: SimilarityScore,
}

#[derive(Debug, Clone)]
pub struct DocumentSimilarity {
    pub overall_similarity: SimilarityScore,
    pub section_similarities: Vec<SectionSimilarity>,
    pub average_chunk_similarity: SimilarityScore,
    pub doc1_path: String,
    pub doc2_path: String,
}

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub cache_size: usize,
    pub model_name: String,
    pub batch_size: usize,
}


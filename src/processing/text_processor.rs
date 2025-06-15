//! Text processing and normalization

use crate::error::Result;
use regex::Regex;
use std::collections::HashSet;
use unicode_segmentation::UnicodeSegmentation;

pub struct TextProcessor {
    stop_words: HashSet<String>,
    email_regex: Regex,
    phone_regex: Regex,
    url_regex: Regex,
}

#[derive(Debug, Clone)]
pub struct ProcessedText {
    pub original: String,
    pub cleaned: String,
    pub tokens: Vec<String>,
    pub sentences: Vec<String>,
    pub word_count: usize,
    pub character_count: usize,
}

impl Default for TextProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl TextProcessor {
    pub fn new() -> Self {
        let stop_words = Self::create_stop_words();
        
        let email_regex = Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
            .expect("Invalid email regex");
        
        let phone_regex = Regex::new(r"\b(?:\+?1[-. ]?)?\(?[0-9]{3}\)?[-. ]?[0-9]{3}[-. ]?[0-9]{4}\b")
            .expect("Invalid phone regex");
        
        let url_regex = Regex::new(r"https?://[^\s]+")
            .expect("Invalid URL regex");
        
        Self {
            stop_words,
            email_regex,
            phone_regex,
            url_regex,
        }
    }

    /// Process text with cleaning, tokenization, and normalization
    pub fn process(&self, text: &str) -> Result<ProcessedText> {
        let original = text.to_string();
        
        // Clean the text
        let cleaned = self.clean_text(text)?;
        
        // Tokenize into words
        let tokens = self.tokenize(&cleaned);
        
        // Split into sentences
        let sentences = self.split_sentences(&cleaned);
        
        let word_count = tokens.len();
        let character_count = cleaned.chars().count();
        
        Ok(ProcessedText {
            original,
            cleaned,
            tokens,
            sentences,
            word_count,
            character_count,
        })
    }

    /// Clean and normalize text
    pub fn clean_text(&self, text: &str) -> Result<String> {
        let mut cleaned = text.to_string();
        
        // Remove URLs
        cleaned = self.url_regex.replace_all(&cleaned, "").to_string();
        
        // Replace emails with placeholder
        cleaned = self.email_regex.replace_all(&cleaned, "[EMAIL]").to_string();
        
        // Replace phone numbers with placeholder
        cleaned = self.phone_regex.replace_all(&cleaned, "[PHONE]").to_string();
        
        // Normalize whitespace
        cleaned = self.normalize_whitespace(&cleaned);
        
        // Remove excessive punctuation
        cleaned = self.clean_punctuation(&cleaned);
        
        // Normalize unicode
        cleaned = self.normalize_unicode(&cleaned);
        
        Ok(cleaned)
    }

    /// Tokenize text into words using Unicode segmentation
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        
        for word in text.unicode_words() {
            let normalized = word.to_lowercase();
            
            // Skip stop words and very short words
            if !self.stop_words.contains(&normalized) && normalized.len() > 1 {
                // Only include alphabetic tokens and numbers
                if normalized.chars().any(|c| c.is_alphabetic()) {
                    tokens.push(normalized);
                }
            }
        }
        
        tokens
    }

    /// Split text into sentences
    pub fn split_sentences(&self, text: &str) -> Vec<String> {
        text.unicode_sentences()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// Extract keywords from text (frequency-based)
    pub fn extract_keywords(&self, text: &str, max_keywords: usize) -> Result<Vec<String>> {
        let processed = self.process(text)?;
        let mut word_freq = std::collections::HashMap::new();
        
        // Count word frequencies
        for token in &processed.tokens {
            if token.len() > 2 { // Only consider words longer than 2 characters
                *word_freq.entry(token.clone()).or_insert(0) += 1;
            }
        }
        
        // Sort by frequency and take top keywords
        let mut keywords: Vec<(String, usize)> = word_freq.into_iter().collect();
        keywords.sort_by(|a, b| b.1.cmp(&a.1));
        
        Ok(keywords
            .into_iter()
            .take(max_keywords)
            .map(|(word, _)| word)
            .collect())
    }

    /// Normalize whitespace (collapse multiple spaces, remove tabs/newlines)
    fn normalize_whitespace(&self, text: &str) -> String {
        let whitespace_regex = Regex::new(r"\s+").expect("Invalid whitespace regex");
        whitespace_regex.replace_all(text, " ").trim().to_string()
    }

    /// Clean excessive punctuation
    fn clean_punctuation(&self, text: &str) -> String {
        let punct_regex = Regex::new(r"[.!?]{2,}").expect("Invalid punctuation regex");
        let multiple_punct = Regex::new(r"[.,;:!?]{3,}").expect("Invalid multiple punctuation regex");
        
        let mut cleaned = punct_regex.replace_all(text, ".").to_string();
        cleaned = multiple_punct.replace_all(&cleaned, ".").to_string();
        
        cleaned
    }

    /// Normalize Unicode characters
    fn normalize_unicode(&self, text: &str) -> String {
        // Basic Unicode normalization - could be enhanced with unicode-normalization crate
        text.chars()
            .map(|c| match c {
                '\u{2018}' | '\u{2019}' => '\'', // Smart quotes to regular quotes
                '\u{201C}' | '\u{201D}' => '"',   // Smart double quotes
                '\u{2013}' | '\u{2014}' => '-',   // En dash, em dash to hyphen
                '\u{2026}' => '.',                // Ellipsis to period
                _ => c,
            })
            .collect()
    }

    /// Create set of common English stop words
    fn create_stop_words() -> HashSet<String> {
        let stop_words = [
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
            "to", "was", "will", "with", "the", "this", "but", "they", "have",
            "had", "what", "said", "each", "which", "she", "do", "how", "their",
            "if", "up", "out", "many", "then", "them", "these", "so", "some",
            "her", "would", "make", "like", "into", "him", "time", "two", "more",
            "go", "no", "way", "could", "my", "than", "first", "been", "call",
            "who", "oil", "sit", "now", "find", "down", "day", "did", "get",
            "come", "made", "may", "part", "over", "new", "sound", "take",
            "only", "little", "work", "know", "place", "year", "live", "me",
            "back", "give", "most", "very", "after", "thing", "our", "just",
            "name", "good", "sentence", "man", "think", "say", "great", "where",
            "help", "through", "much", "before", "line", "right", "too", "mean",
            "old", "any", "same", "tell", "boy", "follow", "came", "want",
            "show", "also", "around", "form", "three", "small", "set", "put",
            "end", "why", "again", "turn", "here", "off", "went", "old", "number",
            "great", "tell", "men", "say", "small", "every", "found", "still",
            "between", "mane", "should", "home", "big", "give", "air", "line",
            "set", "own", "under", "read", "last", "never", "us", "left",
            "end", "along", "while", "might", "next", "sound", "below", "saw",
            "something", "thought", "both", "few", "those", "always", "looked",
            "show", "large", "often", "together", "asked", "house", "don", "world",
            "going", "want", "school", "important", "until", "form", "food",
            "keep", "children", "feet", "land", "side", "without", "boy", "once",
            "animal", "life", "enough", "took", "sometimes", "four", "head",
            "above", "kind", "began", "almost", "live", "page", "got", "earth",
            "need", "far", "hand", "high", "year", "mother", "light", "country",
            "father", "let", "night", "picture", "being", "study", "second",
            "soon", "story", "since", "white", "ever", "paper", "hard", "near",
            "sentence", "better", "best", "across", "during", "today", "however",
            "sure", "knew", "it's", "try", "told", "young", "sun", "thing",
            "whole", "hear", "example", "heard", "several", "change", "answer",
            "room", "sea", "against", "top", "turned", "learn", "point", "city",
            "play", "toward", "five", "himself", "usually", "money", "seen",
            "didn't", "car", "morning", "i'm", "body", "upon", "family", "later",
            "turn", "move", "face", "door", "cut", "done", "group", "true",
            "leave", "color", "red", "friend", "pretty", "eat", "front", "feel",
            "fact", "hand", "week", "eye", "been", "word", "great", "such",
            "make", "shall", "week", "case", "government", "company", "where",
            "system", "each", "right", "program", "hear", "question", "during",
            "work", "play", "government", "run", "small", "number", "off",
            "always", "move", "like", "night", "live", "mr", "point", "believe",
            "hold", "today", "bring", "happen", "next", "without", "before",
            "large", "all", "million", "must", "home", "under", "water", "room",
            "write", "mother", "area", "national", "money", "story", "young",
            "fact", "month", "different", "lot", "right", "study", "book",
            "eye", "job", "word", "though", "business", "issue", "side",
            "kind", "four", "head", "far", "black", "long", "both", "little",
            "house", "yes", "after", "since", "long", "provide", "service",
            "around", "friend", "important", "father", "sit", "away", "until",
            "power", "hour", "game", "often", "yet", "line", "political", "end",
            "among", "ever", "stand", "bad", "lose", "however", "member",
            "pay", "law", "meet", "car", "city", "almost", "include", "continue",
            "set", "later", "community", "much", "name", "five", "once", "white",
            "least", "president", "learn", "real", "change", "team", "minute",
            "best", "several", "idea", "kid", "body", "information", "back",
            "parent", "face", "others", "level", "office", "door", "health",
            "person", "art", "war", "history", "party", "within", "grow",
            "result", "open", "morning", "walk", "reason", "low", "win", "research",
            "girl", "guy", "early", "food", "before", "moment", "himself",
            "air", "teacher", "force", "offer"
        ];
        
        stop_words.iter().map(|&s| s.to_string()).collect()
    }

    /// Calculate text similarity using Jaccard similarity on tokens
    pub fn text_similarity(&self, text1: &str, text2: &str) -> Result<f32> {
        let processed1 = self.process(text1)?;
        let processed2 = self.process(text2)?;
        
        let set1: HashSet<&String> = processed1.tokens.iter().collect();
        let set2: HashSet<&String> = processed2.tokens.iter().collect();
        
        let intersection = set1.intersection(&set2).count();
        let union = set1.union(&set2).count();
        
        if union == 0 {
            Ok(0.0)
        } else {
            Ok(intersection as f32 / union as f32)
        }
    }

    /// Remove personal identifiable information
    pub fn remove_pii(&self, text: &str) -> String {
        let mut cleaned = text.to_string();
        
        // Remove emails
        cleaned = self.email_regex.replace_all(&cleaned, "[EMAIL REMOVED]").to_string();
        
        // Remove phone numbers
        cleaned = self.phone_regex.replace_all(&cleaned, "[PHONE REMOVED]").to_string();
        
        // Remove potential SSNs (basic pattern)
        let ssn_regex = Regex::new(r"\b\d{3}-\d{2}-\d{4}\b")
            .expect("Invalid SSN regex");
        cleaned = ssn_regex.replace_all(&cleaned, "[SSN REMOVED]").to_string();
        
        cleaned
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_processing() {
        let processor = TextProcessor::new();
        let text = "Hello, world! This is a test document with email@example.com and phone (555) 123-4567.";
        
        let result = processor.process(text).unwrap();
        
        assert!(!result.cleaned.is_empty());
        assert!(result.tokens.len() > 0);
        assert!(result.word_count > 0);
        assert!(result.sentences.len() > 0);
    }

    #[test]
    fn test_tokenization() {
        let processor = TextProcessor::new();
        let text = "Rust programming language is awesome!";
        
        let tokens = processor.tokenize(text);
        
        assert!(tokens.contains(&"rust".to_string()));
        assert!(tokens.contains(&"programming".to_string()));
        assert!(tokens.contains(&"language".to_string()));
        assert!(tokens.contains(&"awesome".to_string()));
        
        // Stop words should be filtered out
        assert!(!tokens.contains(&"is".to_string()));
    }

    #[test]
    fn test_text_cleaning() {
        let processor = TextProcessor::new();
        let text = "Contact me at john.doe@email.com or call (555) 123-4567!!!";
        
        let cleaned = processor.clean_text(text).unwrap();
        
        assert!(cleaned.contains("[EMAIL]"));
        assert!(cleaned.contains("[PHONE]"));
        assert!(!cleaned.contains("!!!"));
    }

    #[test]
    fn test_keyword_extraction() {
        let processor = TextProcessor::new();
        let text = "Rust Rust programming language. Rust is memory safe. Programming with Rust is fun.";
        
        let keywords = processor.extract_keywords(text, 5).unwrap();
        
        assert!(keywords.len() <= 5);
        assert!(keywords.contains(&"rust".to_string()));
        assert!(keywords.contains(&"programming".to_string()));
    }

    #[test]
    fn test_text_similarity() {
        let processor = TextProcessor::new();
        let text1 = "Rust programming language";
        let text2 = "Programming in Rust language";
        
        let similarity = processor.text_similarity(text1, text2).unwrap();
        
        assert!(similarity > 0.0);
        assert!(similarity <= 1.0);
    }

    #[test]
    fn test_pii_removal() {
        let processor = TextProcessor::new();
        let text = "Contact John at john.doe@company.com or (555) 123-4567. SSN: 123-45-6789";
        
        let cleaned = processor.remove_pii(text);
        
        assert!(!cleaned.contains("john.doe@company.com"));
        assert!(!cleaned.contains("(555) 123-4567"));
        assert!(!cleaned.contains("123-45-6789"));
    }
}

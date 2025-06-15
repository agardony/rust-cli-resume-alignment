//! File type detection

#[derive(Debug, Clone, PartialEq)]
pub enum FileType {
    Pdf,
    Text,
    Markdown,
    Unknown,
}

impl FileType {
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "pdf" => FileType::Pdf,
            "txt" => FileType::Text,
            "md" | "markdown" => FileType::Markdown,
            _ => FileType::Unknown,
        }
    }
}


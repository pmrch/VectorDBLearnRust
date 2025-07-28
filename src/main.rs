pub mod custom_types;
pub mod llm;
pub mod memory_tools;
pub mod store;
pub mod summary;
pub mod utils;


use lm_studio_api_extended::{embedding::*};


#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut embedder = Embedding::new(Some("http://10.7.0.10:1234/v1/embeddings".to_string())); // Uses default localhost:1234
    let req = EmbeddingRequest {
        model: EmbeddingModel::AllMiniLmL6,
        input: vec!["Rust is magic.".to_string()],
        encoding_format: Some("float".to_string()),
    };

    let response = embedder.embed(req).await?;
    println!("{:#?}", response.len());

    Ok(())
}
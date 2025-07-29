use std::collections::HashMap;
use std::{usize};
use anyhow::Ok;
use lm_studio_api_extended::embedding::embedding::EmbeddingResult;
use log::*;

use lm_studio_api_extended::embedding::*;

use crate::memory_tools::{get_similarities};
use crate::summary::generate_summary;
use crate::utils::{text_to_vec};


pub async fn add_memory(
    input: &str, memories: &mut HashMap<String, Vec<f32>>, 
    memories_text: &mut HashMap<String, String>, count: &mut usize, 
    embedder: &mut Embedding
) -> Result<(), Box<dyn std::error::Error>> {

    let summary = generate_summary(input).await?;
    let input_vectors: Vec<String> = text_to_vec(&summary);
    info!("Converted input to vector of {} tokens", input_vectors.len());

    let req = EmbeddingRequest {
        model: EmbeddingModel::AllMiniLmL6,
        input: input_vectors.clone(),
        encoding_format: Some("float".to_string()),
    };

    let response = embedder.embed(req).await?;

    match response {
        EmbeddingResult::Single(vec) => {
            let key: String = format!("User_Info_{}", *count);
            memories.insert(key.clone(), vec);
            memories_text.insert(key, input_vectors[0].clone());
            *count += 1;
        }
        EmbeddingResult::Multi(vvec) => {
            for (i, emb) in vvec.iter().enumerate() {
                let key: String = format!("User_Info_{}", *count + i);
                memories.insert(key.clone(), emb.clone());
                memories_text.insert(key, input_vectors[i].clone());
            }
            *count += vvec.len();
        }
    }

    std::result::Result::Ok(())
}

pub async fn retrieve_memory(
    memories: HashMap<String, Vec<f32>>, memories_text: HashMap<String, String>, 
    query: &str, top_k: usize, embedder: &mut Embedding
    ) -> anyhow::Result<Vec<String>> {
    // Here we convert our string input to a vector so that it's,
    // in the correct format for the OpanAI Rust API's embedding
    // call.
    let input_vector = text_to_vec(query);
    let req = EmbeddingRequest {
        model: EmbeddingModel::AllMiniLmL6,
        input: input_vector,
        encoding_format: Some("float".to_string())
    };

    let embeddings = embedder.embed(req).await?;
    let mut similarity: Vec<(String, f32)> = Vec::new();
    let mut similarities: Vec<Vec<(String, f32)>> = Vec::new();

    match embeddings {
        EmbeddingResult::Single(vec) => {
            similarity = get_similarities(&vec, &memories, top_k);
        },
        EmbeddingResult::Multi(vvec) => {
            for vec in vvec {
                let sim: Vec<(String, f32)> = get_similarities(&vec, &memories, top_k);
                similarities.push(sim);
            }
        }
    };

    let mut result_keys: Vec<String> = Vec::new();
    let mut results: Vec<String> = Vec::new();

    if similarity.is_empty() {
        for sim in similarities {
            for s in sim {
                result_keys.push(s.0);
            }
        }
    } else {
        result_keys.push(similarity[0].0.clone());
    }
    
    match result_keys.len() {
        0 => Err(anyhow::anyhow!("result_keys was empty")),
        1 => {
            if let Some(key) = result_keys.get(0) {
                results.push(memories_text[key].clone());
            }
            Ok(results)
        },
        _ => {
            for string in result_keys {
                results.push(memories_text[&string].clone());
            }
            Ok(results)
        }
    }
}
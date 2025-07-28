use std::collections::HashMap;
use std::{usize};
use anyhow::Ok;
use lm_studio_api_extended::embedding::embedding::EmbeddingResult;
use log::*;

use ndarray::{ Array1, Array2, ShapeBuilder };
use lm_studio_api_extended::embedding::*;

use crate::memory_tools::{cosine_similarity, make_vector, vec_to_array2};
use crate::utils::{text_to_vec, EMBED_MODEL};


pub async fn add_memory(input: &str, memories: &mut HashMap<String, Vec<f32>>, memories_text: &mut HashMap<String, String>, count: &mut usize, embedder: &mut Embedding) -> anyhow::Result<()> {
    let input_vectors: Vec<String> = text_to_vec(input);
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

    Ok(())
}

pub async fn retrieve_memory(memories: HashMap<String, Vec<f32>>, memories_text: HashMap<String, String>, query: &str, top_k: usize, embedder: Embedding) -> anyhow::Result<Vec<String>> {
    // Here we convert our string input to a vector so that it's,
    // in the correct format for the OpanAI Rust API's embedding
    // call.
    let input_vector = text_to_vec(query);
    let req = EmbeddingRequest {
        model: EmbeddingModel::AllMiniLmL6,
        input: input_vector,
        encoding_format: Some("float".to_string())
    };

    let mut embedding_as_array1: Array1<f32> = Array1::from_vec(vec![]);
    let mut embeddings_as_array2: Array2<f32> = Array2::zeros((0, 0));

    let embeddings = embedder.embed(req).await?;
    match embeddings {
        EmbeddingResult::Single(vec) => {
            embedding_as_array1 = Array1::from_vec(vec);
        },
        EmbeddingResult::Multi(vvec) => {
            embeddings_as_array2 = vec_to_array2(vvec);
        }
    }
    
    // Here we create a Vector that will store the results.
    // To be more specific, we will put the correct values from memories_text
    let mut result: Vec<String> = Vec::new();

    // Here we loop through memories and calculate cosine similarity, and store
    // both the index and the similarity value in sims_and_indexes
    for (i, mem) in memories.iter().enumerate() {
        let mem_vec_as_array: Array1<f32> = Array1::from_vec(mem.1.clone());
        let embeddings_as_array: Array1<f32> = make_vector(input);

        let similarity = cosine_similarity(&actual_embed_as_array, &mem_vec_as_array);
        sims_and_indexes.push((i, similarity));
    }

    // Here we filter out NaN values, since sort_by() doesn't play nice
    // with NaN comparisons
    sims_and_indexes.retain(|x| !x.1.is_nan());
    sims_and_indexes.sort_by(
        |a, b| b.1.partial_cmp(&a.1).unwrap()
    );

    if sims_and_indexes.len() >= top_k {
        for item in 0..top_k {
            let index = sims_and_indexes[item].0;
            let val= memories_text[keys[index]].clone();

            result.push(val);
        }
    } else {
        for item in sims_and_indexes {
            let index = item.0;
            let val = memories_text[keys[index]].clone();

            result.push(val);
        }
    }

    if result.is_empty() { None }
    else { Some(result) }

    Ok(())
}
use std::collections::HashMap;
use std::usize;
use log::*;

use ndarray::{Array1};
use openai_api_rust::embeddings::{*};

use crate::memory::{cosine_similarity, generate_embeddings};
use crate::utils::text_to_vec;


pub fn add_memory(input: &str, memories: &mut HashMap<String, Vec<f64>>, count: &mut usize, memories_text: &mut HashMap<String, String>) -> () {
    let input_vectors: Vec<String> = text_to_vec(input);
    info!("Converted input to vector of {} tokens", input_vectors.len());

    let my_embeddings: Option<Embeddings> = generate_embeddings(input_vectors.clone());
    let Some(emb) = my_embeddings else { 
        error!("Failed to generate embeddings from input text.");
        return 
    };

    let Some(data_vec) = emb.data else { 
        error!("Embeddings returned, but no data found inside.");
        return 
    };

    if data_vec.is_empty() {
        warn!("Embedding data was empty. Nothing to add to memory.");
        return;
    }

    for (i, val) in data_vec.iter().enumerate() {
        let Some(data) = val.embedding.as_ref() else { 
            warn!("Missing embedding vector at index {}, skipping.", i);
            continue 
        };

        let key = format!("User Info{}", *count + i);

        memories.insert(key.to_string(), data.to_vec());
        info!("Added memory at key '{}', vector length {}", key, data.len());

        memories_text.insert(key.to_string(), input.to_string());
        info!("Added memory text at key '{}', string length {}", key, input.to_string().len());
    }

    *count += data_vec.len();
    info!("Updated memory count to {}", *count);
}

pub fn retrieve_memory(memories: HashMap<String, Vec<f64>>, memories_text: HashMap<String, String>, query: &str, top_k: usize) -> Option<Vec<String>> {
    // Here we convert our string input to a vector so that it's,
    // in the correct format for the OpanAI Rust API's embedding
    // call.
    let input_vector = text_to_vec(query);
    
    // This is an instance of the Embeddings struct that has 
    // several attributes, one of which is data, the one we need.
    let embeddings = generate_embeddings(input_vector.to_vec())?;

    // This is a vector of struct EmbeddingsData, which has 
    // several attributes, one of which is embedding, which is
    // the value we need.
    let embedding_data = embeddings.data?;

    // This will store the extracted floating point values,
    // creating a raw vector, the exact one we need.
    let actual_embedding = embedding_data[0].embedding.as_ref()?;

    // Here we convert our Vec to an Array so we can perform
    // cosine similarity on it.
    let actual_embed_as_array: Array1<f64> = Array1::from_vec(actual_embedding.to_vec());

    // Here we define keys, to which we can later reference. these are the 
    // keys of memory_text
    let keys: Vec<&String> = memories_text.keys().collect();

    // Here we create a Vec of a tuple which corresponds to the
    // attributes of HashMap, and an extra, the similarity itslef
    // (index of memory, similarity).
    let mut sims_and_indexes: Vec<(usize, f64)> = Vec::new();

    // Here we create a Vector that will store the results.
    // To be more specific, we will put the correct values from memories_text
    let mut result: Vec<String> = Vec::new();

    // Here we loop through memories and calculate cosine similarity, and store
    // both the index and the similarity value in sims_and_indexes
    for (i, mem) in memories.iter().enumerate() {
        let mem_vec_as_array: Array1<f64> = Array1::from_vec(mem.1.clone());
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
}
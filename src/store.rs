use std::collections::HashMap;

use openai_api_rust::embeddings::*;

use crate::memory::generate_embeddings;
use crate::utils::text_to_vec;


pub fn add_memory(input: &str, memories: &mut HashMap<String, EmbeddingData>, count: &mut usize) -> () {
    let mut local_count = *count;
    let input_vectors = text_to_vec(input);
    let my_embeddings = generate_embeddings(input_vectors.clone());

    if my_embeddings.is_none() {
        return;
    } else {
        let base_key: String = "User_Info".to_string();
        let value = my_embeddings.unwrap().data.unwrap();

        for (i, val) in value.iter().enumerate() {
            let key = format!("User Info{}", local_count + i);
            memories.insert(key.to_string(), val);

            local_count += 1;
        }
    }
}
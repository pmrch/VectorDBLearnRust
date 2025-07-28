use std::collections::HashMap;

pub fn cosine_similarity(a: &Vec<f32>, b: &Vec<f32>) -> f32 {   
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

pub fn get_similarities(query: &Vec<f32>, memories: &HashMap<String, Vec<f32>>, top_k: usize) -> Vec<(String, f32)> {
    let mut sims: Vec<(String, f32)> = Vec::new();

    for (key, value) in memories {
        let sim = cosine_similarity(query, value);
        sims.push((key.clone(), sim));
    }

    sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    if sims.len() > top_k { return sims; }
    else { 
        let mut result: Vec<(String, f32)> = Vec::new();
        for i in 0..top_k { result.push(sims[i].clone()); }
        return result;
    }
}
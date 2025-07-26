use ndarray::{self, Array1, ArrayD, Axis};
use ndarray_linalg::{self, Norm};

use crate::utils::{ load_environment, get_openai, EMBED_MODEL };


pub fn make_vector(input: &ArrayD<f64>) -> Option<Array1<f64>> {
    match input.ndim() {
        1 => Some(input.clone().into_dimensionality::<ndarray::Ix1>().unwrap()),
        2 => {
            input.mean_axis(Axis(0))
                .and_then(|arr_dyn| arr_dyn.into_dimensionality::<ndarray::Ix1>().ok())
        }
        _ => {
            log::error!("Can't cleanly convert to vector");
            None
        }
    }
}

pub fn cosine_similarity(a: &Array1<f64>, b: &Array1<f64>) -> f64 {   
    let dot_product: f64 = a.dot(b);
    let norm_a: f64 = a.norm_l2();
    let norm_b: f64 = b.norm_l2();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

pub fn generate_embeddings() {
    
}
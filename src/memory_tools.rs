use anyhow::Ok;
use lm_studio_api_extended::embedding::embedding::EmbeddingResult;
use ndarray::{self, stack, Array, Array1, Array2, ArrayD, ArrayView1, Axis};
use ndarray_linalg::{self, Norm, Scalar};


pub fn make_vector(input: &ArrayD<f32>) -> Option<Array1<f32>> {
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

pub fn normalize_row(row: ArrayView1<f32>) -> Vec<f32> {
    let norm = row.dot(&row).sqrt();

    if norm == 0.0 {
        row.to_vec()
    } else {
        row.iter().map(|x| x / norm).collect()
    }
}

pub fn normalize_rows(arr: &Array2<f32>) -> Array2<f32> {
    let rows = arr.nrows();
    let cols = arr.ncols();
    let mut out = Array2::<f32>::zeros((rows, cols));

    for (i, row) in arr.outer_iter().enumerate() {
        let normalized = normalize_row(row);
        for (j, val) in normalized.iter().enumerate() {
            out[[i, j]] = *val;
        }
    }
    
    out
}

pub fn cosine_similarity(a: &Array2<f32>, b: &Array2<f32>) -> Array1<f32> {   
    let norm_a: f32 = a.norm_l2();
    let norm_b: f32 = b.norm_l2();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

pub fn to_array2(embedding: EmbeddingResult) -> anyhow::Result<Array2<f32>> {
    match embedding {
        EmbeddingResult::Single(vec) => {
            let ret = Array2::from_shape_vec((1, vec.len()), vec)?;
            Ok(ret)
        },
        EmbeddingResult::Multi(vvec) => {
            let rows = vvec.len();
            let cols = vvec[0].len();
            let flat: Vec<f32> = vvec.into_iter().flatten().collect();

            let ret = Array2::from_shape_vec((rows, cols), flat)?;
            Ok(ret)
        }
    }
}
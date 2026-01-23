use std::iter::Once;

use rand::Rng;




#[derive(Debug, Clone)]
pub struct WeightAndBiasMatrix {
    inner: Vec<(Vec<f32>, f32)>,
    width: usize,
    height: usize
}


impl WeightAndBiasMatrix {
    pub fn new(width: usize, height: usize) -> Self {
        return Self {
            inner: vec![(vec![0.0; width], 0.0); height],
            width,
            height
        }
    }


    pub fn from_nested_vec(weights: Vec<Vec<f32>>, biases: Vec<f32>) -> Option<Self> {
        let height = weights.len();
        let mut width: Option<usize> = None;

        let mut inner: Vec<(Vec<f32>, f32)> = Vec::new();
        for (row, bias) in weights.into_iter().zip(biases) {
            match width {
                None => {
                    width = Some(row.len());
                },
                Some(width_inner) => {
                    if width_inner != row.len() {
                        return None
                    }
                    inner.push((row, bias))
                }
            }
        }

        return Some(Self { inner: inner, width: width?, height })
    }


    pub fn width(&self) -> usize { self.width }
    pub fn height(&self) -> usize { self.height }

    pub fn size(&self) -> usize { (self.width+1) * self.height }
    pub fn size_weights(&self) -> usize { self.width * self.height }
    pub fn size_biases(&self) -> usize { self.height }

    pub fn flat_weights_and_biases(&self) -> Vec<f32> {
        self.inner.iter().map(|(row, bias)| {
            row.iter().cloned().chain(std::iter::once(*bias)).collect::<Vec<f32>>()
        }).flatten().collect()
    }

    pub fn weights(&self) -> Vec<Vec<f32>> {
        self.inner.iter().map(|(row, _)| row.clone()).collect()
    }

    pub fn flat_weights(&self) -> Vec<f32> {
        self.inner.iter().map(|(row, _)| row.clone()).flatten().collect()
    }

    pub fn biases(&self) -> Vec<f32> {
        self.inner.iter().map(|(_, bias)| *bias).collect()
    }
}



impl rand::Fill for WeightAndBiasMatrix {
    fn fill<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        for (weight_row, bias) in self.inner.iter_mut() {
            rng.fill(weight_row.as_mut_slice());
            rng.fill(&mut [*bias]);
        }
    }
}


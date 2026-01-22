use crate::{layer::Layer, matrix::Matrix};







pub struct NeuralNetwork {
    layers: Vec<Layer>,
    weights: Vec<Matrix>,
}

impl NeuralNetwork {
    pub fn new(structure: &[usize]) -> Option<Self> {
        let mut weight_matrix_sizes: Vec<(usize, usize)> = Vec::new();
    
        if structure.len() < 2 { return None }
    
        for (i, &layer) in structure.iter().enumerate() {
            if i == 0 { continue; }
    
            weight_matrix_sizes.push((structure[i-1], layer));
        }
    
        
        let layers = structure.iter().map(|size| Layer::new(*size)).collect();
        let weights = weight_matrix_sizes.iter().map(|(width, height)| Matrix::new(*width, *height)).collect();

        return Some(Self { layers, weights });
    }

    pub fn layer(&self, index: usize) -> Option<&Layer> {
        self.layers.get(index)
    }

    pub fn layer_mut(&mut self, index: usize) -> Option<&mut Layer> {
        self.layers.get_mut(index)
    }

    pub fn weights(&self, output_layer_index: usize) -> Option<&Matrix> {
        self.weights.get(output_layer_index - 1)
    }
}


use crate::matrix::Matrix;







pub struct NeuralNetwork {
    layers: Vec<Vec<f32>>,
    weights: Vec<Matrix>,
    biases: Vec<Vec<f32>>
}

impl NeuralNetwork {
    pub fn new(structure: &[usize]) -> Option<Self> {
        let mut weight_matrix_sizes: Vec<(usize, usize)> = Vec::new();
    
        if structure.len() < 2 { return None }
    
        for (i, &layer) in structure.iter().enumerate() {
            if i == 0 { continue; }
    
            weight_matrix_sizes.push((structure[i-1], layer));
        }
    
        
        let layers: Vec<Vec<f32>> = structure.iter().map(|size| vec![0.0; *size]).collect();
        let weights: Vec<Matrix> = weight_matrix_sizes.iter().map(|(width, height)| Matrix::new(*width, *height)).collect();
        let mut biases: Vec<Vec<f32>> = Vec::new();
        for i in 1..layers.len() {
            biases.push(vec![0.0; layers[i].len()]);
        }

        return Some(Self { layers, weights, biases });
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn layer(&self, index: usize) -> Option<&Vec<f32>> {
        self.layers.get(index)
    }

    pub fn layer_mut(&mut self, index: usize) -> Option<&mut Vec<f32>> {
        self.layers.get_mut(index)
    }

    pub fn set_layer(&mut self, index: usize, data: Vec<f32>) -> Option<()> {
        if self.layers.get_mut(index)?.len() != data.len() { return None }

        *self.layers.get_mut(index)? = data;
        return Some(());
    }

    pub fn weights(&self, output_layer_index: usize) -> Option<&Matrix> {
        self.weights.get(output_layer_index - 1)
    }

    pub fn set_weights(&mut self, output_layer_index: usize, data: Matrix) -> Option<()> {
        let weights = self.weights.get_mut(output_layer_index - 1)?;
        if weights.height != data.height || weights.width != data.width { return None }

        *weights = data;
        return Some(());
    }

    pub fn biases(&self, output_layer_index: usize) -> Option<&Vec<f32>> {
        self.biases.get(output_layer_index - 1)
    }

    pub fn set_biases(&mut self, output_layer_index: usize, data: Vec<f32>) -> Option<()> {
        if self.biases.get_mut(output_layer_index - 1)?.len() != data.len() { return None }

        *self.biases.get_mut(output_layer_index - 1)? = data;
        return Some(());
    }
}


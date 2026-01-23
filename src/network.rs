use crate::matrix::Matrix;







pub struct NeuralNetworkDescriptor {
    pub layers: Vec<usize>,
    pub weights: Vec<Matrix>,
    pub biases: Vec<Vec<f32>>
}

impl NeuralNetworkDescriptor {
    pub fn new(structure: &[usize]) -> Option<Self> {
        let mut weight_matrix_sizes: Vec<(usize, usize)> = Vec::new();
    
        if structure.len() < 2 { return None }
    
        for (i, &layer) in structure.iter().enumerate() {
            if i == 0 { continue; }
    
            weight_matrix_sizes.push((structure[i-1], layer));
        }
    
        
        let weights: Vec<Matrix> = weight_matrix_sizes.iter().map(|(width, height)| Matrix::new(*width, *height)).collect();
        let mut biases: Vec<Vec<f32>> = Vec::new();
        for i in 1..structure.len() {
            biases.push(vec![0.0; structure[i]]);
        }

        return Some(Self { layers: structure.to_vec(), weights, biases });
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn layer_size(&self, index: usize) -> Option<usize> {
        Some(*(self.layers.get(index)?))
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


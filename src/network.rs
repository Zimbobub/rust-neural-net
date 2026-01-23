use crate::weight_bias_matrix::WeightAndBiasMatrix;







pub struct NeuralNetworkDescriptor {
    pub layers: Vec<usize>,
    // pub weights: Vec<Matrix>,
    // pub biases: Vec<Vec<f32>>,
    // bias at end of each row
    pub weights_and_biases: Vec<WeightAndBiasMatrix>
}

impl NeuralNetworkDescriptor {
    pub fn new(structure: &[usize]) -> Option<Self> {
        let mut weight_matrix_sizes: Vec<(usize, usize)> = Vec::new();
    
        if structure.len() < 2 { return None }
    
        for (i, &layer) in structure.iter().enumerate() {
            if i == 0 { continue; }
    
            weight_matrix_sizes.push((structure[i-1], layer));
        }
    
        
        let weights_and_biases: Vec<WeightAndBiasMatrix> = weight_matrix_sizes.iter().map(|(width, height)| WeightAndBiasMatrix::new(*width + 1, *height)).collect();

        return Some(Self {
            layers: structure.to_vec(),
            weights_and_biases
        });
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn layer_size(&self, index: usize) -> Option<usize> {
        Some(*(self.layers.get(index)?))
    }

    pub fn weights_and_biases(&self, output_layer_index: usize) -> Option<&WeightAndBiasMatrix> {
        self.weights_and_biases.get(output_layer_index - 1)
    }

    pub fn weights_and_biases_mut(&mut self, output_layer_index: usize) -> Option<&mut WeightAndBiasMatrix> {
        self.weights_and_biases.get_mut(output_layer_index - 1)
    }

    pub fn set_weights_and_biases(&mut self, output_layer_index: usize, data: WeightAndBiasMatrix) -> Option<()> {
        let weights_and_biases = self.weights_and_biases.get_mut(output_layer_index - 1)?;
        if weights_and_biases.height() != data.height() || weights_and_biases.width() != data.width() { return None }

        *weights_and_biases = data;
        return Some(());
    }


    /// Returns flattened weights and biases, as well as pointers to the start of each matrix
    pub fn flat_weights_and_biases(&self) -> (Vec<f32>, Vec<usize>) {
        let mut output: Vec<f32> = Vec::new();
        let mut ptr: usize = 0;
        let mut ptrs: Vec<usize> = Vec::new();

        for matrix in self.weights_and_biases.iter() {
            ptrs.push(ptr);
            ptr += matrix.size();

            output.append(&mut matrix.flat_weights_and_biases());
        }

        return (output, ptrs);
    }
}


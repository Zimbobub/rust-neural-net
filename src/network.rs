use crate::matrix::Matrix;



pub struct NeuralNetworkDescriptor {
    pub structure: Vec<usize>
}

impl NeuralNetworkDescriptor {
    pub fn new(structure: Vec<usize>) -> Self {
        return Self { structure }
    }


    pub fn matrix_sizes(&self) -> Vec<(usize, usize)> {
        let mut matricies: Vec<(usize, usize)> = Vec::new();
    
        if self.structure.len() < 2 { return matricies }
    
        for (i, &layer) in self.structure.iter().enumerate() {
            if i == 0 { continue; }
    
            matricies.push((self.structure[i-1], layer));
        }
    
        return matricies;
    }
    
}



pub struct NeuralNetwork {
    layers: Vec<Vec<f32>>,
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
    
        
        let layers = structure.iter().map(|size| vec![0.0; *size]).collect();
        let weights = weight_matrix_sizes.iter().map(|(width, height)| Matrix::new(*width, *height)).collect();

        return Some(Self { layers, weights });
    }

    pub fn layer(&self, index: usize) -> Option<&Vec<f32>> {
        self.layers.get(index)
    }

    pub fn layer_mut(&mut self, index: usize) -> Option<&mut Vec<f32>> {
        self.layers.get_mut(index)
    }

    pub fn weights(&self, output_layer_index: usize) -> Option<&Matrix> {
        self.weights.get(output_layer_index - 1)
    }
}


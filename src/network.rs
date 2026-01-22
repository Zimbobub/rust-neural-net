

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



pub trait NeuralNetwork {

}

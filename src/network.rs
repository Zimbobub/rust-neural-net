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


pub trait NeuralNetwork {

}

pub struct MyNeuralNetwork {
    descriptor: NeuralNetworkDescriptor,
    weights: Vec<Matrix>
}


// macro_rules! create_neural_network {
//     (($layer:expr),+) => {
//         let mut num_layers = 0;
//         let mut weights: Vec<Vec<Vec<f32>>> = Vec::new();
//         $(
//             // Each repeat will contain the following statement, with
//             // $element replaced with the corresponding expression.
//             weights.push([[]]);
//         )*

//         pub struct MyNeuralNetwork {
//             weights: [[];];
//         }
//     };
// }

use crate::matrix::Matrix;



pub struct NeuralNetworkDescriptor {
    structure: Vec<usize>
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

pub mod gpu;
pub mod matrix;
pub mod network;
pub mod forward_propagation;

use std::io::Write;

use pollster::FutureExt;

use crate::{forward_propagation::ForwardPropagation, matrix::Matrix, network::NeuralNetworkDescriptor};








pub async fn run() -> anyhow::Result<()> {
    let matrix_multiplier = gpu::MatrixMultiplier::new().await;
    
    let mut neural_network = NeuralNetworkDescriptor::new(&[784, 256, 128, 10]).unwrap();

    // set random weights
    for i in 1..neural_network.num_layers() {
        let mut weights = Matrix::new(neural_network.weights(i).unwrap().width, neural_network.weights(i).unwrap().height);
        
        for y in 0..weights.height {
            for x in 0..weights.width {
                weights.inner[y][x] = rand::random()
            }
        }

        neural_network.set_weights(i, weights);
    }

    // set random biases
    for i in 1..neural_network.num_layers() {
        let mut biases = vec![0.0f32; neural_network.biases(i).unwrap().len()];
        rand::fill(biases.as_mut_slice());
        neural_network.set_biases(i, biases);
    }

    // instance network
    let mut instance = ForwardPropagation::new(&neural_network);

    // set first layer data
    let mut rand_data = vec![0.0; 784];
    rand::fill(rand_data.as_mut_slice());
    instance.set_layer(0, rand_data).unwrap();

    instance.run(&matrix_multiplier);

    // run network
    // for i in 1..neural_network.num_layers() {
    //     let result = matrix_multiplier.run_once(
    //         neural_network.layer(i-1).unwrap(),
    //         neural_network.weights(i).unwrap(),
    //         neural_network.biases(i).unwrap(),
    //         neural_network.layer(i).unwrap().len()
    //     ).await?;

    //     // println!("{:?}", result);

    //     assert_eq!(result.len(), neural_network.layer(i).unwrap().len());

    //     *neural_network.layer_mut(i).unwrap() = result;
    // }

    Ok(())
}

fn main() {
    env_logger::init();
    run().block_on().unwrap();
}

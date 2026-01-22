pub mod gpu;
pub mod matrix;
pub mod network;


use std::io::Write;

use pollster::FutureExt;

use crate::network::NeuralNetwork;








pub async fn run() -> anyhow::Result<()> {
    let matrix_multiplier = gpu::MatrixMultiplier::new().await;
    
    let mut neural_network = NeuralNetwork::new(&[784, 256, 128, 10]).unwrap();

    // set random weights
    for i in 1..neural_network.num_layers() {
        let weights = neural_network.weights_mut(i).unwrap();
        
        for y in 0..weights.height {
            for x in 0..weights.width {
                weights.inner[y][x] = rand::random()
            }
        }
    }

    // set first layer
    *neural_network.layer_mut(0).unwrap() = vec![1.0; 1000];

    // run network
    for i in 1..neural_network.num_layers() {
        let input_layer = neural_network.layer(i-1).unwrap();
        let weights = neural_network.weights(i).unwrap();
        let output_layer_size = neural_network.layer(i).unwrap().len();

        // println!("{:?} {:#?} {}", input_layer, weights, output_layer_size);

        let result = matrix_multiplier.run_once(input_layer, weights, output_layer_size).await?;

        // println!("{:?}", result);

        assert_eq!(result.len(), neural_network.layer(i).unwrap().len());

        *neural_network.layer_mut(i).unwrap() = result;
    }

    Ok(())
}

fn main() {
    env_logger::init();
    run().block_on().unwrap();
}

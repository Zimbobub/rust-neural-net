pub mod gpu;
pub mod matrix;
pub mod network;


use pollster::FutureExt;

use crate::{matrix::Matrix, network::NeuralNetwork};








pub async fn run() -> anyhow::Result<()> {
    let matrix_multiplier = gpu::MatrixMultiplier::new().await;
    
    let mut neural_network = NeuralNetwork::new(&[4, 3, 1]).unwrap();
    neural_network.layer_mut(0).unwrap();

    let weights = Matrix::from_nested_vec(vec![
        vec![0.0, 0.1, 0.2, 0.3],
        vec![0.0, 0.1, 0.2, 0.4],
        vec![0.0, 0.1, 0.2, 0.5],
        vec![0.0, 0.1, 0.2, 0.6],
    ]).unwrap();

    // set first layer
    *neural_network.layer_mut(0).unwrap() = vec![1.0, 2.0, 3.0, 4.0];

    for i in 1..neural_network.num_layers() {

    }

    let result = matrix_multiplier.run_once(input_data, weights).await?;

    println!("{:?}", result);

    Ok(())
}

fn main() {
    env_logger::init();


    run().block_on().unwrap();
}

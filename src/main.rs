pub mod gpu;
pub mod matrix;


use flume::bounded;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use pollster::FutureExt;

use crate::matrix::Matrix;

pub async fn run() -> anyhow::Result<()> {
    let matrix_multiplier = gpu::MatrixMultiplier::new().await;
    

    // let input_data = (0..10u32).map(|x| x as f32).collect::<Vec<_>>();
    let input_data = vec![0.1, 0.2, 0.3, 0.4];
    // let weights = Matrix::new(vec![vec![0.1]; 4]).unwrap();
    let weights = Matrix::new(vec![
        vec![0.0, 0.1, 0.2, 0.3],
        vec![0.0, 0.1, 0.2, 0.3],
        vec![0.0, 0.1, 0.2, 0.3],
        vec![0.0, 0.1, 0.2, 0.3],
    ]).unwrap();

    let result = matrix_multiplier.run_once(input_data, weights).await?;

    println!("{:?}", result);

    println!("Success!");

    Ok(())
}

fn main() {
    env_logger::init();
    run().block_on().unwrap();
}

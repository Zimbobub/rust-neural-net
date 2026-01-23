use crate::{gpu::MatrixMultiplier, network::NeuralNetworkDescriptor};

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};


#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ShaderParameters {
    b_is_input_layer: u32,
    weight_matrix_index: u32,
    weight_matrix_width: u32,
    weight_matrix_height: u32
}


pub struct ForwardPropagation<'a> {
    layers: Vec<Vec<f32>>,
    descriptor: &'a NeuralNetworkDescriptor
}

impl<'a> ForwardPropagation<'a> {
    pub fn new(descriptor: &'a NeuralNetworkDescriptor) -> Self {
        let layers: Vec<Vec<f32>> = descriptor.layers.iter().map(|size| vec![0.0; *size]).collect();
        
        Self { layers, descriptor }
    }


    pub fn layer(&self, index: usize) -> Option<&Vec<f32>> {
        self.layers.get(index)
    }

    pub fn layer_mut(&mut self, index: usize) -> Option<&mut Vec<f32>> {
        self.layers.get_mut(index)
    }

    pub fn set_layer(&mut self, index: usize, data: Vec<f32>) -> Option<()> {
        if self.layers.get_mut(index)?.len() != data.len() { return None }

        *self.layers.get_mut(index)? = data;
        return Some(());
    }




    pub async fn run(&mut self, gpu_handle: &MatrixMultiplier) -> anyhow::Result<Vec<f32>> {
        let start = std::time::Instant::now();

        // compute shader will keep switching between using a_layers as input neurons and b_layers as output neurons and vice versa
        // a buffer for each will be the size of the biggest layer in a_layers and b_layers
        let a_layers: Vec<usize> = self.descriptor.layers.iter().enumerate().filter_map(|(i, size)| if i % 2 == 0 { Some(*size) } else { None }).collect();
        let b_layers: Vec<usize> = self.descriptor.layers.iter().enumerate().filter_map(|(i, size)| if i % 2 == 1 { Some(*size) } else { None }).collect();

        let a_buffer_size = a_layers.iter().max().expect("No layers in neural network descriptor!");
        let b_buffer_size = b_layers.iter().max().expect("Only one layer in neural network descriptor!");

        println!("a layers {:?}", a_layers);
        println!("b layers {:?}", b_layers);

        let mut initial_a_buffer = self.layer(0).unwrap().clone();
        initial_a_buffer.resize(*a_buffer_size, 0.0);

        let a_neuron_buffer: wgpu::Buffer = gpu_handle.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("a_neurons"),
            contents: bytemuck::cast_slice(&initial_a_buffer),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
        });

        let b_neuron_buffer = gpu_handle.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("b_neurons"),
            size: (b_buffer_size * size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });


        // let layer_size_buffer: wgpu::Buffer = gpu_handle.device.create_buffer_init(&BufferInitDescriptor {
        //     label: Some("layer_sizes"),
        //     contents: bytemuck::cast_slice(&self.descriptor.layers),
        //     usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        // });

        
        let (weights_and_biases, weight_matrix_ptrs) = self.descriptor.flat_weights_and_biases();

        let weight_and_bias_buffer: wgpu::Buffer = gpu_handle.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("weights"),
            contents: bytemuck::cast_slice(weights_and_biases.as_slice()),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        });

        
        // output transfer buffer the size of the output neuron layer
        let temp_buffer = gpu_handle.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("temp"),
            size: (self.descriptor.layers.last().unwrap() * size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });



        let uniform_buffer = gpu_handle.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: std::mem::size_of::<ShaderParameters>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });


        


        
        let bind_group: wgpu::BindGroup = gpu_handle.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &gpu_handle.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_neuron_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_neuron_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weight_and_bias_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });


        // let mut weight_ptr: usize = 0;
        for i in 1..self.layers.len() {
            // send dispath params
            let shader_params = ShaderParameters {
                b_is_input_layer: (i % 2 == 0) as u32,
                weight_matrix_index: weight_matrix_ptrs[i-1] as u32,
                weight_matrix_width: self.descriptor.weights_and_biases(i).unwrap().width() as u32,
                weight_matrix_height: self.descriptor.weights_and_biases(i).unwrap().height() as u32,
            };

            gpu_handle.queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&shader_params));

            let mut encoder = gpu_handle.device.create_command_encoder(&Default::default());

            // create pass
            {
                // We specified 64 threads per workgroup in the shader, so we need to compute how many
                // workgroups we need to dispatch.
                let num_dispatches = self.descriptor.layer_size(i).unwrap().div_ceil(64) as u32;

                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&gpu_handle.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(num_dispatches, 1, 1);
            }

            // we could instead create a vector of passes and send it all to the gpu at once
            gpu_handle.queue.submit(Some(encoder.finish()));

            // + layer_size at the end is for the bias at the end of each row
            // weight_ptr += self.descriptor.weights_and_biases(i).unwrap().size() + self.descriptor.layer_size(i).unwrap();
        }






        let output = {
            // The mapping process is async, so we'll need to create a channel to get
            // the success flag for our mapping
            let (tx, rx) = flume::bounded(1);

            // We send the success or failure of our mapping via a callback
            temp_buffer.map_async(wgpu::MapMode::Read, .., move |result| {
                tx.send(result).unwrap()
            });

            // The callback we submitted to map async will only get called after the
            // device is polled or the queue submitted
            gpu_handle.device.poll(wgpu::PollType::wait_indefinitely())?;

            // We check if the mapping was successful here
            rx.recv_async().await??;

            // We then get the bytes that were stored in the buffer
            let transfer_back = std::time::Instant::now();
            let output_data = temp_buffer.get_mapped_range(..);
            println!("transfer_back: {}us", transfer_back.elapsed().as_micros());
            // Now we have the data on the CPU we can do what ever we want to with it
            // assert_eq!(&input_data, bytemuck::cast_slice(&output_data));

            bytemuck::cast_slice(&output_data).to_vec()
        };

        // We need to unmap the buffer to be able to use it again
        // temp_buffer.unmap();


        println!("time {}us", start.elapsed().as_micros());





        Ok(output)
    }
}
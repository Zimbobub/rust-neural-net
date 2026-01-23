use crate::{gpu::MatrixMultiplier, network::NeuralNetworkDescriptor};

use wgpu::util::{BufferInitDescriptor, DeviceExt};





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




    pub fn run(&mut self, gpu_handle: &MatrixMultiplier) -> Option<()> {
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


        let layer_size_buffer: wgpu::Buffer = gpu_handle.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("layer_sizes"),
            contents: bytemuck::cast_slice(&self.descriptor.layers),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        });


        // weights is a 3d vector, these point to the start of each weight matrix
        let mut weight_ptrs: Vec<usize> = Vec::new();
        let mut acc: usize = 0;
        for weight_matrix in self.descriptor.weights.iter() {
            weight_ptrs.push(acc);
            acc += weight_matrix.width * weight_matrix.height;
        }

        let weight_ptr_buffer: wgpu::Buffer = gpu_handle.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("weight_ptrs"),
            contents: bytemuck::cast_slice(&initial_a_buffer),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        });

        let weight_buffer: wgpu::Buffer = gpu_handle.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("weights"),
            contents: bytemuck::cast_slice(&weights.flatten()),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        });

        let bias_buffer: wgpu::Buffer = gpu_handle.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("biases"),
            contents: bytemuck::cast_slice(&biases),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        });

        
    
        let temp_buffer = gpu_handle.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("temp"),
            size: (output_layer_size * size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
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
                    resource: weight_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bias_buffer.as_entire_binding(),
                },
            ],
        });


        let mut weight_ptr: usize = 0;
        for i in 1..self.layers.len() {


            weight_ptr += self.descriptor.weights(i).unwrap().
        }

        Some(())
    }
}
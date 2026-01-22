use flume::bounded;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use pollster::FutureExt;

use crate::matrix::Matrix;





pub struct MatrixMultiplier {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pipeline: wgpu::ComputePipeline
}



impl MatrixMultiplier {
    pub async fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
    
        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/matrix_multiply.wgsl"));
        // wgpu::ShaderModuleDescriptor::
        
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Matrix Multiply Compute Pipeline"),
            layout: None,
            module: &shader,
            entry_point: None,
            compilation_options: Default::default(),
            cache: Default::default(),
        });


        return Self {
            device,
            queue,
            pipeline
        };
    }



    pub async fn run_once(&self, input_data: &Vec<f32>, weights: &Matrix, output_layer_size: usize) -> anyhow::Result<Vec<f32>> {
        let input_neuron_buffer: wgpu::Buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("input_neurons"),
            contents: bytemuck::cast_slice(&input_data),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        });

        let weight_buffer: wgpu::Buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("weights"),
            contents: bytemuck::cast_slice(&weights.flatten()),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        });

        let output_neuron_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output_neurons"),
            size: (output_layer_size * size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
    
        let temp_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("temp"),
            size: (output_layer_size * size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });




        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_neuron_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_neuron_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());

        {
            // We specified 64 threads per workgroup in the shader, so we need to compute how many
            // workgroups we need to dispatch.
            let num_dispatches = output_layer_size.div_ceil(64) as u32;

            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_dispatches, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&output_neuron_buffer, 0, &temp_buffer, 0, output_neuron_buffer.size());

        self.queue.submit([encoder.finish()]);

        let output = {
            // The mapping process is async, so we'll need to create a channel to get
            // the success flag for our mapping
            let (tx, rx) = bounded(1);

            // We send the success or failure of our mapping via a callback
            temp_buffer.map_async(wgpu::MapMode::Read, .., move |result| {
                tx.send(result).unwrap()
            });

            // The callback we submitted to map async will only get called after the
            // device is polled or the queue submitted
            self.device.poll(wgpu::PollType::wait_indefinitely())?;

            // We check if the mapping was successful here
            rx.recv_async().await??;

            // We then get the bytes that were stored in the buffer
            let output_data = temp_buffer.get_mapped_range(..);
            
            // Now we have the data on the CPU we can do what ever we want to with it
            // assert_eq!(&input_data, bytemuck::cast_slice(&output_data));

            bytemuck::cast_slice(&output_data).to_vec()
        };

        // We need to unmap the buffer to be able to use it again
        temp_buffer.unmap();

        println!("Success!");

        Ok(output)

    }
}
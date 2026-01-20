use flume::bounded;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use pollster::FutureExt;





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
    
        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/blit.wgsl"));
    
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Introduction Compute Pipeline"),
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
}
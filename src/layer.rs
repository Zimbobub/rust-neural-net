







pub struct Layer {
    inner: Vec<f32>
}


impl Layer {
    pub fn new(size: usize) -> Self {
        Self { inner: vec![0.0; size] }
    }
}
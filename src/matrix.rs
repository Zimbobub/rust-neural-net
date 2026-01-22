



pub struct Matrix {
    inner: Vec<Vec<f32>>,
    width: usize,
    height: usize
}


impl Matrix {
    pub fn new(data: Vec<Vec<f32>>) -> Option<Self> {
        let height = data.len();
        let mut width: Option<usize> = None;

        for row in data.iter() {
            match width {
                None => {
                    width = Some(row.len());
                },
                Some(width_inner) => {
                    if width_inner != row.len() {
                        return None
                    }
                }
            }
        }

        return Some(Self { inner: data, width: width?, height })
    }

    pub fn flatten(&self) -> Vec<f32> {
        return self.inner.iter().flatten().map(|f| *f).collect();
    }
}
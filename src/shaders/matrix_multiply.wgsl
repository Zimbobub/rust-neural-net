// Data f32 read only array
@group(0) @binding(0) var<storage, read> input_neurons: array<f32>;
// Weights f32 read only matrix
@group(0) @binding(0) var<storage, read> weights: array<array<f32>>;
// Output data f32 writeable array
@group(0) @binding(1) var<storage, read_write> output_neurons: array<u32>;


@compute
// Specifies the "dimension" of this work group, max 256
// one thread run per output neuron
@workgroup_size(64, 1, 1)
// this matrix multiplier only works for (n*m) * (m*1) multiplications, used in neural networks
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let index = global_invocation_id.x;
    let total = arrayLength(&input);

    // workgroup_size may not be a multiple of the array size so
    // we need to exit out a thread would index out of bounds.
    if (index >= total) {
        return;
    }

    var weighted_sum: f32 = 0.0;
    for (var i = 0; i < 10; i++) {
        weighted_sum += weights[global_invocation_id.x][i] * input_neurons[i]
    }

    output[global_invocation_id.x] = weighted_sum;
}
// Data f32 read only array
@group(0) @binding(0) var<storage, read> input_neurons: array<f32>;
// Weights f32 read only matrix, cols = #inputs, rows = #outputs
@group(0) @binding(1) var<storage, read> weights: array<f32>;
// Output data f32 writeable array
@group(0) @binding(2) var<storage, read_write> output_neurons: array<u32>;


@compute
// Specifies the "dimension" of this work group, max 256
// one thread run per output neuron
@workgroup_size(64, 1, 1)
// this matrix multiplier only works for (n*m) * (m*1) multiplications, used in neural networks
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let output_index = global_invocation_id.x;
    let n_inputs = arrayLength(&input_neurons);
    let n_outputs = arrayLength(&output_neurons);

    // workgroup_size may not be a multiple of the array size so
    // we need to exit out a thread would index out of bounds.
    if (output_index >= n_outputs) {
        return;
    }

    let start_index = n_inputs * output_index;
    let end_index = n_inputs * (output_index + 1);
    var weighted_sum: f32 = 0.0;

    for (var i = start_index; i < end_index; i++) {
        weighted_sum += weights[start_index + i] * input_neurons[i]
    }

    output[global_invocation_id.x] = weighted_sum;
}
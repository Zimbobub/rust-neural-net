
struct Globals {
    b_is_input_layer: u32,
    weight_matrix_index: u32,
    weight_matrix_size: vec2<u32>
}


@group(0) @binding(0) var<storage, read_write> a_neurons: array<f32>;
@group(0) @binding(1) var<storage, read_write> b_neurons: array<f32>;

// Weights f32 read only matrix, cols = #inputs, rows = #outputs
// 3d vector where each 2d is a weight matrix, with each row ending with the bias
// weights are multiplied then wherever the pointer finishes, the next item is added as the bias
@group(0) @binding(2) var<storage, read> weights_and_biases: array<f32>;

// changes on each iteration
@group(0) @binding(3) var<uniform> globals: Globals;

@compute
// Specifies the "dimension" of this work group, max 256
// one thread run per output neuron
@workgroup_size(64, 1, 1)
// this matrix multiplier only works for (n*m) * (m*1) multiplications, used in neural networks
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let output_index = global_invocation_id.x;

    let n_inputs: u32 = globals.weight_matrix_size.x;
    let n_outputs: u32 = globals.weight_matrix_size.y;


    // workgroup_size may not be a multiple of the array size so
    // we need to exit out a thread would index out of bounds.
    if (output_index >= n_outputs) {
        return;
    }

    let row_index = globals.weight_matrix_index + n_inputs * output_index;
    var weighted_sum: f32 = 0.0;

    for (var i: u32 = 0; i < n_inputs; i++) {
        if globals.b_is_input_layer == 0 {
            weighted_sum += weights_and_biases[row_index + i] * a_neurons[i];
        } else {
            weighted_sum += weights_and_biases[row_index + i] * b_neurons[i];
        }
    }

    if globals.b_is_input_layer == 0 {
        b_neurons[global_invocation_id.x] = weighted_sum + weights_and_biases[row_index + n_inputs];
    } else {
        a_neurons[global_invocation_id.x] = weighted_sum + weights_and_biases[row_index + n_inputs];
    }
}
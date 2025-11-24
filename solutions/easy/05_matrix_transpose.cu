#include <cuda_runtime.h>

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int output_row = blockIdx.x * blockDim.x + threadIdx.x;
    int output_col = blockIdx.y * blockDim.y + threadIdx.y;

    if (output_row < cols && output_col < rows) {
        int input_index = output_col * cols + output_row;

        int output_index = output_row * rows + output_col;

        output[output_index] = input[input_index];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}

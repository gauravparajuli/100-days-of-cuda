#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel declaration
__global__ void conv3d_kernel(const float* input, const float* kernel, float* output,
    int input_depth, int input_rows, int input_cols,
    int kernel_depth, int kernel_rows, int kernel_cols,
    int output_depth, int output_rows, int output_cols) {
    // Get output coordinates from thread indices
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // x-dimension
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // y-dimension
    int dep = blockIdx.z;                            // z-dimension

    // Check if thread is within output bounds
    if (dep < output_depth && row < output_rows && col < output_cols) {
        float sum = 0.0f;

        // Perform 3D convolution
        for (int kd = 0; kd < kernel_depth; kd++) {
        for (int kr = 0; kr < kernel_rows; kr++) {
        for (int kc = 0; kc < kernel_cols; kc++) {
        // Calculate input indices
        int input_d = dep + kd;
        int input_r = row + kr;
        int input_c = col + kc;

        // Calculate flat indices
        int input_idx = input_d * (input_rows * input_cols) + 
                input_r * input_cols + 
                input_c;
        int kernel_idx = kd * (kernel_rows * kernel_cols) + 
                kr * kernel_cols + 
                kc;

        // Accumulate the convolution sum
        sum += input[input_idx] * kernel[kernel_idx];
        }
        }
        }

        // Write result to output
        int output_idx = dep * (output_rows * output_cols) + 
        row * output_cols + 
        col;
        output[output_idx] = sum;
    }
}

void solve(const float* input, const float* kernel, float* output,
           int input_depth, int input_rows, int input_cols,
           int kernel_depth, int kernel_rows, int kernel_cols) {
    
    // Calculate output dimensions
    int output_depth = input_depth - kernel_depth + 1;
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;

    // Define block and grid dimensions
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (output_cols + blockDim.x - 1) / blockDim.x,
        (output_rows + blockDim.y - 1) / blockDim.y,
        output_depth
    );

    // Launch CUDA kernel
    conv3d_kernel<<<gridDim, blockDim>>>(
        input, kernel, output,
        input_depth, input_rows, input_cols,
        kernel_depth, kernel_rows, kernel_cols,
        output_depth, output_rows, output_cols
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // Synchronize device
    cudaDeviceSynchronize();
}

int main() {
    // Example dimensions
    const int input_depth = 32;
    const int input_rows = 64;
    const int input_cols = 64;
    const int kernel_depth = 3;
    const int kernel_rows = 3;
    const int kernel_cols = 3;

    // Calculate sizes
    size_t input_size = input_depth * input_rows * input_cols * sizeof(float);
    size_t kernel_size = kernel_depth * kernel_rows * kernel_cols * sizeof(float);
    int output_depth = input_depth - kernel_depth + 1;
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;
    size_t output_size = output_depth * output_rows * output_cols * sizeof(float);

    // Host pointers
    float *h_input, *h_kernel, *h_output;
    
    // Device pointers
    float *d_input, *d_kernel, *d_output;

    // Allocate host memory
    h_input = (float*)malloc(input_size);
    h_kernel = (float*)malloc(kernel_size);
    h_output = (float*)malloc(output_size);

    // Allocate device memory
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_kernel, kernel_size);
    cudaMalloc(&d_output, output_size);

    // Initialize input and kernel data here (not shown)

    // Copy data to device
    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_size, cudaMemcpyHostToDevice);

    // Call the convolution function
    solve(d_input, d_kernel, d_output,
          input_depth, input_rows, input_cols,
          kernel_depth, kernel_rows, kernel_cols);

    // Copy results back to host
    cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    free(h_input);
    free(h_kernel);
    free(h_output);

    return 0;
}
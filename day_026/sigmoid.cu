#include <cuda_runtime.h>
#include <stdio.h>

// Sigmoid function\_
__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// CUDA kernel to apply sigmoid activation
__global__ void sigmoidKernel(float* d_out, const float* d_in, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_out[idx] = sigmoid(d_in[idx]);
    }
}

// Host function to launch kernel
void applySigmoid(float* h_out, const float* h_in, int size) {
    float *d_in, *d_out;
    size_t bytes = size * sizeof(float);
    
    // Allocate device memory
    cudaMalloc((void**)&d_in, bytes);
    cudaMalloc((void**)&d_out, bytes);
    
    // Copy data to device
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    
    // Define grid and block sizes
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    // Launch the kernel
    sigmoidKernel<<<gridSize, blockSize>>>(d_out, d_in, size);
    
    // Copy result back to host
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_in);
    cudaFree(d_out);
}

int main() {
    int size = 10;
    float h_in[10] = { -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, -3.0f, 4.0f, -4.0f, 5.0f };
    float h_out[10];
    
    applySigmoid(h_out, h_in, size);
    
    // Print results
    printf("Sigmoid Activation Results:\n");
    for (int i = 0; i < size; i++) {
        printf("sigmoid(%f) = %f\n", h_in[i], h_out[i]);
    }
    
    return 0;
}

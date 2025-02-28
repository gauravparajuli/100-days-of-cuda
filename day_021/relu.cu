#include <stdio.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(float *d_in, float *d_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_out[idx] = fmaxf(0.0f, d_in[idx]);
    }
}

void relu_cuda(float *h_in, float *h_out, int n) {
    float *d_in, *d_out;
    size_t size = n * sizeof(float);
    
    // Allocate device memory
    cudaMalloc((void **)&d_in, size);
    cudaMalloc((void **)&d_out, size);
    
    // Copy input data to device
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    
    // Define CUDA kernel launch parameters
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    // Launch kernel
    relu_kernel<<<gridSize, blockSize>>>(d_in, d_out, n);
    
    // Copy result back to host
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_in);
    cudaFree(d_out);
}

int main() {
    const int n = 10;
    float h_in[n] = {-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -4.0, 4.5, -0.5};
    float h_out[n];
    
    relu_cuda(h_in, h_out, n);
    
    printf("ReLU output: \n");
    for (int i = 0; i < n; i++) {
        printf("%f ", h_out[i]);
    }
    printf("\n");
    
    return 0;
}

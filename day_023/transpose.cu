#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix transposition
__global__ void transposeMatrix(float *input, float *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        output[x * height + y] = input[y * width + x];
    }
}

// Helper function to check CUDA errors
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        printf("%s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Matrix dimensions
    const int WIDTH = 4;
    const int HEIGHT = 4;
    const int SIZE = WIDTH * HEIGHT * sizeof(float);

    // Host matrices
    float h_input[] = {
        1.0f,  2.0f,  3.0f,  4.0f,
        5.0f,  6.0f,  7.0f,  8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
       13.0f, 14.0f, 15.0f, 16.0f
    };
    float *h_output = (float*)malloc(SIZE);

    // Device matrices
    float *d_input, *d_output;
    
    // Allocate device memory
    checkCudaError(cudaMalloc(&d_input, SIZE), "CUDA malloc input failed");
    checkCudaError(cudaMalloc(&d_output, SIZE), "CUDA malloc output failed");

    // Copy input matrix to device
    checkCudaError(cudaMemcpy(d_input, h_input, SIZE, cudaMemcpyHostToDevice), 
                  "CUDA memcpy to device failed");

    // Set grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, 
                  (HEIGHT + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    transposeMatrix<<<gridSize, blockSize>>>(d_input, d_output, WIDTH, HEIGHT);
    checkCudaError(cudaGetLastError(), "Kernel launch failed");

    // Synchronize device
    checkCudaError(cudaDeviceSynchronize(), "CUDA sync failed");

    // Copy result back to host
    checkCudaError(cudaMemcpy(h_output, d_output, SIZE, cudaMemcpyDeviceToHost),
                  "CUDA memcpy to host failed");

    // Print original matrix
    printf("Original Matrix:\n");
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            printf("%6.1f ", h_input[i * WIDTH + j]);
        }
        printf("\n");
    }

    // Print transposed matrix
    printf("\nTransposed Matrix:\n");
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < HEIGHT; j++) {
            printf("%6.1f ", h_output[i * HEIGHT + j]);
        }
        printf("\n");
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_output);

    return 0;
}
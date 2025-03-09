#include <stdio.h>
#include <cuda_runtime.h>
#include <float.h>  // For FLT_MAX

// Define block size
#define TILE_WIDTH 16
#define KERNEL_SIZE 3
#define POOL_SIZE 2

// CUDA kernel for convolution
__global__ void conv2d_kernel(float *d_input, float *d_kernel, float *d_output, int inputWidth, int inputHeight, int kernelSize) {
    int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int y = blockIdx.y * TILE_WIDTH + threadIdx.y;
    
    if (x < inputWidth && y < inputHeight) {
        float sum = 0.0f;
        int halfKernel = kernelSize / 2;
        
        // Convolve the kernel with the image
        for (int i = -halfKernel; i <= halfKernel; ++i) {
            for (int j = -halfKernel; j <= halfKernel; ++j) {
                int ix = min(max(x + i, 0), inputWidth - 1);
                int iy = min(max(y + j, 0), inputHeight - 1);
                sum += d_input[iy * inputWidth + ix] * d_kernel[(i + halfKernel) * kernelSize + (j + halfKernel)];
            }
        }
        d_output[y * inputWidth + x] = sum;
    }
}

// CUDA kernel for max pooling
__global__ void max_pooling_kernel(float *d_input, float *d_output, int inputWidth, int inputHeight, int poolSize) {
    int x = blockIdx.x * POOL_SIZE + threadIdx.x;
    int y = blockIdx.y * POOL_SIZE + threadIdx.y;
    
    if (x < inputWidth / poolSize && y < inputHeight / poolSize) {
        float maxVal = -FLT_MAX;
        
        // Max pooling: find maximum value in the pool
        for (int i = 0; i < poolSize; ++i) {
            for (int j = 0; j < poolSize; ++j) {
                int ix = min(x * poolSize + i, inputWidth - 1);
                int iy = min(y * poolSize + j, inputHeight - 1);
                maxVal = fmaxf(maxVal, d_input[iy * inputWidth + ix]);
            }
        }
        
        d_output[y * (inputWidth / poolSize) + x] = maxVal;
    }
}

int main() {
    // Example input image and kernel dimensions
    int inputWidth = 8;
    int inputHeight = 8;
    int kernelSize = 3;
    int poolSize = 2;
    
    float h_input[] = {
        1, 2, 3, 4, 5, 6, 7, 8,
        1, 2, 3, 4, 5, 6, 7, 8,
        1, 2, 3, 4, 5, 6, 7, 8,
        1, 2, 3, 4, 5, 6, 7, 8,
        1, 2, 3, 4, 5, 6, 7, 8,
        1, 2, 3, 4, 5, 6, 7, 8,
        1, 2, 3, 4, 5, 6, 7, 8,
        1, 2, 3, 4, 5, 6, 7, 8
    };

    float h_kernel[] = {
        0, 1, 0,
        1, 1, 1,
        0, 1, 0
    };

    float *d_input, *d_kernel, *d_output, *h_output;
    
    // Allocate memory on device
    cudaMalloc(&d_input, inputWidth * inputHeight * sizeof(float));
    cudaMalloc(&d_kernel, kernelSize * kernelSize * sizeof(float));
    cudaMalloc(&d_output, inputWidth * inputHeight * sizeof(float));  // convolution result
    h_output = (float*)malloc(inputWidth * inputHeight * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, h_input, inputWidth * inputHeight * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block sizes
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid((inputWidth + TILE_WIDTH - 1) / TILE_WIDTH, (inputHeight + TILE_WIDTH - 1) / TILE_WIDTH, 1);
    
    // Launch convolution kernel
    conv2d_kernel<<<dimGrid, dimBlock>>>(d_input, d_kernel, d_output, inputWidth, inputHeight, kernelSize);
    
    // Copy back convolution result
    cudaMemcpy(h_output, d_output, inputWidth * inputHeight * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Display convolution result
    printf("Convolution Output:\n");
    for (int i = 0; i < inputHeight; i++) {
        for (int j = 0; j < inputWidth; j++) {
            printf("%f ", h_output[i * inputWidth + j]);
        }
        printf("\n");
    }

    // Max pooling: Adjust the output size based on pool size
    float *d_pool_output, *h_pool_output;
    cudaMalloc(&d_pool_output, (inputWidth / poolSize) * (inputHeight / poolSize) * sizeof(float));
    h_pool_output = (float*)malloc((inputWidth / poolSize) * (inputHeight / poolSize) * sizeof(float));

    // Launch max pooling kernel
    max_pooling_kernel<<<dimGrid, dimBlock>>>(d_output, d_pool_output, inputWidth, inputHeight, poolSize);

    // Copy back pooled result
    cudaMemcpy(h_pool_output, d_pool_output, (inputWidth / poolSize) * (inputHeight / poolSize) * sizeof(float), cudaMemcpyDeviceToHost);

    // Display pooled output
    printf("\nMax Pooling Output:\n");
    for (int i = 0; i < inputHeight / poolSize; i++) {
        for (int j = 0; j < inputWidth / poolSize; j++) {
            printf("%f ", h_pool_output[i * (inputWidth / poolSize) + j]);
        }
        printf("\n");
    }

    // Clean up
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    cudaFree(d_pool_output);
    free(h_output);
    free(h_pool_output);

    return 0;
}

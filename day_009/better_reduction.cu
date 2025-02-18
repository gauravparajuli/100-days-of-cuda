#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// Kernel function for parallel sum reduction
__global__ void sumReduction(float *input, float *output, int n) {
    __shared__ float partialSum[2 * BLOCK_SIZE];

    unsigned int t = threadIdx.x;
    unsigned int start = blockIdx.x * 2 * blockDim.x;

    // Load two elements per thread into shared memory
    partialSum[t] = input[start + t];
    partialSum[blockDim.x + t] = input[start + blockDim.x + t];

    // Perform reduction in shared memory
    for (unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
        __syncthreads(); // Ensure all threads have written their values
        if (t < stride) {
            partialSum[t] += partialSum[t + stride];
        }
    }

    // Write the result of this block to the output array
    if (t == 0) {
        output[blockIdx.x] = partialSum[0];
    }
}

int main() {
    const int n = 1 << 16; // Input size (2^16 elements)
    const size_t size = n * sizeof(float);

    // Allocate host memory for input and output
    float *h_input = (float *)malloc(size);
    float *h_output = (float *)malloc((n / (2 * BLOCK_SIZE)) * sizeof(float));

    // Initialize input array with random values
    for (int i = 0; i < n; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX; // Random float between 0 and 1
    }

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, (n / (2 * BLOCK_SIZE)) * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Launch the kernel
    int numBlocks = (n + (2 * BLOCK_SIZE - 1)) / (2 * BLOCK_SIZE);
    sumReduction<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, n);

    // Copy the result back to host
    cudaMemcpy(h_output, d_output, (n / (2 * BLOCK_SIZE)) * sizeof(float), cudaMemcpyDeviceToHost);

    // Perform final reduction on the host
    float finalSum = 0.0f;
    for (int i = 0; i < numBlocks; i++) {
        finalSum += h_output[i];
    }

    // Verify the result
    float expectedSum = 0.0f;
    for (int i = 0; i < n; i++) {
        expectedSum += h_input[i];
    }

    printf("Computed Sum: %f\n", finalSum);
    printf("Expected Sum: %f\n", expectedSum);

    // Free memory
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
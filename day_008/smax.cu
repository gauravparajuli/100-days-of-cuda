#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void softmax(int w, int h, float *input, float *output) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // boundary condition
    if (row < h && col < w) {

        float maxval = input[row*w]; // set max value as first value of that row
        // find the maximum value
        for (int i = 1; i < w; i++) {
            maxval = max(maxval, input[row * w + i]);
        }
        
        float divisor = 0.f;
        for (int i = 0; i < w; i++) {
            divisor += exp(input[row*w+i] - maxval);
        }

        output[row * w + col] = exp(input[row * w + col] - maxval) / divisor;
    }
}

int main() {
    int w = 4; // width of the matrix
    int h = 3; // height of the matrix

    // Allocate host memory
    float *h_input = (float *)malloc(w * h * sizeof(float));
    float *h_output = (float *)malloc(w * h * sizeof(float));

    // Initialize input data
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            h_input[row * w + col] = (float)(row * w + col); // Example initialization
        }
    }

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, w * h * sizeof(float));
    cudaMalloc((void **)&d_output, w * h * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, w * h * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(16, 16); // 16x16 threads per block
    dim3 gridDim((w + blockDim.x - 1) / blockDim.x, (h + blockDim.y - 1) / blockDim.y); // because I want block to cover the entire image/matrix

    // Launch the kernel
    softmax<<<gridDim, blockDim>>>(w, h, d_input, d_output);

    // Copy the result back to host
    cudaMemcpy(h_output, d_output, w * h * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    printf("Input:\n");
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            printf("%f ", h_input[i * w + j]);
        }
        printf("\n");
    }

    printf("\nOutput:\n");
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            printf("%f ", h_output[i * w + j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    free(h_input);
    free(h_output);

    return 0;
}
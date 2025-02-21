#include <stdio.h>
#include <cuda_runtime.h>
#include <cmath>
#include "timer.h"

// GPU Kernel for GELU
__global__ void gelu_forward(float *input, float *output, unsigned int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        float x = input[tid];
        float cdf = 0.5f * (1.0f + tanhf(sqrtf(2 / M_PI) * (x + 0.044715f * x * x * x)));
        output[tid] = x * cdf;
    }
}

// CPU Version of GELU
void gelu_forward_cpu(float *input, float *output, unsigned int n) {
    for (unsigned int i = 0; i < n; i++) {
        float x = input[i];
        float cdf = 0.5f * (1.0f + tanhf(sqrtf(2 / M_PI) * (x + 0.044715f * x * x * x)));
        output[i] = x * cdf;
    }
}

// Function to compare two arrays
bool compare_arrays(float *a, float *b, unsigned int n, float tolerance = 1e-5) {
    for (unsigned int i = 0; i < n; i++) {
        if (fabs(a[i] - b[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

int main() {
    // Define the size of the input array
    const unsigned int n = 102400000;
    const unsigned int size = n * sizeof(float);

    // Timer object
    Timer timer;

    // Allocate host memory
    float *h_input = (float *)malloc(size);
    float *h_output_gpu = (float *)malloc(size);
    float *h_output_cpu = (float *)malloc(size);

    // Initialize host input with some values
    for (unsigned int i = 0; i < n; i++) {
        h_input[i] = (float)(i) / n; // Normalized values between 0 and 1
    }

    // 1. Measure time to allocate GPU memory
    startTime(&timer);
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, size);
    stopTime(&timer);
    printElapsedTime(timer, "Time to allocate GPU memory", BLUE);

    // 2. Measure time to copy input data to device
    startTime(&timer);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    stopTime(&timer);
    printElapsedTime(timer, "Time to copy input data to GPU", YELLOW);

    // Define block size and grid size
    unsigned int blockSize = 1024;
    unsigned int gridSize = (n + blockSize - 1) / blockSize;

    // 3. Measure time to execute the GELU kernel on GPU
    startTime(&timer);
    gelu_forward<<<gridSize, blockSize>>>(d_input, d_output, n);
    cudaDeviceSynchronize(); // Ensure kernel execution is complete
    stopTime(&timer);
    printElapsedTime(timer, "Time to execute GELU kernel on GPU", GREEN);

    // 4. Measure time to copy the result back to host
    startTime(&timer);
    cudaMemcpy(h_output_gpu, d_output, size, cudaMemcpyDeviceToHost);
    stopTime(&timer);
    printElapsedTime(timer, "Time to copy results back to host", MAGENTA);

    // 5. Measure time to deallocate GPU memory
    startTime(&timer);
    cudaFree(d_input);
    cudaFree(d_output);
    stopTime(&timer);
    printElapsedTime(timer, "Time to deallocate GPU memory", RED);

    // 6. Measure time to execute the GELU kernel on CPU
    startTime(&timer);
    gelu_forward_cpu(h_input, h_output_cpu, n);
    stopTime(&timer);
    printElapsedTime(timer, "Time to execute GELU kernel on CPU", CYAN);

    // 7. Compare GPU and CPU results
    if (compare_arrays(h_output_gpu, h_output_cpu, n)) {
        printf(GREEN "GPU and CPU versions match!\n" RESET);
    } else {
        printf(RED "GPU and CPU versions do not match!\n" RESET);
    }

    // Free host memory
    free(h_input);
    free(h_output_gpu);
    free(h_output_cpu);

    return 0;
}
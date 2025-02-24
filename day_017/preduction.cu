#include "timer.h"
#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_DIM 1024

__global__ void reduce_kernel(float *input, float *partialSums, unsigned int N) {
    unsigned int segment = blockIdx.x * blockDim.x * 2;
    unsigned int i = segment + threadIdx.x;
    
    if (i >= N) return;
    
    for(unsigned int stride = BLOCK_DIM; stride >= 1; stride /= 2) {
        if(threadIdx.x < stride && (i + stride) < N) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }
    
    if(threadIdx.x == 0) {
        partialSums[blockIdx.x] = input[i];
    }
}

// CPU reduction function for verification
float reduce_cpu(float *input, unsigned int N) {
    float result = 0.0f;
    for(unsigned int i = 0; i < N; i++) {
        result += input[i];
    }
    return result;
}

float reduce_gpu(float *input, unsigned int N) {
    Timer timer;
    float *d_input, *d_partialSums;
    float *h_partialSums;
    float result = 0.0f;
    
    unsigned int gridDim = (N + (BLOCK_DIM * 2) - 1) / (BLOCK_DIM * 2);
    
    startTime(&timer);
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_partialSums, gridDim * sizeof(float));
    h_partialSums = new float[gridDim];
    stopTime(&timer);
    printElapsedTime(timer, "GPU memory allocation time", CYAN);
    
    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    startTime(&timer);
    reduce_kernel<<<gridDim, BLOCK_DIM>>>(d_input, d_partialSums, N);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel execution time", GREEN);
    
    startTime(&timer);
    cudaMemcpy(h_partialSums, d_partialSums, gridDim * sizeof(float), cudaMemcpyDeviceToHost);
    stopTime(&timer);
    printElapsedTime(timer, "GPU to CPU copy time", YELLOW);
    
    for(unsigned int i = 0; i < gridDim; i++) {
        result += h_partialSums[i];
    }
    
    startTime(&timer);
    cudaFree(d_input);
    cudaFree(d_partialSums);
    delete[] h_partialSums;
    stopTime(&timer);
    printElapsedTime(timer, "Memory deallocation time", MAGENTA);
    
    return result;
}

int main() {
    const unsigned int N = 1 << 20; // 1M elements
    float *input = new float[N];
    
    // Initialize test data
    for(unsigned int i = 0; i < N; i++) {
        input[i] = static_cast<float>(i % 10); // More interesting test pattern: 0-9 repeating
    }
    
    Timer totalTimer;
    
    // CPU computation
    startTime(&totalTimer);
    float cpu_result = reduce_cpu(input, N);
    stopTime(&totalTimer);
    printElapsedTime(totalTimer, "CPU total execution time", BLUE);
    
    // GPU computation
    startTime(&totalTimer);
    float gpu_result = reduce_gpu(input, N);
    stopTime(&totalTimer);
    printElapsedTime(totalTimer, "GPU total execution time", WHITE);
    
    // Compare results
    std::cout << "CPU Result: " << cpu_result << std::endl;
    std::cout << "GPU Result: " << gpu_result << std::endl;
    
    // Check if results match within floating-point tolerance
    float tolerance = 1e-5;
    if (std::abs(cpu_result - gpu_result) < tolerance) {
        std::cout << GREEN << "Results match!" << RESET << std::endl;
    } else {
        std::cout << RED << "Results differ!" << RESET << std::endl;
        std::cout << "Difference: " << std::abs(cpu_result - gpu_result) << std::endl;
    }
    
    delete[] input;
    return 0;
}

// #include <timer.h>

// #define BLOCK_DIM 1024

// __global__ void reduce_kernel(float *input, float *partialSums, unsigned int N) {
//     unsigned int segment = blockIdx.x * blockDim.x * 2; // every thread block is responsible for twice the number of elements than its thread

//     unsigned int i = segment + threadIdx.x;
    
//     for(unsigned int stride = BLOCK_DIM; stride >= 1; stride /=2) {
//         if(threadIdx.x < stride) {
//             input[i] += input[i + stride];
//         }
//         __syncthreads();
//     }
//     if(threadIdx.x == 0) {
//         partialSums[blockIdx.x] = input[i];
//     }
// }

// float reduce_gpu(float *input, unsigned int N) {

// }

// int main() {
//     return 0;
// }
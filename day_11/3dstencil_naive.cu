#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "timer.h"

#define BLOCK_DIM 8
#define C0 0.5f
#define C1 0.5f

__global__ void stencil_kernel(float* in, float *out, unsigned int N) {
    unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;

    // boundary condition (only compute for inside output values)
    if (i>=1 && i<N-1 && j>=1 && j<N-1 && k>=1 && k<N-1) {
        out[(i*N+j)*N + k] = C0 * in[(i*N+j)*N + k] +
            C1 * (
                in[(i*N+j)*N + k-1]+
                in[(i*N+j)*N + k+1]+
                in[(i*N+(j-1))*N + k]+
                in[(i*N+(j+1))*N + k]+
                in[((i-1)*N+j)*N + k]+
                in[((i+1)*N+j)*N + k]
            );
    }
}

void stencil_cpu(float* in, float* out, unsigned int N) {
    Timer cpuTimer;
    startTime(&cpuTimer);

    for (unsigned int i = 1; i < N - 1; i++) {
        for (unsigned int j = 1; j < N - 1; j++) {
            for (unsigned int k = 1; k < N - 1; k++) {
                out[(i * N + j) * N + k] = C0 * in[(i * N + j) * N + k] +
                    C1 * (
                        in[(i * N + j) * N + k - 1] +
                        in[(i * N + j) * N + k + 1] +
                        in[(i * N + (j - 1)) * N + k] +
                        in[(i * N + (j + 1)) * N + k] +
                        in[((i - 1) * N + j) * N + k] +
                        in[((i + 1) * N + j) * N + k]
                    );
            }
        }
    }

    stopTime(&cpuTimer);
    printElapsedTime(cpuTimer, "CPU Stencil Time", WHITE);
}

void stencil_gpu(float* h_in, float* h_out, unsigned int N) {
    Timer timer;
    Timer overAllTimer;

    float *d_in, *d_out;
    size_t size = N * N * N * sizeof(float);

    startTime(&overAllTimer);

    // 1. Time required for GPU memory allocation
    startTime(&timer);
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU memory allocation time", RED);

    // 2. Time required to copy from host memory to GPU memory
    startTime(&timer);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Host to Device memory copy time", YELLOW);

    // Kernel launch configuration
    dim3 blockDim(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim((N + BLOCK_DIM - 1) / BLOCK_DIM, (N + BLOCK_DIM - 1) / BLOCK_DIM, (N + BLOCK_DIM - 1) / BLOCK_DIM);

    // 3. Time required to execute kernel
    startTime(&timer);
    stencil_kernel<<<gridDim, blockDim>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU Kernel execution time", GREEN);

    // 4. Time required to copy from device to host
    startTime(&timer);
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Device to Host memory copy time", MAGENTA);

    // 5. Time required for deallocation of GPU memory
    startTime(&timer);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU memory deallocation time", CYAN);

    // 6. Overall time required for GPU
    stopTime(&overAllTimer);
    printElapsedTime(overAllTimer, "Overall GPU time", BLUE);
}

int main() {
    unsigned int N = 512;
    size_t size = N * N * N * sizeof(float);

    // Allocate host memory
    float* h_in = (float*)malloc(size);
    float* h_out_gpu = (float*)malloc(size);
    float* h_out_cpu = (float*)malloc(size);

    // Initialize input with some values
    for (unsigned int i = 0; i < N * N * N; i++) {
        h_in[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Perform stencil operation on GPU
    stencil_gpu(h_in, h_out_gpu, N);

    // Perform stencil operation on CPU
    stencil_cpu(h_in, h_out_cpu, N);

    // Verify GPU and CPU results
    bool match = true;
    for (unsigned int i = 0; i < N * N * N; i++) {
        if (fabs(h_out_gpu[i] - h_out_cpu[i]) > 1e-5) {
            match = false;
            break;
        }
    }
    if (match) {
        printf(GREEN); // print in green color if cpu computation and gpu computation matches (color constants defined in timer.h file)
        printf("GPU and CPU results match!\n");
    } else {
        printf(RED);
        printf("GPU and CPU results do not match!\n");
    }

    // Free host memory
    free(h_in);
    free(h_out_gpu);
    free(h_out_cpu);

    return 0;
}
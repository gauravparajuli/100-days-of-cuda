#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "timer.h"

#define BLOCK_DIM 8
#define IN_TILE_DIM BLOCK_DIM
#define OUT_TILE_DIM (IN_TILE_DIM - 2)
#define C0 0.5f
#define C1 0.5f

__global__ void register_tiling_stencil_kernel(float* in, float* out, unsigned int N) {
    int iStart = blockIdx.z * OUT_TILE_DIM;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

    float inPrev;
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    float inNext;

    if(iStart - 1 >=0 && iStart - 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
        inPrev = in[(iStart - 1)* N * N + j * N +k];
    }
    if(iStart >= 0 && iStart < N && j >= 0 && j < N && k >=0 && k < N) {
        inCurr_s[threadIdx.y][threadIdx.x] = in[iStart * N * N + j*N +k];
    }

    __syncthreads();

    for(int i = iStart; i < iStart + OUT_TILE_DIM; i++) {
        if(i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
            inNext = in[(i + 1) * N * N + j*N +k];
        }

        __syncthreads();
        if(i >= 1 && i < N - 1 && j >=1 && j < N - 1 && k >= 1 && k < N - 1) {
            if(threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 && threadIdx.x >=1 && threadIdx.x < IN_TILE_DIM -1) {
                out[(i*N + j)*N + k] = C0 * inCurr_s[threadIdx.y][threadIdx.x] +
                            C1 * (
                                inCurr_s[threadIdx.y][threadIdx.x - 1] +
                                inCurr_s[threadIdx.y][threadIdx.x + 1] +
                                inCurr_s[threadIdx.y + 1][threadIdx.x] +
                                inCurr_s[threadIdx.y - 1][threadIdx.x] +
                                inPrev +
                                inNext
                            );
            }    
        }
        __syncthreads();
        inPrev = inCurr_s[threadIdx.y][threadIdx.x];
        inCurr_s[threadIdx.y][threadIdx.x] = inNext;
    }
}

__global__ void tiled_stencil_kernel(float* in, float *out, unsigned int N) {

    // we used BLOCK_DIM or IN_TILE_DIM to define blockDim. So, use OUT_TILE_DIM

    int i = blockIdx.z * OUT_TILE_DIM + threadIdx.z - 1;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

    __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];

    // at the boundary, make sure that I donot load out of bounds elements from input tile
    if (i >= 0 && i < N && j >=0 && j < N && k >= 0 && k < N) {
        in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[(i*N + j)* N + k]; 
    }
    __syncthreads();

    // boundary condition (only compute for inside output values)
    if (i>=1 && i<N-1 && j>=1 && j<N-1 && k>=1 && k<N-1) {
        
        // make sure only internal threads of thread block is active for computing output tile
        if(threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1 && threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 && threadIdx.z >= 1 && threadIdx.z < IN_TILE_DIM - 1)
            out[(i*N+j)*N + k] = C0 * in_s[ threadIdx.z][threadIdx.y][threadIdx.x] + 
                C1 * (
                    in_s[threadIdx.z][threadIdx.y][threadIdx.x - 1]+
                    in_s[threadIdx.z][threadIdx.y][threadIdx.x + 1]+
                    in_s[threadIdx.z][threadIdx.y - 1][threadIdx.x]+
                    in_s[threadIdx.z][threadIdx.y + 1][threadIdx.x]+
                    in_s[threadIdx.z - 1][threadIdx.y][threadIdx.x]+
                    in_s[threadIdx.z + 1][threadIdx.y][threadIdx.x]
                );
    }
}

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

void stencil_gpu(float* h_in, float* h_out_naive, float* h_out_tiled, float* h_out_reg_tiled, unsigned int N) {
    Timer timer;
    Timer overAllTimer;

    float *d_in, *d_out_naive, *d_out_tiled, *d_out_reg_tiled;
    size_t size = N * N * N * sizeof(float);

    startTime(&overAllTimer);

    // 1. Time required for GPU memory allocation
    startTime(&timer);
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out_naive, size);
    cudaMalloc((void**)&d_out_tiled, size);
    cudaMalloc((void**)&d_out_reg_tiled, size);
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

    // 3.1 Launch Naive 3D stencil kernel
    dim3 gridDim((N + BLOCK_DIM - 1) / BLOCK_DIM, (N + BLOCK_DIM - 1) / BLOCK_DIM, (N + BLOCK_DIM - 1) / BLOCK_DIM);
    startTime(&timer);
    stencil_kernel<<<gridDim, blockDim>>>(d_in, d_out_naive, N);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU Kernel (3D Stencil Naive) execution time", GREEN);

    // 3.2 Launch Tiled 3D stencil kernel (use tiling for optimization)
    // I will need number of blocks enough to cover output tiles. So, I will be using OUT_TIME_DIM to calculate grid size
    dim3 gridDimTiled((N + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM);
    startTime(&timer);
    tiled_stencil_kernel<<<gridDimTiled, blockDim>>>(d_in, d_out_tiled, N);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU Kernel (3D Stencil Tiled) execution time", GREEN);

    // 3.3 Launch Register tiled stencil kernel
    startTime(&timer);
    register_tiling_stencil_kernel<<<gridDimTiled, blockDim>>>(d_in, d_out_reg_tiled, N);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU Kernel (3D Stencil Register Tiled) execution time", GREEN);

    // 4. Time required to copy from device to host
    startTime(&timer);
    cudaMemcpy(h_out_naive, d_out_naive, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_tiled, d_out_tiled, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_reg_tiled, d_out_reg_tiled, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Device to Host memory copy time", MAGENTA);

    // 5. Time required for deallocation of GPU memory
    startTime(&timer);
    cudaFree(d_in);
    cudaFree(d_out_naive);
    cudaFree(d_out_tiled);
    cudaFree(d_out_reg_tiled);
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
    float* h_out_naive = (float*)malloc(size);
    float* h_out_tiled = (float*)malloc(size);
    float* h_out_reg_tiled = (float*)malloc(size);
    float* h_out_cpu = (float*)malloc(size);

    // Initialize input with some values
    for (unsigned int i = 0; i < N * N * N; i++) {
        h_in[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Perform stencil operation on GPU
    stencil_gpu(h_in, h_out_naive, h_out_tiled, h_out_reg_tiled, N);

    // Perform stencil operation on CPU
    stencil_cpu(h_in, h_out_cpu, N);

    bool match_naive = true;
    bool match_tiled = true;
    bool match_reg_tiled = true;

    for (unsigned int i = 0; i < N * N * N; i++) {
        if (fabs(h_out_naive[i] - h_out_cpu[i]) > 1e-5) {
            match_naive = false;
            break;
        }
    }
    for (unsigned int i = 0; i < N * N * N; i++) {
        if (fabs(h_out_tiled[i] - h_out_cpu[i]) > 1e-5) {
            match_tiled = false;
            break;
        }
    }

    for (unsigned int i = 0; i < N * N * N; i++) {
        if (fabs(h_out_reg_tiled[i] - h_out_cpu[i]) > 1e-5) {
            match_reg_tiled = false;
            break;
        }
    }

    printf(GREEN);
    if (match_naive) {
        printf("Naive GPU and CPU results match!\n");
    } else {
        printf(RED);
        printf("Naive GPU and CPU results do not match!\n");
    }
    printf(GREEN);
    if (match_tiled) {
        printf("Tiled GPU and CPU results match!\n");
    } else {
        printf(RED);
        printf("Tiled GPU and CPU results do not match!\n");
    }

    if (match_reg_tiled) {
        printf("Register Tiled GPU and CPU results match!\n");
    } else {
        printf(RED);
        printf("Register Tiled GPU and CPU results do not match!\n");
    }

    free(h_in);
    // FIXED: Free both naive and tiled host arrays
    free(h_out_naive);
    free(h_out_tiled);
    free(h_out_reg_tiled);
    free(h_out_cpu);

    return 0;
}
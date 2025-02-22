#include <iostream>
#include "timer.h"

#define BLOCK_DIM 32
#define TILE_DIM 32
#define COARSE_FACTOR 4 // one thread block will be responsible for computing 4 horizontally adjacent tiles in output matrix

__shared__ int A_S[TILE_DIM][TILE_DIM];
__shared__ int B_S[TILE_DIM][TILE_DIM];

// CPU matrix multiplication
void cpu_mmul(int* a, int* b, int* c, unsigned int N) {
    for (unsigned int i = 0; i < N; i++) {
        for (unsigned int j = 0; j < N; j++) {
            int sum = 0;
            for (unsigned int k = 0; k < N; k++) {
                sum += a[i * N + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

// Verify results - updated verification function
bool verify_results(int* cpu, int* naive, int* tiled, int* tiled_opt, unsigned int N) {    // Changed opt to tiled_opt
    for (unsigned int i = 0; i < N * N; i++) {
        if (cpu[i] != naive[i] || cpu[i] != tiled[i] || cpu[i] != tiled_opt[i]) {    // Changed opt to tiled_opt
            std::cout << "Mismatch at index " << i << ": CPU=" << cpu[i] 
                     << ", Naive=" << naive[i] << ", Tiled=" << tiled[i]
                     << ", Optimized=" << tiled_opt[i] << std::endl;    // Changed opt to tiled_opt
            return false;
        }
    }
    return true;
}

// // Verification function
// bool verify_results(int* cpu, int* naive, int* tiled, unsigned int N) {
//     for (unsigned int i = 0; i < N * N; i++) {
//         if (cpu[i] != naive[i] || cpu[i] != tiled[i]) {
//             std::cout << "Mismatch at index " << i << ": CPU=" << cpu[i] 
//                      << ", Naive=" << naive[i] << ", Tiled=" << tiled[i] << std::endl;
//             return false;
//         }
//     }
//     return true;
// }

__global__ void naive_mmul_kernel(int* a, int* b, int* c, unsigned int N) {
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;

    int temp_sum = 0;
    // boundary protection
    if ((outRow < N) && (outCol < N)) { // n is the MATRIX size along single dimension
        for (int k = 0; k < N; k++) {
            // accumulate result for a single element
            temp_sum += a[outRow * N + k] * b[k * N + outCol];
        }
        c[outRow*N+outCol] = temp_sum;
    }
}

__global__ void tiled_mmul_kernel(int *a, int *b, int *c, unsigned int N) {

    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;

    int sum = 0;

    for (unsigned int tile=0; tile < (N / TILE_DIM); tile++) {
        A_S[threadIdx.y][threadIdx.x] = a[outRow*N + tile * TILE_DIM + threadIdx.x];
        B_S[threadIdx.y][threadIdx.x] = b[(TILE_DIM * tile + threadIdx.y)*N + outCol];

        __syncthreads(); // wait till all the threads load data into the shared memory

        // multiply the tiles now
        for (unsigned int i = 0; i < TILE_DIM; i++) {
            sum += A_S[threadIdx.y][i] * B_S[i][threadIdx.x];
        }
        __syncthreads(); // make sure all threads finish their computation before we load the next set of values

    }

    c[outRow * N + outCol] = sum;
}

__global__ void optimized_tiled_mmul_kernel(int *a, int *b, int *c, unsigned int N) {
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int colStart = blockIdx.x * blockDim.x  * COARSE_FACTOR + threadIdx.x;

    int sum[COARSE_FACTOR];
    for(unsigned int c=0; c < COARSE_FACTOR; c++) {
        sum[c] = 0;
    }

    for (unsigned int tile=0; tile < (N / TILE_DIM); tile++) {
        A_S[threadIdx.y][threadIdx.x] = a[outRow*N + tile * TILE_DIM + threadIdx.x];
        
        // for every tile of A, we have to load multiple tiles of B
        for (unsigned int c = 0;  c < COARSE_FACTOR; c++) {

            unsigned int outCol = colStart + c*TILE_DIM;

            B_S[threadIdx.y][threadIdx.x] = b[(TILE_DIM * tile + threadIdx.y)*N + outCol];

            __syncthreads(); // wait till all the threads load data into the shared memory
    
            // multiply the tiles now
            for (unsigned int i = 0; i < TILE_DIM; i++) {
                sum[c] += A_S[threadIdx.y][i] * B_S[i][threadIdx.x];
            }
            __syncthreads(); // make sure all threads finish their computation before we load the next set of values
        }
    }


    for (unsigned int i = 0; i < COARSE_FACTOR; i++) {
        unsigned int outCol = colStart + i*TILE_DIM;
        c[outRow*N + outCol] = sum[i]; 
    }
}   

int main() {
    const unsigned int N = 1024; // Matrix size along single dimension (must be multiple of TILE_DIM)
    size_t size = N * N * sizeof(int); // we are assuming a square matrix
    
    // Host arrays
    int *h_a = (int*)malloc(size);
    int *h_b = (int*)malloc(size);
    int *h_c_naive = (int*)malloc(size);
    int *h_c_tiled = (int*)malloc(size);
    int *h_c_tiled_opt = (int*)malloc(size);    // Changed from h_c_opt to h_c_tiled_opt
    int *h_c_cpu = (int*)malloc(size);

    // Device arrays
    int *d_a, *d_b, *d_c_naive, *d_c_tiled, *d_c_tiled_opt;    // Changed d_c_opt to d_c_tiled_opt
    
    // Initialize input matrices with test data
    for (unsigned int i = 0; i < N * N; i++) {
        h_a[i] = rand() % 100;
        h_b[i] = rand() % 100;
    }

    Timer timer;
    
    // 1. GPU Memory Allocation
    startTime(&timer);
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c_naive, size);
    cudaMalloc(&d_c_tiled, size);
    cudaMalloc(&d_c_tiled_opt, size);    // Changed from d_c_opt to d_c_tiled_opt
    stopTime(&timer);
    printElapsedTime(timer, "GPU Memory Allocation", BLUE);

    // 2. Host to GPU Transfer
    startTime(&timer);
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);    // Fixed typo from previous version
    stopTime(&timer);
    printElapsedTime(timer, "Host to GPU Transfer", BLUE);

    // 3. Kernel Executions
    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim((N + BLOCK_DIM - 1) / BLOCK_DIM, (N + BLOCK_DIM - 1) / BLOCK_DIM);
    dim3 gridDimOpt((N + BLOCK_DIM - 1) / BLOCK_DIM/COARSE_FACTOR, (N + BLOCK_DIM - 1) / BLOCK_DIM);
    
    // Naive Kernel
    startTime(&timer);
    naive_mmul_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_c_naive, N);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Naive Kernel Execution", GREEN);

    // Tiled Kernel
    startTime(&timer);
    tiled_mmul_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_c_tiled, N);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Tiled Kernel Execution", GREEN);

    // Optimized Tiled Kernel
    startTime(&timer);
    optimized_tiled_mmul_kernel<<<gridDimOpt, blockDim>>>(d_a, d_b, d_c_tiled_opt, N);    // Changed d_c_opt to d_c_tiled_opt
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Optimized Tiled Kernel Execution", GREEN);

    // 4. GPU to Host Transfer
    startTime(&timer);
    cudaMemcpy(h_c_naive, d_c_naive, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c_tiled, d_c_tiled, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c_tiled_opt, d_c_tiled_opt, size, cudaMemcpyDeviceToHost);    // Changed h_c_opt to h_c_tiled_opt
    stopTime(&timer);
    printElapsedTime(timer, "GPU to Host Transfer", BLUE);

    // 5. CPU Execution
    startTime(&timer);
    cpu_mmul(h_a, h_b, h_c_cpu, N);
    stopTime(&timer);
    printElapsedTime(timer, "CPU Execution", YELLOW);

    // 6. GPU Memory Deallocation
    startTime(&timer);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c_naive);
    cudaFree(d_c_tiled);
    cudaFree(d_c_tiled_opt);    // Changed d_c_opt to d_c_tiled_opt
    stopTime(&timer);
    printElapsedTime(timer, "GPU Memory Deallocation", BLUE);

    bool verified = verify_results(h_c_cpu, h_c_naive, h_c_tiled, h_c_tiled_opt, N);    // Changed h_c_opt to h_c_tiled_opt
    std::cout << (verified ? GREEN : RED) << "Results " 
              << (verified ? "match!" : "do not match!") << RESET << std::endl;

    // Cleanup
    free(h_a);
    free(h_b);
    free(h_c_naive);
    free(h_c_tiled);
    free(h_c_tiled_opt);    // Changed h_c_opt to h_c_tiled_opt
    free(h_c_cpu);

    return 0;
}

// int main() {
//     const unsigned int N = 1024; // Matrix size along single dimension (must be multiple of TILE_DIM)
//     size_t size = N * N * sizeof(int); // we are assuming a square matrix
    
//     // Host arrays
//     int *h_a = (int*)malloc(size);
//     int *h_b = (int*)malloc(size);
//     int *h_c_naive = (int*)malloc(size);
//     int *h_c_tiled = (int*)malloc(size);
//     int *h_c_cpu = (int*)malloc(size);

//     // Device arrays
//     int *d_a, *d_b, *d_c_naive, *d_c_tiled;
    
//     // Initialize input matrices with test data
//     for (unsigned int i = 0; i < N * N; i++) {
//         h_a[i] = rand() % 100;
//         h_b[i] = rand() % 100;
//     }

//     Timer timer;
    
//     // 1. GPU Memory Allocation
//     startTime(&timer);
//     cudaMalloc(&d_a, size);
//     cudaMalloc(&d_b, size);
//     cudaMalloc(&d_c_naive, size);
//     cudaMalloc(&d_c_tiled, size);
//     stopTime(&timer);
//     printElapsedTime(timer, "GPU Memory Allocation", BLUE);

//     // 2. Host to GPU Transfer
//     startTime(&timer);
//     cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
//     stopTime(&timer);
//     printElapsedTime(timer, "Host to GPU Transfer", BLUE);

//     // 3. Kernel Executions
//     dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
//     dim3 gridDim((N + BLOCK_DIM - 1) / BLOCK_DIM, (N + BLOCK_DIM - 1) / BLOCK_DIM);
    
//     // Naive Kernel
//     startTime(&timer);
//     naive_mmul_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_c_naive, N);
//     cudaDeviceSynchronize();
//     stopTime(&timer);
//     printElapsedTime(timer, "Naive Kernel Execution", GREEN);

//     // Tiled Kernel
//     startTime(&timer);
//     tiled_mmul_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_c_tiled, N);
//     cudaDeviceSynchronize();
//     stopTime(&timer);
//     printElapsedTime(timer, "Tiled Kernel Execution", GREEN);

//     // 4. GPU to Host Transfer
//     startTime(&timer);
//     cudaMemcpy(h_c_naive, d_c_naive, size, cudaMemcpyDeviceToHost);
//     cudaMemcpy(h_c_tiled, d_c_tiled, size, cudaMemcpyDeviceToHost);
//     stopTime(&timer);
//     printElapsedTime(timer, "GPU to Host Transfer", BLUE);

//     // 5. CPU Execution
//     startTime(&timer);
//     cpu_mmul(h_a, h_b, h_c_cpu, N);
//     stopTime(&timer);
//     printElapsedTime(timer, "CPU Execution", YELLOW);

//     // 6. GPU Memory Deallocation
//     startTime(&timer);
//     cudaFree(d_a);
//     cudaFree(d_b);
//     cudaFree(d_c_naive);
//     cudaFree(d_c_tiled);
//     stopTime(&timer);
//     printElapsedTime(timer, "GPU Memory Deallocation", BLUE);

//     // Verify results
//     bool verified = verify_results(h_c_cpu, h_c_naive, h_c_tiled, N);
//     std::cout << (verified ? GREEN : RED) << "Results " 
//               << (verified ? "match!" : "do not match!") << RESET << std::endl;

//     // Cleanup
//     free(h_a);
//     free(h_b);
//     free(h_c_naive);
//     free(h_c_tiled);
//     free(h_c_cpu);

//     return 0;
// }
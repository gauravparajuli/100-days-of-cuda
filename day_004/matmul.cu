#include <cuda_runtime.h>
// #include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

__global__ void matrixMul(int* a, int* b, int* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int temp_sum = 0;
    // boundary protection
    if ((row < n) && (col < n)) {
        for (int k = 0; k < n; k++) {
            // accumulate result for a single element
            temp_sum += a[row * n + k] * b[k * n + col];
        }
        c[row*n+col] = temp_sum;
    }
}

void init_matrices(int *a, int *b, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i * n + j] = rand() % 100;
            b[i * n + j] = rand() % 100;
        }
    }
}

// check result
void verify_result(int* a, int* b, int *c, int n) {
    int *verify_c;

    verify_c = (int*)malloc(n * n * sizeof(int));

    // serial implementation of matix multiplication
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                verify_c[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }
    
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            assert(c[i * n + j] == verify_c[i * n + j]);
        }
    }
}

int main() {
    int n = 1 << 10; // 1024

    size_t bytes = n * n * sizeof(int);

    // host pointers
    int *h_a, *h_b, *h_c;

    // allocate host memory
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);

    // device pointers
    int *d_a, *d_b, *d_c;

    // allocate device memory
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // initialize matrices
    init_matrices(h_a, h_b, n);

    // copy data to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // threads per block
    int BLOCK_SIZE=16;

    // blocks in each dimension
    int GRID_SIZE=(int)ceil(n / BLOCK_SIZE);

    // use dim3 objects
    dim3 grid(GRID_SIZE, GRID_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    matrixMul <<<grid, threads>>>(d_a, d_b, d_c, n);

    // copy back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // check result
    verify_result(h_a, h_b, h_c, n);

    printf("completed");

    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}
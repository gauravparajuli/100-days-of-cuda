#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define TILE_WIDTH 16

__global__ void tiledMatrixMul(int m, int n, int k, int *a, int *b, int *c) {

    __shared__ int ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ int ds_B[TILE_WIDTH][TILE_WIDTH];

    // details regarding this thread
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    // parallel matmul
    float Cvalue = 0;

    for(int t=0; t < n/TILE_WIDTH; ++t) {
        // collaborative loading of A and B tiles into shared memory
        ds_A[ty][tx] = a[row * n + t * TILE_WIDTH + tx];
        ds_B[ty][tx] = b[(t * TILE_WIDTH + ty) * k + col];

        __syncthreads();

        for (int i=0; i < TILE_WIDTH; ++i) {
            Cvalue += ds_A[ty][i] * ds_B[i][tx];
        }
        __syncthreads();
    }

    c[row * k + col] = Cvalue;

}

// Function to perform matrix multiplication on the CPU
void matrixMulCPU(int m, int n, int k, int *a, int *b, int *c) {
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < k; col++) {
            int sum = 0;
            for (int t = 0; t < n; t++) {
                sum += a[row * n + t] * b[t * k + col];
            }
            c[row * k + col] = sum;
        }
    }
}

int main() {
    // Define matrix dimensions
    int m = 1024; // Number of rows in matrix A
    int n = 1024; // Number of columns in matrix A and rows in matrix B
    int k = 1024; // Number of columns in matrix B

    // Allocate host memory for matrices A, B, and C
    size_t size_A = m * n * sizeof(int);
    size_t size_B = n * k * sizeof(int);
    size_t size_C = m * k * sizeof(int);

    int *h_A = (int *)malloc(size_A);
    int *h_B = (int *)malloc(size_B);
    int *h_C = (int *)malloc(size_C);
    int *h_C_CPU = (int *)malloc(size_C); // For CPU result

    // Initialize matrices A and B with some values
    for (int i = 0; i < m * n; i++) h_A[i] = rand() % 100;
    for (int i = 0; i < n * k; i++) h_B[i] = rand() % 100;

    // Allocate device memory for matrices A, B, and C
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((k + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);

    // Launch the kernel
    tiledMatrixMul<<<dimGrid, dimBlock>>>(m, n, k, d_A, d_B, d_C);

    // Copy the result matrix C from device to host
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Perform matrix multiplication on the CPU for verification
    matrixMulCPU(m, n, k, h_A, h_B, h_C_CPU);

    // Verify the results using assert
    for (int i = 0; i < m * k; i++) {
        assert(h_C[i] == h_C_CPU[i]);
    }

    printf("Matrix multiplication results match!\n");

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

__global__ void vectorAdd(int* a, int* b, int* c, int n) {
    // calculate global thread id
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid]; // each thread adds a single element
    }
}

// intialize vector of size n between 0-99
void matrix_init(int* a, int n) {
    for (int i=0; i < n; i++) {
        a[i] = rand() % 100;
    }
}

void error_check(int* a, int *b, int* c, int n) {
    for (int i = 0; i < n; i++) {
        assert(c[i] == a[i] + b[i]);
    }
}

int main() {
    int n = 1 << 16; // 2^16
    int *h_a, *h_b, *h_c; // host pointers
    int *d_a, *d_b, *d_c; // device pointers

    // allocation size for all vectors
    size_t bytes = sizeof(int) * n;

    // alocate host memory
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);

    // allocate device memory (provide pointers to pointers)
    cudaMalloc(&d_a, bytes); 
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    matrix_init(h_a, n);
    matrix_init(h_b, n);

    // copy data from
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // threadblock size (will be translated to warps of size 32, so always a multiple of 32)
    int NUM_THREADS = 256; 

    // grid size
    int NUM_BLOCKS = (int)ceil(n / NUM_THREADS);

    // launch kernel
    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, n);

    // copy sum vector from device to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // check results for errors
    error_check(h_a, h_b, h_c, n);

    printf("completed!");

    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}
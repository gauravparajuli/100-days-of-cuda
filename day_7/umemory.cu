#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(int *a, int *b, int *c, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid<N) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    const int N = 1 << 16;
    size_t bytes = N * sizeof(int);

    // get device id
    int id = cudaGetDevice(&id);

    // declare unified memory pointers
    int *a, *b, *c;

    // allocate memory for these pointers (unified memory)
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // initialize vectors
    for (int i=0; i < N; i++) {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }

    int BLOCK_SIZE = 1 << 10; // 1024 thread pers cta

    int GRID_SIZE = (int)ceil(N / BLOCK_SIZE);

    // prefetching data to device
    cudaMemPrefetchAsync(a, bytes, id);
    cudaMemPrefetchAsync(b, bytes, id);

    vectorAdd<<<GRID_SIZE, BLOCK_SIZE>>>(a,b,c, N);

    cudaDeviceSynchronize(); // wait for all previous operations/events> to complete

    cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId); // prefetch c back to host

    // verify result on the cpu
    for (int i = 0; i < N; i++) {
        assert(c[i] == a[i]+b[i]);
    }

    // free unified memory
    cudaFree(a); cudaFree(b); cudaFree(c);

    printf("completed successfully!\n");

    return 0;

}
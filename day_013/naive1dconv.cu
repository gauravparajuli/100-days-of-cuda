#include <stdio.h>
#include <cuda_runtime.h>
#include <assert.h>
#include "timer.h"

__global__ void naive1dconv_kernel(int *array, int *mask, int *result, int n, int m) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    const int r = m / 2;
    
    int sum = 0;

    int start = tid - r;

    if (tid < n)
        for (int j=0; j < m; j++) 
            if (start+j >= 0 && start+j<n)
                sum+= mask[j] * array[start+j];
        

        result[tid] = sum;

}

// Verify the result on the CPU
void verify_result(int *array, int *mask, int *result, int n, int m){
    int radius = m / 2;
    int temp;
    int start;
    for(int i = 0; i < n; i++){
        start = i - radius;
        temp = 0;
        for(int j = 0; j < m; j++){
            if((start + j >= 0) && (start + j < n)){
                temp += array[start + j] * mask[j];
            }
        }
        assert(temp == result[i]);
    }
}


int main() {
    // number of the elements in input array
    int n = 1 << 20; // 2 to the power 20

    // size of the input array in bytes
    size_t bytes_n = n * sizeof(int);

    // number of elements in 1d convolution mask
    int m = 7;

    // size of convolution mask in bytes
    int bytes_m = m * sizeof(int);

    // allocate the array
    int *h_array = new int[n];

    // initialize it
    for(int i = 0; i < n; i++)
        h_array[i] = rand() % 100;

    // allocate the mask and initialize it
    int *h_mask = new int[m];
    for (int i = 0; i < m; i++)
        h_mask[i] = rand() % 10;

    // allocate the space for final result
    int *h_result = new int[n];

    // allocate the space on the gpu
    int *d_array, *d_mask, *d_result;
    cudaMalloc((void**)&d_array, bytes_n);
    cudaMalloc((void**)&d_mask, bytes_m);
    cudaMalloc((void**)&d_result, bytes_n);

    // copy data to device
    cudaMemcpy(d_array, h_array, bytes_n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, bytes_m, cudaMemcpyHostToDevice);

    const int threadsPerBlock = 256;
    const int numBlocks = (n + threadsPerBlock -1) / threadsPerBlock;

    naive1dconv_kernel<<<numBlocks, threadsPerBlock>>>(d_array, d_mask, d_result, n, m);

    // copy back the result
    cudaMemcpy(h_result, d_result, bytes_n, cudaMemcpyDeviceToHost);

    // verify the result
    verify_result(h_array, h_mask, h_result, n, m);

    printf("cpu computation and gpu computation matches!\n");

    // Free memory
    delete[] h_array;
    delete[] h_mask;
    delete[] h_result;
    cudaFree(d_array);
    cudaFree(d_mask);
    cudaFree(d_result);

    return 0;
}
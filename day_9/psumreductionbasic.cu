#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void sumReduction(float* input, float* output, int n) {
    __shared__ float partialSum[2*BLOCK_SIZE];

    unsigned int t = threadIdx.x;
    unsigned int start =  blockIdx.x * 2 * blockDim.x;
    partialSum[t] = input[start + t];
    partialSum[blockDim+t] = input[start + blockDim.x+t];
    
    for (unsigned int stride=1; stride <= blockDim.x; stride *= 2) {
        __syncthread(); // ensure we have partial sum from all the threads before continuing
        if (t % stride == 0)
            partialSum[2*t] += partialSum[2*t+stride];
    }
}


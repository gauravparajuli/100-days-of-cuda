#include <stdio.h>

__global__ void helloCUDA() {
    printf("hello from CUDA!\n");
}

int main() {
    helloCUDA<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
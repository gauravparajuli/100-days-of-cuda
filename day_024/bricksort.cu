#include <iostream>
#include <cuda_runtime.h>

#define N 100  // Size of the array

__global__ void oddEvenSort(int *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int temp;

    for (int phase = 0; phase < n; ++phase) {
        // Even phase
        if (phase % 2 == 0) {
            if (idx % 2 == 0 && idx < n - 1) {
                if (data[idx] > data[idx + 1]) {
                    temp = data[idx];
                    data[idx] = data[idx + 1];
                    data[idx + 1] = temp;
                }
            }
        }
        // Odd phase
        else {
            if (idx % 2 == 1 && idx < n - 1) {
                if (data[idx] > data[idx + 1]) {
                    temp = data[idx];
                    data[idx] = data[idx + 1];
                    data[idx + 1] = temp;
                }
            }
        }
        __syncthreads();
    }
}

int main() {
    int h_data[N];
    int *d_data;

    // Initialize the array with random values
    for (int i = 0; i < N; ++i) {
        h_data[i] = rand() % 100;
    }

    // Allocate memory on the device
    cudaMalloc((void**)&d_data, N * sizeof(int));

    // Copy data to the device
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

    // Define block size and grid size
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Launch the kernel
    oddEvenSort<<<gridSize, blockSize>>>(d_data, N);

    // Copy the sorted array back to the host
    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the sorted array
    for (int i = 0; i < N; ++i) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_data);

    return 0;
}
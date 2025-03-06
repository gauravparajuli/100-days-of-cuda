#include <iostream>

// Layer normalization kernel
__global__ void layerNormKernel(float* input, float* output, 
                               float* gamma, float* beta,
                               int batchSize, int layerSize,
                               float epsilon = 1e-5f) {
    // Calculate global thread index
    int batchIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int layerIdx = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if thread is within bounds
    if (batchIdx >= batchSize || layerIdx >= layerSize) {
        return;
    }

    // Shared memory for mean and variance calculation
    extern __shared__ float sharedMem[];
    float* meanShared = sharedMem;
    float* varShared = &sharedMem[blockDim.y];

    // Calculate mean
    float sum = 0.0f;
    for (int i = threadIdx.y; i < layerSize; i += blockDim.y) {
        sum += input[batchIdx * layerSize + i];
    }
    meanShared[threadIdx.y] = sum;
    __syncthreads();

    // Reduce sum for mean
    for (int stride = blockDim.y / 2; stride > 0; stride /= 2) {
        if (threadIdx.y < stride) {
            meanShared[threadIdx.y] += meanShared[threadIdx.y + stride];
        }
        __syncthreads();
    }

    float mean = meanShared[0] / layerSize;

    // Calculate variance
    sum = 0.0f;
    for (int i = threadIdx.y; i < layerSize; i += blockDim.y) {
        float diff = input[batchIdx * layerSize + i] - mean;
        sum += diff * diff;
    }
    varShared[threadIdx.y] = sum;
    __syncthreads();

    // Reduce sum for variance
    for (int stride = blockDim.y / 2; stride > 0; stride /= 2) {
        if (threadIdx.y < stride) {
            varShared[threadIdx.y] += varShared[threadIdx.y + stride];
        }
        __syncthreads();
    }

    float variance = varShared[0] / layerSize;
    float stdDev = sqrtf(variance + epsilon);

    // Normalize and apply scale/shift
    int idx = batchIdx * layerSize + layerIdx;
    float normalized = (input[idx] - mean) / stdDev;
    output[idx] = normalized * gamma[layerIdx] + beta[layerIdx];
}

// Host function to launch the kernel
void launchLayerNorm(float* d_input, float* d_output, 
                    float* d_gamma, float* d_beta,
                    int batchSize, int layerSize) {
    // Define block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (batchSize + blockDim.x - 1) / blockDim.x,
        (layerSize + blockDim.y - 1) / blockDim.y
    );

    // Calculate shared memory size (for mean and variance)
    size_t sharedMemSize = 2 * blockDim.y * sizeof(float);

    // Launch kernel
    layerNormKernel<<<gridDim, blockDim, sharedMemSize>>>(
        d_input, d_output, d_gamma, d_beta, 
        batchSize, layerSize
    );

    cudaDeviceSynchronize();
}

// Example usage
int main() {
    const int batchSize = 32;
    const int layerSize = 768;

    // Allocate and initialize host memory
    float* h_input = new float[batchSize * layerSize];
    float* h_output = new float[batchSize * layerSize];
    float* h_gamma = new float[layerSize];
    float* h_beta = new float[layerSize];

    // Initialize gamma to 1 and beta to 0
    for (int i = 0; i < layerSize; i++) {
        h_gamma[i] = 1.0f;
        h_beta[i] = 0.0f;
    }

    // Allocate device memory
    float *d_input, *d_output, *d_gamma, *d_beta;
    cudaMalloc(&d_input, batchSize * layerSize * sizeof(float));
    cudaMalloc(&d_output, batchSize * layerSize * sizeof(float));
    cudaMalloc(&d_gamma, layerSize * sizeof(float));
    cudaMalloc(&d_beta, layerSize * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, h_input, 
               batchSize * layerSize * sizeof(float), 
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, 
               layerSize * sizeof(float), 
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, 
               layerSize * sizeof(float), 
               cudaMemcpyHostToDevice);

    // Launch kernel
    launchLayerNorm(d_input, d_output, d_gamma, d_beta, 
                   batchSize, layerSize);

    // Copy results back to host
    cudaMemcpy(h_output, d_output, 
               batchSize * layerSize * sizeof(float), 
               cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    delete[] h_input;
    delete[] h_output;
    delete[] h_gamma;
    delete[] h_beta;

    return 0;
}
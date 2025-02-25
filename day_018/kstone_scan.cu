#include "timer.h"

#define BLOCK_DIM 1024

void scan_cpu(float* input, float* output, unsigned int N) {
    output[0] = input[0];
    for (unsigned int i = 1; i < N; i++) {
        output[i] = output[i-1] + input[i];
    }
}

bool verify_results(float* cpu_result, float* gpu_result, unsigned int N) {
    const float epsilon = 1e-5;
    for (unsigned int i = 0; i < N; i++) {
        if (abs(cpu_result[i] - gpu_result[i]) > epsilon) {
            std::cout << RED << "Verification failed at index " << i 
                      << ": CPU = " << cpu_result[i] 
                      << ", GPU = " << gpu_result[i] << RESET << std::endl;
            return false;
        }
    }
    return true;
}

__global__ void scan_partials_kernel(float *partialSums, unsigned int numBlocks) {
    unsigned int i = threadIdx.x;

    if (i >= numBlocks) return;

    // perform inclusive scan on the partialSums array
    for(unsigned int stride=1; stride <= numBlocks / 2; stride *= 2) {
        float temp = 0;
        if (i >= stride)
            temp = partialSums[i - stride];

        __syncthreads();

        if (i >= stride)
            partialSums[i] += temp;

        __syncthreads();
    }

}

__global__ void scan_kernel_shared_memory(float *input, float *output, float *partialSums, unsigned int N) {

    // global index of thread (in grid)
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float buffer_s[BLOCK_DIM];
    buffer_s[threadIdx.x] = input[i];
    __syncthreads();

    if (i >= N) return; // donot let thread access out of bounds memory

    for(unsigned int stride = 1; stride <= (BLOCK_DIM / 2); stride *= 2) {
        float temp;
        if (threadIdx.x >= stride)  // use local index of thread because we are approaching by segmented/hierarchial scan
            temp = buffer_s[threadIdx.x - stride];
        
        __syncthreads();

        if (threadIdx.x >= stride)
            buffer_s[threadIdx.x] += temp;
         
        __syncthreads();

    }

    // last thread in a block is responsible for allocating partial sums
    if (threadIdx.x == BLOCK_DIM - 1)
        partialSums[blockIdx.x] = buffer_s[threadIdx.x];

    output[i] = buffer_s[threadIdx.x];

}

__global__ void scan_kernel(float *input, float *output, float *partialSums, unsigned int N) {

    // global index of thread (in grid)
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N) return; // donot let thread access out of bounds memory

    // move value from input array to output array
    output[i] = input[i];
    __syncthreads();

    for(unsigned int stride = 1; stride <= (BLOCK_DIM / 2); stride *= 2) {
        float temp;
        if (threadIdx.x >= stride)  // use local index of thread because we are approaching by segmented/hierarchial scan
            temp = output[i - stride];
        
        __syncthreads();

        if (threadIdx.x >= stride)
            output[i] += temp;
         
        __syncthreads();
    }

    // last thread in a block is responsible for allocating partial sums
    if (threadIdx.x == BLOCK_DIM - 1)
        partialSums[blockIdx.x] = output[i];

}

__global__ void add_kernel(float *output, float *partialSums, unsigned int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (blockIdx.x > 0) {
        output[i] += partialSums[blockIdx.x - 1];
    }
}

int main() {
    const unsigned int N = 1 << 20; // Example size: 1M elements
    const unsigned int BLOCKS = (N + BLOCK_DIM - 1) / BLOCK_DIM;
    const size_t bytes = N * sizeof(float);
    const size_t partial_bytes = BLOCKS * sizeof(float);

    // Host arrays
    float *h_input = new float[N];
    float *h_output_gpu = new float[N];
    float *h_output_cpu = new float[N];

    // Device arrays
    float *d_input, *d_output, *d_partialSums;

    // Initialize input data
    srand(time(nullptr));
    for (unsigned int i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(rand() % 10); // Random numbers 0-9
    }

    Timer timer;

    // Allocate device memory
    startTime(&timer);
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMalloc(&d_partialSums, partial_bytes);
    stopTime(&timer);
    printElapsedTime(timer, "Memory allocation time", CYAN);

    // 1. Time to copy from host to device
    startTime(&timer);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    stopTime(&timer);
    printElapsedTime(timer, "Host to Device Copy Time", CYAN);

    // 2. GPU kernel execution

    Timer gpuTimer;

    startTime(&gpuTimer);
    startTime(&timer);
    scan_kernel<<<BLOCKS, BLOCK_DIM>>>(d_input, d_output, d_partialSums, N);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU (Kogge Stone Scan Kernel) Execution Time", GREEN);
    startTime(&timer);
    scan_partials_kernel<<<1, BLOCKS>>>(d_partialSums, BLOCKS);
    cudaDeviceSynchronize();
    add_kernel<<<BLOCKS, BLOCK_DIM>>>(d_output, d_partialSums, N);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU (Addition Kernel) Execution Time", GREEN);
    stopTime(&gpuTimer);
    printElapsedTime(gpuTimer, "Overall GPU Execution Time", GREEN);

    // 3. Time to copy from device to host
    startTime(&timer);
    cudaMemcpy(h_output_gpu, d_output, bytes, cudaMemcpyDeviceToHost);
    stopTime(&timer);
    printElapsedTime(timer, "Device to Host Copy Time", CYAN);

    // 4. CPU computation
    startTime(&timer);
    scan_cpu(h_input, h_output_cpu, N);
    stopTime(&timer);
    printElapsedTime(timer, "CPU Computation Time", YELLOW);

    // Verify results
    if (verify_results(h_output_cpu, h_output_gpu, N)) {
        std::cout << GREEN << "Verification successful: CPU and GPU results match!" << RESET << std::endl;
    } else {
        std::cout << RED << "Verification failed!" << RESET << std::endl;
    }

    // 5. Time to deallocate memory
    startTime(&timer);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_partialSums);
    delete[] h_input;
    delete[] h_output_gpu;
    delete[] h_output_cpu;
    stopTime(&timer);
    printElapsedTime(timer, "Memory Deallocation Time", MAGENTA);

    return 0;
}
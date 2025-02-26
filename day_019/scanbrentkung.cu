#include "timer.h"
#include <vector>
#include <random>
#include <cuda_runtime.h>

#define BLOCK_DIM 1024

// brent kung scanning

// CPU sequential scan implementation
void scan_cpu(const float* input, float* output, unsigned int N) {
    output[0] = input[0];
    for (unsigned int i = 1; i < N; i++) {
        output[i] = output[i-1] + input[i];
    }
}

// Function to verify results
bool verify_results(const float* cpu_result, const float* gpu_result, unsigned int N, float epsilon = 1e-5) {
    for (unsigned int i = 0; i < N; i++) {
        if (fabs(cpu_result[i] - gpu_result[i]) > epsilon) {
            std::cout << RED << "Mismatch at index " << i 
                      << ": CPU = " << cpu_result[i] 
                      << ", GPU = " << gpu_result[i] << RESET << std::endl;
            return false;
        }
    }
    return true;
}

__global__ void scan_kernel(float *input, float *output, float* partialSums, unsigned int N) {
    unsigned int segment = blockIdx.x * blockDim.x * 2; // twice as many as input elements as number of threads

    __shared__ float buffer_s[2*BLOCK_DIM]; // twice as many input elements as number of threads
    buffer_s[threadIdx.x] = input[segment + threadIdx.x];
    buffer_s[threadIdx.x + BLOCK_DIM] = input[segment + threadIdx.x + BLOCK_DIM];
    __syncthreads();

    // reduction step
    for(unsigned int stride=1; stride <= BLOCK_DIM; stride *= 2) {
        unsigned int i = (threadIdx.x + 1) * 2 * stride - 1;

        if(i < 2 * BLOCK_DIM)
            buffer_s[i] += buffer_s[i - stride];
        
        __syncthreads();
    }

    // post-reduction step
    for(unsigned int stride=BLOCK_DIM/2; stride >= 1; stride /= 2) {
        unsigned int i = (threadIdx.x + 1) * 2 * stride - 1;
        if (i + stride < 2*BLOCK_DIM) {
            buffer_s[i + stride] += buffer_s[i];
        }
        __syncthreads();
    }

    // update the partial sums
    if (threadIdx.x == 0)
        partialSums[blockIdx.x] = buffer_s[2 * BLOCK_DIM - 1];

    // copy from shared memory to output (each thread has to store 2 values in the output memory)
    output[segment + threadIdx.x] = buffer_s[threadIdx.x];
    output[segment + threadIdx.x + BLOCK_DIM] = buffer_s[threadIdx.x + BLOCK_DIM];
}

__global__ void add_kernel(float *output, float *partialSums, unsigned int N) {
    unsigned int segment = 2*blockIdx.x*blockDim.x;
    
    if (blockIdx.x > 0) {
        output[segment + threadIdx.x] += partialSums[blockIdx.x - 1];
        output[segment + threadIdx.x + BLOCK_DIM] += partialSums[blockIdx.x - 1];
    }
}

void scan_gpu_d(float* input_d, float* output_d, unsigned int N) {
    const unsigned int numThreadsPerBlock=BLOCK_DIM;
    const unsigned int numElementsPerBlock = 2 * numThreadsPerBlock;
    const unsigned int numBlocks = (N + numElementsPerBlock - 1)/numElementsPerBlock;

    float *partialSums_d;
    cudaMalloc((void**) &partialSums_d, numBlocks * sizeof(float));

    scan_kernel<<<numBlocks, numThreadsPerBlock>>>(input_d, output_d, partialSums_d, N);
    cudaDeviceSynchronize();

    // scan partial sums then add
    if(numBlocks > 1) {
        // scan partial sums
        scan_gpu_d(partialSums_d, partialSums_d, numBlocks);

        // add scanned sums
        add_kernel <<<numBlocks, numThreadsPerBlock>>> (output_d, partialSums_d, N);
    }

    cudaFree(partialSums_d);
}

int main() {
    const unsigned int N = 1 << 20; // 1M elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    std::vector<float> h_input(N), h_output_cpu(N), h_output_gpu(N);

    // Generate random input
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (auto& val : h_input) val = dis(gen);

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice);

    // Time and execute CPU version
    Timer cpu_timer;
    startTime(&cpu_timer);
    scan_cpu(h_input.data(), h_output_cpu.data(), N);
    stopTime(&cpu_timer);
    printElapsedTime(cpu_timer, "CPU Time", YELLOW);

    // Time and execute GPU version
    Timer gpu_timer;
    startTime(&gpu_timer);
    scan_gpu_d(d_input, d_output, N);
    cudaDeviceSynchronize();
    stopTime(&gpu_timer);
    printElapsedTime(gpu_timer, "GPU Time", GREEN);

    // Get GPU results and verify
    cudaMemcpy(h_output_gpu.data(), d_output, bytes, cudaMemcpyDeviceToHost);
    std::cout << (verify_results(h_output_cpu.data(), h_output_gpu.data(), N) 
                  ? GREEN "Results match!" RESET 
                  : RED "Results differ!" RESET) << std::endl;

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
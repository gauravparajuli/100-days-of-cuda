#include "timer.h"

#define NUM_BINS 256 // 0 - 255 range of values

__global__ void histogram_kernel(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int bins_s[NUM_BINS];

    // initialize shared memory
    unsigned int tid = threadIdx.x;
    if (tid < 256) {
        bins_s[tid] = 0;
    }
    __syncthreads(); // ensure all threads have initialized the shared memory

    if(i < width * height) {
        unsigned char b = image[i];
        atomicAdd(&bins_s[b], 1); // instead of ++bins[b] in order to avoid the race condition
    }

    // merge private histogram with global histogram
    if (tid < 256)
        if (bins_s[tid] > 0)
            atomicAdd(&bins[tid], bins_s[tid]); // merge into global memory
}

// CPU version of histogram computation
void histogram_cpu(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {
    memset(bins, 0, NUM_BINS * sizeof(unsigned int));
    for(unsigned int i = 0; i < width * height; i++) {
        bins[image[i]]++;
    }
}

void histogram_gpu(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {
    Timer timer;

    unsigned char *image_d;
    unsigned int *bins_d;
    startTime(&timer);
    cudaMalloc((void**) &image_d, width*height*sizeof(unsigned char));
    cudaMalloc((void**) &bins_d, NUM_BINS *sizeof(unsigned int));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU Memory Allocation", CYAN);

    startTime(&timer);
    cudaMemcpy(image_d, image, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(bins_d, 0, NUM_BINS*sizeof(unsigned int));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Host to Device Copy", BLUE);

    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (width * height + numThreadsPerBlock - 1)/numThreadsPerBlock;
    startTime(&timer);
    histogram_kernel<<<numBlocks, numThreadsPerBlock>>>(image_d, bins_d, width, height);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU Kernel Execution", GREEN);

    startTime(&timer);
    cudaMemcpy(bins, bins_d, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Device to Host Copy", YELLOW);

    cudaFree(image_d);
    cudaFree(bins_d);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU Memory Deallocation", MAGENTA);
}

int main() {
    // Test with sample data
    unsigned int width = 1920;
    unsigned int height = 1080;
    unsigned int size = width * height;
    
    // Allocate and initialize test image
    unsigned char* image = new unsigned char[size];
    for(unsigned int i = 0; i < size; i++) {
        image[i] = i % 256; // Fill with values 0-255
    }

    // Allocate bins for both CPU and GPU
    unsigned int* bins_cpu = new unsigned int[NUM_BINS];
    unsigned int* bins_gpu = new unsigned int[NUM_BINS];

    // Time and run CPU version
    Timer timer;
    startTime(&timer);
    histogram_cpu(image, bins_cpu, width, height);
    stopTime(&timer);
    printElapsedTime(timer, "CPU Kernel Execution", RED);

    // Run GPU version (includes all timed operations)
    histogram_gpu(image, bins_gpu, width, height);

    // Verify results
    bool match = true;
    for(int i = 0; i < NUM_BINS; i++) {
        if(bins_cpu[i] != bins_gpu[i]) {
            match = false;
            std::cout << "Mismatch at bin " << i << ": CPU=" << bins_cpu[i] 
                      << ", GPU=" << bins_gpu[i] << std::endl;
            break;
        }
    }
    std::cout << (match ? GREEN : RED) << "Results " 
              << (match ? "match" : "don't match") << RESET << std::endl;

    // Cleanup
    delete[] image;
    delete[] bins_cpu;
    delete[] bins_gpu;

    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "timer.h"
#include <cuda_runtime.h>

#define MASK_RADIUS 2
#define MASK_DIM (MASK_RADIUS * 2 +1)

__constant__ float mask_c[MASK_DIM][MASK_DIM];

__global__ void convolutionKernel(float *input, float *output, unsigned int width, unsigned int height) {
    // we will be assigning thread to compute every output element
    // so lets first determine the row and col of output element first

    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;

    // only threads within bound of input continues
    if (outRow < height && outCol < width) {
        float sum = 0.0f;

        for(int maskRow=0; maskRow<MASK_DIM; maskRow++) {
            for(int maskCol=0; maskCol<MASK_DIM; maskCol++) {
                // map mask col, row to corresponding input col, row
                int inRow = outRow - MASK_RADIUS + maskRow;
                int inCol = outCol - MASK_RADIUS + maskCol;
                
                if (inRow < height && inRow >= 0 && inCol < width && inCol >=0)
                    sum += mask_c[maskRow][maskCol] * input[inRow * width + inCol];
            }
        }

        output[outRow * width + outCol] = sum;
    }
}

void convolution_gpu(float *h_input, float *h_output, float h_mask[MASK_DIM][MASK_DIM], unsigned int width, unsigned int height) {
    Timer timer;
    Timer overallTimer;
    startTime(&overallTimer);

    float *d_input, *d_output;
    size_t size = width * height * sizeof(float);

    // 1. Time for GPU memory allocation
    startTime(&timer);
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);
    stopTime(&timer);
    printElapsedTime(timer, "GPU Memory Allocation", YELLOW);

    // 2. Time for copying from host to device
    startTime(&timer);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mask_c, h_mask, MASK_DIM * MASK_DIM * sizeof(float));
    stopTime(&timer);
    printElapsedTime(timer, "Host to Device Copy", BLUE);

    // Kernel launch configuration
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // 3. Time for kernel execution
    startTime(&timer);
    convolutionKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel Execution", GREEN);

    // 4. Time for copying from device to host
    startTime(&timer);
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    stopTime(&timer);
    printElapsedTime(timer, "Device to Host Copy", MAGENTA);

    // 5. Time for GPU memory deallocation
    startTime(&timer);
    cudaFree(d_input);
    cudaFree(d_output);
    stopTime(&timer);
    printElapsedTime(timer, "GPU Memory Deallocation", RED);

    // 6. Overall time for GPU
    stopTime(&overallTimer);
    printElapsedTime(overallTimer, "Overall GPU Time", CYAN);
}

void convolution_cpu(float *input, float *output, float mask[MASK_DIM][MASK_DIM], unsigned int width, unsigned int height) {
    Timer timer;
    startTime(&timer);

    for (int outRow = 0; outRow < height; outRow++) {
        for (int outCol = 0; outCol < width; outCol++) {
            float sum = 0.0f;

            for(int maskRow=0; maskRow<MASK_DIM; maskRow++) {
                for(int maskCol=0; maskCol<MASK_DIM; maskCol++) {
                    int inRow = outRow - MASK_RADIUS + maskRow;
                    int inCol = outCol - MASK_RADIUS + maskCol;
                    
                    if (inRow < height && inRow >= 0 && inCol < width && inCol >=0)
                        sum += mask[maskRow][maskCol] * input[inRow * width + inCol];
                }
            }

            output[outRow * width + outCol] = sum;
        }
    }

    stopTime(&timer);
    printElapsedTime(timer, "CPU Convolution Time", WHITE);
}

int main() {
    unsigned int width = 1024;
    unsigned int height = 1024;
    size_t size = width * height * sizeof(float);

    float *h_input = (float*)malloc(size);
    float *h_output_gpu = (float*)malloc(size);
    float *h_output_cpu = (float*)malloc(size);

    float h_mask[MASK_DIM][MASK_DIM] = {
        {1, 1, 1, 1, 1},
        {1, 2, 2, 2, 1},
        {1, 2, 3, 2, 1},
        {1, 2, 2, 2, 1},
        {1, 1, 1, 1, 1}
    };

    // Initialize input with some values
    srand(time(NULL));
    for (int i = 0; i < width * height; i++) {
        h_input[i] = (float)rand() / RAND_MAX;
    }

    // Perform convolution on GPU
    convolution_gpu(h_input, h_output_gpu, h_mask, width, height);

    // Perform convolution on CPU
    convolution_cpu(h_input, h_output_cpu, h_mask, width, height);

    // Verify results
    int match = 1;
    for (int i = 0; i < width * height; i++) {
        if (fabs(h_output_gpu[i] - h_output_cpu[i]) > 1e-5) {
            match = 0;
            break;
        }
    }

    if (match) {
        printf("GPU and CPU results match!\n");
    } else {
        printf("GPU and CPU results do not match!\n");
    }

    free(h_input);
    free(h_output_gpu);
    free(h_output_cpu);

    return 0;
}
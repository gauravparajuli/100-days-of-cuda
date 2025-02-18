#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 32

__global__ void softmax(int w, int h, float* input, float* output)
{
    __shared__ float reduction_max[BLOCK_DIM_Y];
    __shared__ float reduction_sum[BLOCK_DIM_Y];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int ty = threadIdx.y;

    if (row < h && col < w)
    {
        // Step 1: Find the maximum value in the row
        float maxval = input[row * w + col];
        for (int i = col + blockDim.x; i < w; i += blockDim.x)
        {
            maxval = fmaxf(maxval, input[row * w + i]);
        }
        reduction_max[ty] = maxval;

        // Parallel reduction to find the maximum value in the row
        for (int stride = blockDim.y / 2; stride >= 1; stride /= 2)
        {
            __syncthreads();
            if (ty < stride)
            {
                reduction_max[ty] = fmaxf(reduction_max[ty], reduction_max[ty + stride]);
            }
        }
        __syncthreads();
        maxval = reduction_max[0];

        // Step 2: Compute the sum of exponentials
        float exp_val = expf(input[row * w + col] - maxval);
        float sum = exp_val;
        for (int i = col + blockDim.x; i < w; i += blockDim.x)
        {
            sum += expf(input[row * w + i] - maxval);
        }
        reduction_sum[ty] = sum;

        // Parallel reduction to compute the sum of exponentials
        for (int stride = blockDim.y / 2; stride >= 1; stride /= 2)
        {
            __syncthreads();
            if (ty < stride)
            {
                reduction_sum[ty] += reduction_sum[ty + stride];
            }
        }
        __syncthreads();
        float divisor = reduction_sum[0];

        // Step 3: Compute the softmax output
        output[row * w + col] = exp_val / divisor;
    }
}

int main()
{
    int w = 1024; // Width of the matrix
    int h = 1024; // Height of the matrix

    // Allocate host memory
    float* h_input = (float*)malloc(w * h * sizeof(float));
    float* h_output = (float*)malloc(w * h * sizeof(float));

    // Initialize input matrix with random values
    for (int i = 0; i < w * h; i++)
    {
        h_input[i] = (float)rand() / RAND_MAX;
    }

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, w * h * sizeof(float));
    cudaMalloc((void**)&d_output, w * h * sizeof(float));

    // Copy input matrix to device
    cudaMemcpy(d_input, h_input, w * h * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridDim((w + BLOCK_DIM_X - 1) / BLOCK_DIM_X, (h + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

    // Launch the kernel
    softmax<<<gridDim, blockDim>>>(w, h, d_input, d_output);

    // Copy the result back to host
    cudaMemcpy(h_output, d_output, w * h * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    free(h_input);
    free(h_output);

    return 0;
}
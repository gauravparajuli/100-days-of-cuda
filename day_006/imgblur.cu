#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define BLUR_SIZE 1

__global__ void blurKernel(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pixel_index = (y * width + x) * channels;
    int red = 0; int green = 0; int blue = 0; int count = 0;

    // iterate over kernel regions
    for (int dy = -BLUR_SIZE; dy < BLUR_SIZE + 1; dy++) {
        for (int dx = -BLUR_SIZE; dx < BLUR_SIZE + 1; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int index = (ny * width + nx) * channels;
                red += input[index];
                green += input[index+1];
                blue += input[index+2];
                count++;
            }
        }
    }

    output[pixel_index] = red / count;
    output[pixel_index+1] = green / count;
    output[pixel_index+2] = blue / count;
}

int main(void) {
    int width, height, channels;

    // load the image here
    unsigned char *h_img = stbi_load("image.png", &width, &height, &channels, 3); // force 3 channels

    if (h_img == NULL) {
        printf("error loading the image\n");
        return -1;
    }

    printf("width: %d height: %d channels: %d\n", width, height, channels);

    size_t image_size = width * height * 3; // sizeof(unsigned char) is 1

    // allocate memory in host
    unsigned char* h_output = (unsigned char*)malloc(image_size);

    // allocate memory to gpu
    unsigned char *d_input, *d_output;
    cudaMalloc((void**)&d_input, image_size);
    cudaMalloc((void**)&d_output, image_size);

    // copy data to gpu
    cudaMemcpy(d_input, h_img, image_size, cudaMemcpyHostToDevice);

    dim3 block_size(16,16);
    dim3 grid_size((width + 15)/16, (height+15)/16);

    // convert to gray scale
    blurKernel<<<grid_size, block_size>>>(d_input, d_output, width, height, 3);
    
    // copy back to device
    cudaMemcpy(h_output, d_output, image_size, cudaMemcpyDeviceToHost);

    int success = stbi_write_png("blurred.png", width, height, 3, h_output, width*3);

    if (success) {
        printf("blurred image exported!\n");
    } else {
        printf("error while saving blurred image!\n");
    }

    // clear the memory
    stbi_image_free(h_img);
    free(h_output);
    cudaFree(d_input); cudaFree(d_output);

    return 0;
}
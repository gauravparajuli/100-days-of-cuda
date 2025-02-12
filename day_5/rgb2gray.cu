#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

__global__ void rgb2gray_kernel(unsigned char* red, unsigned char* green, unsigned char* blue, unsigned char* gray, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        gray[tid] = (unsigned char)(0.3f * red[tid] + 0.59f * green[tid] + 0.11f * blue[tid]);
    }
}

int main(void) {
    int width, height, channels;

    // load the image here
    unsigned char *img = stbi_load("image.png", &width, &height, &channels, 3); // force 3 channels

    if (img == NULL) {
        printf("error loading the image\n");
        return -1;
    }

    printf("width: %d height: %d channels: %d\n", width, height, channels);

    int total_pixels = width * height;

    // allocate memory in host (host pointer)
    unsigned char* red = (unsigned char*)malloc(total_pixels);
    unsigned char* green = (unsigned char*)malloc(total_pixels);
    unsigned char* blue = (unsigned char*)malloc(total_pixels);
    unsigned char* gray = (unsigned char*)malloc(total_pixels);
    
    // load rgb channels
    for (unsigned int i = 0; i < total_pixels; i++) {
        int index = i * 3;
        red[i] = img[index];
        green[i] = img[index + 1];
        blue[i] = img[index + 2];
    }

    // device pointers
    unsigned char *d_red, *d_green, *d_blue, *d_gray;

    // allocate memory on device
    cudaMalloc(&d_red, total_pixels);
    cudaMalloc(&d_green, total_pixels);
    cudaMalloc(&d_blue, total_pixels);
    cudaMalloc(&d_gray, total_pixels);

    // copy data from host to device
    cudaMemcpy(d_red, red, total_pixels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_green, green, total_pixels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_blue, blue, total_pixels, cudaMemcpyHostToDevice);


    int BLOCK_SIZE = 256; // number of threads in the block
    int GRID_SIZE = (int)ceil(total_pixels / BLOCK_SIZE); // number of blocks in the grid

    // convert to gray scale
    rgb2gray_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_red, d_green, d_blue, d_gray, total_pixels);
    
    // // grayscale conversion by sequential host code
    // for (int i = 0; i < total_pixels; i++) {
    //     gray[i] = (unsigned char)(0.3f * red[i] + 0.59f * green[i] + 0.11f * blue[i]);
    // }

    // copy back to device
    cudaMemcpy(gray, d_gray, total_pixels, cudaMemcpyDeviceToHost);

    int success = stbi_write_png("grayscale.png", width, height, 1, gray, width);

    if (success) {
        printf("grayscale image exported!\n");
    } else {
        printf("error while saving grayscale image!\n");
    }

    // clear the memory
    stbi_image_free(img);
    free(red); free(green); free(blue); free(gray);
    cudaFree(d_red); cudaFree(d_green); cudaFree(d_blue); cudaFree(d_gray);

    return 0;

}
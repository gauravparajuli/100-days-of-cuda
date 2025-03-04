#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA kernel for merging two sorted subarrays
__global__ void mergeKernel(int *arr, int *temp, int left, int mid, int right) {
    int i = left;      // Left subarray index
    int j = mid + 1;   // Right subarray index
    int k = left;      // Temp array index
    
    // Merge the two sorted subarrays
    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }
    
    // Copy remaining elements from left subarray
    while (i <= mid) {
        temp[k++] = arr[i++];
    }
    
    // Copy remaining elements from right subarray
    while (j <= right) {
        temp[k++] = arr[j++];
    }
    
    // Copy back to original array
    for (i = left; i <= right; i++) {
        arr[i] = temp[i];
    }
}

// Host function to perform merge sort
void mergeSortCUDA(int *arr, int n) {
    int *d_arr, *d_temp;
    int size = n * sizeof(int);
    
    // Allocate device memory
    cudaMalloc(&d_arr, size);
    cudaMalloc(&d_temp, size);
    
    // Copy input array to device
    cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);
    
    // Calculate grid and block dimensions
    dim3 blockSize(256);
    int numBlocks;
    
    // Iterative merge sort
    for (int width = 1; width < n; width *= 2) {
        numBlocks = (n + blockSize.x - 1) / blockSize.x;
        
        // Launch kernel for each pair of subarrays
        for (int left = 0; left < n; left += 2 * width) {
            int mid = min(left + width - 1, n - 1);
            int right = min(left + 2 * width - 1, n - 1);
            
            if (mid < right) {
                mergeKernel<<<1, blockSize.x>>>(d_arr, d_temp, left, mid, right);
            }
        }
        
        cudaDeviceSynchronize(); // Wait for all merges to complete
    }
    
    // Copy result back to host
    cudaMemcpy(arr, d_arr, size, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_arr);
    cudaFree(d_temp);
}

// Main function to test the implementation
int main() {
    int n = 16; // Example array size (should be power of 2 for simplicity)
    int *arr = (int*)malloc(n * sizeof(int));
    
    // Initialize array with random numbers
    printf("Original array: ");
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 100;
        printf("%d ", arr[i]);
    }
    printf("\n");
    
    // Perform merge sort
    mergeSortCUDA(arr, n);
    
    // Print sorted array
    printf("Sorted array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
    
    free(arr);
    return 0;
}

// // Utility function to find minimum of two numbers
// int min(int a, int b) {
//     return (a < b) ? a : b;
// }
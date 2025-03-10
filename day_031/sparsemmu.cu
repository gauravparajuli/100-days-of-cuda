#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void spmv_csr(int rows, const int *row_ptr, const int *col_idx, 
                         const float *values, const float *x, float *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = 0.0f;
        int start = row_ptr[row];
        int end = row_ptr[row + 1];
        for (int i = start; i < end; i++) {
            sum += values[i] * x[col_idx[i]];
        }
        y[row] = sum;
    }
}

void spmv_csr_host(int rows, int nnz, const std::vector<int> &row_ptr,
                   const std::vector<int> &col_idx, const std::vector<float> &values,
                   const std::vector<float> &x, std::vector<float> &y) {
    int *d_row_ptr, *d_col_idx;
    float *d_values, *d_x, *d_y;

    // Allocate device memory
    cudaMalloc((void**)&d_row_ptr, (rows + 1) * sizeof(int));
    cudaMalloc((void**)&d_col_idx, nnz * sizeof(int));
    cudaMalloc((void**)&d_values, nnz * sizeof(float));
    cudaMalloc((void**)&d_x, rows * sizeof(float));
    cudaMalloc((void**)&d_y, rows * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_row_ptr, row_ptr.data(), (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, col_idx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), rows * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int block_size = 256;
    int grid_size = (rows + block_size - 1) / block_size;
    spmv_csr<<<grid_size, block_size>>>(rows, d_row_ptr, d_col_idx, d_values, d_x, d_y);
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(y.data(), d_y, rows * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
}

int main() {
    // Example sparse matrix in CSR format
    int rows = 4;
    int nnz = 7;  // Number of non-zero values

    std::vector<int> row_ptr = {0, 2, 4, 6, 7};
    std::vector<int> col_idx = {0, 1, 1, 2, 2, 3, 3};
    std::vector<float> values = {10, 20, 30, 40, 50, 60, 70};
    std::vector<float> x = {1, 2, 3, 4};  // Input vector
    std::vector<float> y(rows, 0);        // Output vector

    spmv_csr_host(rows, nnz, row_ptr, col_idx, values, x, y);

    // Print result
    std::cout << "Result y: ";
    for (float v : y) std::cout << v << " ";
    std::cout << std::endl;

    return 0;
}

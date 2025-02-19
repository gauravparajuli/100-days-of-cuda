#include <stdio.h>
#include <cublas_v2.h>

int main() {
    const int N = 1024;
    float A[N], B[N], C[N];

    // initialization
    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = i;
    }

    // create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // allocate device memory
    float *d_a, *d_b;
    cudaMalloc((void**)&d_a, N*sizeof(float));
    cudaMalloc((void**)&d_b, N*sizeof(float));

    // copy data from host to device
    cudaMemcpy(d_a, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, N * sizeof(float), cudaMemcpyHostToDevice);

    // scaling factor
    const float alpha =2.0f;

    // perform vector addition
    cublasSaxpy(handle, N, &alpha, d_a, 1, d_b, 1);

    // copy result back to host (result is in d_b)
    cudaMemcpy(C, d_b, N * sizeof(float), cudaMemcpyDeviceToHost);

    // print results
    for (int i = 0; i < N; i++) {
        printf("%f ", C[i]);
    }

    // cleanup
    cudaFree(d_a); cudaFree(d_b);
    cublasDestroy(handle);

    return 0;
}


#include <iostream>
#include <math.h>

__global__ void add(int n, float *x, float *y) {
    for (int i=0; i < n; i++) {
        y[i] = x[i] + y[i];
    }
}

int main() {
    int N = 1<<20;

    float *x, *y;
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // initialize x and y arrays
    for (int i=0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    add<<<1, 1>>>(N, x, y);

    cudaDeviceSynchronize();

    // check for errors (all value should be 3.0f)
    float maxError = 0.0f;
    for (int i=0; i < N; i++) {
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    // free memory
    cudaFree(x);
    cudaFree(y);

    return 0;

}
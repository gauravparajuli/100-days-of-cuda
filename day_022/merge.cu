#include <algorithm> // for std::is_sorted

#include "timer.h"

#define ELEM_PER_THREAD 8
#define THREADS_PER_BLOCK 128
#define ELEM_PER_BLOCK (THREADS_PER_BLOCK * ELEM_PER_THREAD)

__device__ __host__ void mergeSequential(float *A, float *B, float *C, unsigned int m, unsigned int n) {
    unsigned int i = 0; // A
    unsigned int j = 0; // B
    unsigned int k = 0; // C

    while(i < m && j < n) {
        if(A[i] < B[j])
            C[k++] = A[i++];
        else 
            C[k++] = B[j++];
    }

    while(i < m)
        C[k++] = A[i++];

    while(j < n)
        C[k++] = B[j++];
}

// given k finds corresponding i. then corresponding j can be computed by k-i
__device__ unsigned int coRank(float *A, float *B, unsigned int m, unsigned int n, unsigned int k) {
    unsigned int iLow = (k > n)?(k - n):0;
    unsigned int iHigh = (m < k)?m:k;

    // binary search
    while(true) {
        unsigned int i = (iLow + iHigh) / 2; // first guess in the middle of the bound
        unsigned int j = k - i;

        // check if guess is too high (also do boundary check)
        if(i > 0 && j < n && A[i-1] > B[j]) {
            iHigh = i;
            
        // check if guess is too low
        } else if (j > 0 && i < m && B[j - 1] > A[i]) {
            iLow = i;
        } else{
            return i;
        }
    }
}

__global__ void mergeKernel(float *A, float *B, float *C, unsigned int m, unsigned int n) {
    unsigned int k = (blockDim.x * blockIdx.x + threadIdx.x) * ELEM_PER_THREAD;

    if(k < m + n) {
        unsigned int i = coRank(A, B, m, n, k);
        unsigned int j = k - i;
        unsigned int kNext = (k + ELEM_PER_THREAD < m + n)?(k + ELEM_PER_THREAD):(m+n);
        unsigned int iNext = coRank(A, B, m, n, kNext);
        unsigned int jNext = kNext - iNext; // now I have the beginning and end of each of the input segments
        mergeSequential(&A[i], &B[j], &C[k], iNext - i, jNext - j);
    }
}

void merge_gpu(float *A, float *B, float *C, unsigned int m, unsigned int n) {
    Timer timer;

    float *A_d, *B_d, *C_d;
    cudaMalloc((void**) &A_d, m*sizeof(float));
    cudaMalloc((void**) &B_d, n*sizeof(float));
    cudaMalloc((void**) &C_d, (m+n)*sizeof(float));

    cudaMemcpy(A_d, A, m*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, m*sizeof(float), cudaMemcpyHostToDevice);

    unsigned int numBlocks = (m + n + ELEM_PER_BLOCK - 1)/ELEM_PER_BLOCK;
    
    Timer gpu_timer;
    startTime(&gpu_timer);
    mergeKernel <<<numBlocks, THREADS_PER_BLOCK>>>(A_d, B_d, C_d, m, n);
    cudaDeviceSynchronize();
    stopTime(&gpu_timer);
    printElapsedTime(gpu_timer, "GPU Merge Time", GREEN);

    cudaMemcpy(C, C_d, (m+n)*sizeof(float), cudaMemcpyDeviceToHost);
}

int main() {
    // Test parameters
    const unsigned int m = 1<<20; // Size of array A
    const unsigned int n = 1<<20; // Size of array B
    
    // Allocate memory
    float *A = new float[m];
    float *B = new float[n];
    float *C_cpu = new float[m + n];
    float *C_gpu = new float[m + n];
    
    // Initialize sorted input arrays with predictable values
    for(unsigned int i = 0; i < m; i++) {
        A[i] = (float)(i * 2);      // Even numbers: 0, 2, 4, 6, ...
    }
    for(unsigned int i = 0; i < n; i++) {
        B[i] = (float)(i * 2 + 1);  // Odd numbers: 1, 3, 5, 7, ...
    }

    // CPU timing and execution
    Timer cpu_timer;
    startTime(&cpu_timer);
    mergeSequential(A, B, C_cpu, m, n);
    stopTime(&cpu_timer);
    
    // GPU timing and execution
    merge_gpu(A, B, C_gpu, m, n);
    
    // Print timing results
    printElapsedTime(cpu_timer, "CPU Merge Time", RED);
    
    
    // Verify results match
    bool results_match = true;
    for(unsigned int i = 0; i < m + n; i++) {
        if(C_cpu[i] != C_gpu[i]) {
            results_match = false;
            std::cout << RED << "Mismatch at index " << i 
                      << ": CPU=" << C_cpu[i] << ", GPU=" << C_gpu[i] << RESET << std::endl;
            break;
        }
    }
    
    if(results_match) {
        std::cout << YELLOW << "CPU and GPU results match!" << RESET << std::endl;
    } else {
        std::cout << RED << "CPU and GPU results don't match!" << RESET << std::endl;
    }
    
    // Verify output is sorted
    bool is_sorted = std::is_sorted(C_gpu, C_gpu + m + n);
    std::cout << (is_sorted ? YELLOW : RED) 
              << "GPU Result is " << (is_sorted ? "" : "not ") << "sorted" 
              << RESET << std::endl;

    // Cleanup
    delete[] A;
    delete[] B;
    delete[] C_cpu;
    delete[] C_gpu;
    
    return 0;
}
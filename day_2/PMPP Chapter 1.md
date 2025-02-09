# PMPP Chapter 1
## Thread organization
`threadIdx.x`: identifies thread within a block <br>
`blockIdx.x`: identifies a block within a grid <br>
`blockDim.x`: number of threads per block

unique index of each thread given by:
`int i = blockIdx.x * threadIdx.x * blockDim.x`

## Memory Model
global memory -> slow -> accessible by all threads -> to store large datasets but in cost of high latency
shared memory -> fast -> accessible by all threads in a block -> to reduce memory access time
registers -> fastest -> private to each thread -> to store frequently used data

`cudaMalloc()`: allocates memory on device <br>
`cudaMemcpy()`: transfers data between GPU and CPU <br>
`cudaFree()`: free GPU memory


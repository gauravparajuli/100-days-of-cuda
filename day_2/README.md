# Day 2 Notes

Compute Unified Device Architecture (CUDA)
Parallel Thread Execution (PTX)

functions with `__global__` keyword should always return void.

blocks -> number of thread groupings
blocks are then grouped in grid (this is the highest entity in CUDA thread hierarchy)

gpuWork<<<2, 4>>>() 2 blocks and 4 threads in each block

launching kernels is asynchronous

each block cannot have different number of threads

`blockDim.x` to determine number of threads in a block
`gridDim.x` gives number of blocks in a grid

a variable whose declaration is preceded by the `__device__` keyword is a global variable.

CUDA assumes that kernels depend upon one another unless specified otherwise

CPU is optimized to minimize latency vs GPU is optimized to maximize throughput.

## Memory

Global memory/Device Memory is accessible to all the threads

device memory <-> l2$ <-> l1$ <-> register

## Keywords
`__host__`: executes on host machine (cpu) <br>
`__global__`: executes on the device, called from the host <br>
`__device__`: executes on the device, called from the device

## Architecture
### Hardware wise
SM -> Block/Warp -> Thread

Streaming Multiprocessor (SMs) manage 2048 threads (or 64 warps of threads 32*64=2048 threads)
A warp is the vector element of the GPU, it consists of 32 consecutive threads. `threadIdx` 0-31 for 1st warp, 32-63 for second warp and so on. 

one block always run on a single SM. It can never span 2 SMs.

threads will execute in the group of 32 regardless of the blocksize being used.

### Software Perspective
Grid -> Block -> Thread

`1024` is the maximum number of threads in a block.
`65,535` is the maximum number of blocks in a single dimension of the grid.

### Better Peformance
1. Choose the number of blocks that is a multiple of SMs.
2. Choose a block size that is multiple of 32. (because warps consists of 32 threads)











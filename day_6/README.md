An atomic operation ensures that only a single thread has access to piece of memory while an operation completes.
Atomics are slower than normal load/store. (you may have the entire machine queuing for a single location in memory)
You cannot use normal load/store for inter thread communication because of race conditions.

each thread block is mapped to one or more warps.
hardware schedules each warp independently.
Streaming Multiprocessor only executes on warp at a time.

Peformance drops with the degree of control flow divergence but it will be correct.

memory types:
thread -> local memory, register
block -> shared memory
grid -> global memory, constant memory

__shared__ int *ptr -> shared pointer variable

global memory is slower than shared memory
tile data to take advantage of fast shared memory

## common programming strategy
1. partition data into subsets that fit into shared memory
2. handle each datasubset with one thread block
3. load subset from global memory to shared memory (using multiple threads to exploit memory level parallelism)
4. perform computations on the subset from shared memory
5. copy the results from shared memory to global memory



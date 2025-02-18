Device pointer (allocated with cudaMalloc) cannot be dereferenced in host code.
Host pointers cannot be used to access device memory

`cudaMalloc(void **devPtr, size_t count);` cudaMalloc expects pointer to a pointer
`cudaFree(void *devPtr);` 
`cudaMemcpy(void *dst, void *src, size_t count, cudaMemcpKind)`
# 100-Day CUDA Challenge

| Day | Description | Links | Status |
| --- | --- | --- | --- |
| 1 | Hello World <br> Adding two vectors | [hello_world.cu](./day_001/hello_world.cu) <br> [add_array.cu](./day_001/add_array.cu) | ✅ |
| 2 | Readings on CUDA and taking notes | [README.md](./day_002/README.md) <br> [PMPP](./day_002/PMPP%20Chapter%201.md) | ✅ |
| 3 | Learnt to use cudaMalloc, cudaMemcpy and cudaFree | [vector_add.cu](./day_003/vector_add.cu) | ✅ |
| 4 | Vector addition by leveraging multiple threads <br> Matrix Multiplication | [vector_add.cu](./day_004/vector_add.cu) <br> [matmul.cu](./day_004/matmul.cu) | ✅ |
| 5 | RGB2GRAY Kernel | [rgb2gray.cu](./day_005/rgb2gray.cu) | ✅ |
| 6 | Image Blurring Kernel | [imgblur.cu](./day_006/imgblur.cu) | ✅ |
| 7 | Read about unified memory architecture <br> TiledMatrixMultiplication using Shared Memory | [umemory.cu](./day_007/umemory.cu) <br> [tiledmmu.cu](./day_007/tiledmmu.cu) | ✅ |
| 8 | Implemented Softmax Kernel <br> Will learn about reduction techniques and optimize this tomorrow | [smax.cu](./day_008/smax.cu) | ✅ |
| 9 | Implemented Improved Parallel Sum Reduction Kernel <br> Implemented Optimized Softmax Kernel using parallel reductions | [better_reduction.cu](./day_009/better_reduction.cu) <br> [better_softmax.cu](./day_009/better_softmax.cu) | ✅ |
| 10 | Implemented Vector addition in Triton <br> Implemented 2D Convolution using constant memory | [01_vec_addition.py](./day_010/01_vec_addition.py) <br> [conv2d.cu](./day_010/conv2d.cu) | ✅ |
| 11 | Implemented Naive 3D Stencil operation | [3dstencil_naive.cu](./day_011/3dstencil_naive.cu) | ✅ |
| 12 | Introduction to cuBLAS Vector Addition| [cublas_vadd.cu](./day_012/cublas_vadd.cu) | ✅ |
| 13 | Naive 1D Convolution | [naive1dconv.cu](./day_013/naive1dconv.cu) | ✅ |
| 14 | Implemented GELU Activation Function <br> Implementing C/C++ extensions in PyTorch | [gelu.cu](./day_014/gelu.cu) <br> [polynomial_cuda.cu](./day_014_pytorch_extensions/polynomial_cuda.cu) <br> [polynomial_activation.py](./day_014_pytorch_extensions/polynomial_activation.py) | ✅ |
| 15 | Optimized Tiled Matrix Multiplication by using Thread Coarsening | [mmul.cu](./day_015/mmul.cu) | ✅ |
| 16 | Optimized stencil kernel by implementing tiling and finally register tilling. <br> there is some error in register tiling  but will fix this tomorrow. | [3dstencil.cu](./day_016/3dstencil.cu) | ✅ |
| 17 | Optimized parallel reduction kernel by memory coalescing and minimizing control divergence | [preduction.cu](./day_017/preduction.cu) | Will do some minor bug fixes tomorrow |
| 18 | Implemented Koggee Stone Scanning Algorithm | [kstone_scan.cu](./day_018/kstone_scan.cu) | ✅ |
| 19 | Implemented Brent Kung Scanning Algorithm | [scanbrentkung.cu](./day_019/scanbrentkung.cu) | ✅ |
| 20 | Implemented Histogram Algorithm <br> Optimized it by using Privatisation and Shared Memory <br> Also learnt about atomic operations | [histogram.cu](./day_020/histogram.cu) | ✅ |
| 21 | Implemented ReLU activation kernel | [relu.cu](./day_021/relu.cu) | ✅ |
| 22 | Implemented Merge Parallel Patten | [merge.cu](./day_022/merge.cu) | ✅ |
| 23 | Matrix Transpose | [transpose.cu](./day_023/transpose.cu) | ✅ |
| 24 | Brick Sort | [bricksort.cu](./day_024/bricksort.cu) | ✅ |
| 25 | Merge Sort | [merge_sort.cu](./day_025/merge_sort.cu) | ✅ |

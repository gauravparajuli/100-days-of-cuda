from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='polynomial_cuda',
    ext_modules=[
        # CppExtension(
        #     name='cppcuda_tutorial',
        #     sources=['interpolation.cpp']
        # )
        CUDAExtension(
            name='polynomial_cuda',
            sources=[
                'polynomial_cuda.cu'
            ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
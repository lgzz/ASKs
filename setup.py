from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='ask_cuda',
    ext_modules=[
        CUDAExtension('ask_cuda', [
            'ask_cuda.cpp',
            'ask_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

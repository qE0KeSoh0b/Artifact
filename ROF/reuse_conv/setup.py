from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='ROF',
    ext_modules=[
        CUDAExtension(
        name='ROF', 
        sources=[   
                    'reuse_conv.cpp', 
                    'reuse_conv_kernel.cu'
                ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
        
    })
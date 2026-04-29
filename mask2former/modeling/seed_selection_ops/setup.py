import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CppExtension, CUDAExtension


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    sources = [os.path.join(this_dir, "src", "seed_selection.cpp")]
    cuda_sources = [os.path.join(this_dir, "src", "seed_selection_cuda.cu")]
    use_cuda = (os.environ.get('FORCE_CUDA') or torch.cuda.is_available()) and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension
    if use_cuda:
        sources += cuda_sources
    return [
        extension(
            "LatentFormerSeedSelection",
            sources,
            define_macros=[("WITH_CUDA", None)] if use_cuda else [],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            }
            if use_cuda
            else {"cxx": ["-O3"]},
        )
    ]


setup(
    name="LatentFormerSeedSelection",
    version="1.0",
    description="Native clustering seed selection op for LatentFormer",
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)

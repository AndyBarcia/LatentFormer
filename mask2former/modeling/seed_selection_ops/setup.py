import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    sources = [os.path.join(this_dir, "src", "seed_selection.cpp")]
    return [
        CppExtension(
            "LatentFormerSeedSelection",
            sources,
            extra_compile_args={"cxx": ["-O3"]},
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

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup

setup(
    name="learned-bloom",
    version="0.1.0",
    description="Learned Bloom Filter research implementation scaffold",
    packages=find_packages("src/python"),
    package_dir={"": "src/python"},
    ext_modules=[
        Pybind11Extension(
            "learned_bloom_cpp",
            [
                "cpp/src/pybind_module.cpp",
                "cpp/src/bloom_filter.cpp",
                "cpp/src/fused_trigram_scorer.cpp",
            ],
            include_dirs=["cpp/include"],
            cxx_std=17,
        )
    ],
    cmdclass={"build_ext": build_ext},
) 
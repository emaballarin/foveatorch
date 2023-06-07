#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#  Copyright (c) 2023 Emanuele Ballarin <emanuele@ballarin.cc>
#  Released under the terms of the MIT License
#  (see: https://url.ballarin.cc/mitlicense)
#
# ------------------------------------------------------------------------------
import os
import warnings

from setuptools import find_packages
from setuptools import setup


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read().strip()


def check_dependencies(dependencies: list[str]):
    missing_dependencies: list[str] = []
    package_name: str
    for package_name in dependencies:
        try:
            __import__(package_name)
        except ImportError:
            missing_dependencies.append(package_name)

    if missing_dependencies:
        warnings.warn(f"Missing dependencies: {missing_dependencies}")


DEPENDENCY_PACKAGE_NAMES: list[str] = [
    "kornia",
    "torch",
    "torchvision",
]

check_dependencies(DEPENDENCY_PACKAGE_NAMES)

PACKAGENAME: str = "foveatorch"


setup(
    name=PACKAGENAME,
    version="0.0.2",
    author="Emanuele Ballarin",
    author_email="emanuele@ballarin.cc",
    url="https://github.com/emaballarin/foveatorch",
    description="Foveated vision for Deep Learning methods",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    keywords=[
        "Deep Learning",
        "Machine Learning",
        "Computer Vision",
        "Computational Neuroscience",
    ],
    license="MIT",
    packages=[
        package for package in find_packages() if package.startswith(PACKAGENAME)
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Environment :: Console",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.10",
    include_package_data=True,
    zip_safe=False,
)

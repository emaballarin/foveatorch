#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#  Copyright (c) 2024 Emanuele Ballarin <emanuele@ballarin.cc>
#  Released under the terms of the MIT License
#  (see: https://url.ballarin.cc/mitlicense)
#
# ------------------------------------------------------------------------------
import os

from setuptools import find_packages
from setuptools import setup


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read().strip()


PACKAGENAME: str = "foveatorch"

setup(
    name=PACKAGENAME,
    version="0.1.7",
    author="Emanuele Ballarin",
    author_email="emanuele@ballarin.cc",
    url="https://github.com/emaballarin/foveatorch",
    description="Differentiable foveated vision for Deep Learning methods",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    keywords=[
        "Deep Learning",
        "Machine Learning",
        "Computer Vision",
        "Computational Neuroscience",
        "Differentiable Programming",
    ],
    license="MIT",
    packages=[
        package for package in find_packages() if package.startswith(PACKAGENAME)
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Environment :: Console",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.11",
    install_requires=[
        "torch>=2",
        "kornia>=0.6.12",
    ],
    include_package_data=False,
    zip_safe=True,
)

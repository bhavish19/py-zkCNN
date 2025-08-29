#!/usr/bin/env python3
"""
Setup script for ZKCNN Multi-Models Implementation
Phase 3: Advanced Cryptographic ZKCNN with BLS12-381
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="zkcnn-multi-models",
    version="3.0.0",
    author="ZKCNN Research Team",
    author_email="research@zkcnn.com",
    description="Advanced Zero-Knowledge Convolutional Neural Networks with BLS12-381 Cryptography",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/zkcnn/zkcnn-multi-models",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security :: Cryptography",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "zkcnn=zkCNN_multi_models:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.so", "*.dll", "*.dylib", "*.cpp", "*.h", "*.txt", "*.md"],
    },
    keywords="zero-knowledge proofs, cryptography, neural networks, CNN, BLS12-381, GKR, Hyrax",
    project_urls={
        "Bug Reports": "https://github.com/zkcnn/zkcnn-multi-models/issues",
        "Source": "https://github.com/zkcnn/zkcnn-multi-models",
        "Documentation": "https://github.com/zkcnn/zkcnn-multi-models/blob/main/README.md",
    },
)

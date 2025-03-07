"""Setup script for packaging the automatic text alignment library."""

from setuptools import setup, find_packages

setup(
    name="ukraa",
    version="0.1.0",
    description="Library for automatic text alignment",
    author="Danylo",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "laserembeddings",
        "hydra-core",
        "setuptools",
        "faiss-cpu",
    ],
    entry_points={
        "console_scripts": [
            "auto-align=auto_align.cli:main",
        ],
    },
)
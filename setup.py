from setuptools import setup, find_packages

setup(
    name="featureselection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
    ],
    description="Package for feature selection",
)

from setuptools import setup, find_packages

setup(
    name="GRLT_analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "pandas>=1.0.0",
        "seaborn>=0.11.0",
    ],
    description="Package for loading and manipulating Golden Retriever Lifetime Study data",
)

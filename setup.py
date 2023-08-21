from distutils.core import setup

from setuptools import find_packages

setup(
    name="slide_tools",
    version="0.2",
    packages=find_packages(),
    long_description=open("README.md").read(),
    install_requires=[
        "cucim",
        "numpy",
        "pandas",
        "pytorch_lightning",
        "rasterio",
        "scipy",
        "shapely",
        "torch",
        "tqdm",
        "xmltodict",
        "openslide-python",
    ],
)

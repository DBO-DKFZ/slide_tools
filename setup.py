from distutils.core import setup

from setuptools import find_packages

setup(
    name="slide_tools",
    version="0.2.1",
    packages=find_packages(),
    long_description=open("README.md").read(),
    install_requires=[
        "cucim",
        "numpy",
        "pandas",
        "lightning",
        "rasterio",
        "scipy",
        "shapely",
        "torch",
        "tqdm",
        "xmltodict",
        "openslide-python",
    ],
)

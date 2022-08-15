from distutils.core import setup

setup(
    name="slide_tools",
    version="0.1.1",
    packages=[
        "slide_tools",
    ],
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
    ]
)

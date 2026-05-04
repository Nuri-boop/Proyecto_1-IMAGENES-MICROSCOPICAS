from setuptools import setup, find_packages

setup(
    name="cellcounter",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "matplotlib",
    ],
)
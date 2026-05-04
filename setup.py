from setuptools import setup, find_packages

setup(
    name="cellcounter",
    version="1.0.0",
    author="Casillas M., Nuñez L., Muñoz V., Santos N.",
    description="Librería para detección y conteo de células en imágenes microscópicas",
    long_description="Proyecto de Procesamiento de Imágenes - Ingeniería ROBOTICA",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "opencv-python",
        "scipy"
    ],
    python_requires='>=3.8',
)
from setuptools import setup, find_packages

setup(
    name="morphic-cv",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.20.0",
        "Pillow>=9.0.0"
    ],
    entry_points={
        "console_scripts": [
            "morphic-cv=morphic_cv.cli.main:main",
        ],
    },
)

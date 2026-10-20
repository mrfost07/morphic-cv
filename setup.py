from setuptools import setup, find_packages

setup(
    name="morphic-cv",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.20.0",
        "Pillow>=9.0.0",
        "opencv-python>=4.5.0",
    ],
    entry_points={
        "console_scripts": [
            "morphic-cv=morphic_cv.cli.main:main",
            "morphic-inference=morphic_cv.inference.__main__:main",
        ],
    },
)

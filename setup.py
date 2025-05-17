from setuptools import setup, find_packages

setup(
    name="deep_particle_tracker",
    version="0.1.0",
    description="Deep learning-based particle tracking for microscopy",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/deep_particle_tracker",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "matplotlib>=3.3.0",
        "PyQt5>=5.15.0",
        "torch>=1.7.0",
        "tifffile>=2020.9.3",
        "h5py>=3.1.0",
        "tqdm>=4.50.0",
        "pillow>=8.0.0",
    ],
    entry_points={
        "console_scripts": [
            "deep_particle_tracker=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
)

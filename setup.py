from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="virus-host-classification",
    version="0.1.0",
    author="Alan C.",
    author_email="ga5808@wayne.edu",
    description="A machine learning pipeline for classifying virus hosts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hiyata/virus-host-classification",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "h5py>=3.1.0",
        "biopython>=1.79",
        "umap-learn>=0.5.1",
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
        "joblib>=1.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.5b2",
            "flake8>=3.9.0",
            "isort>=5.9.0",
            "jupyter>=1.0.0",
            "sphinx>=4.0.0",
            "google-generativeai>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "virus-preprocess=scripts.preprocess_fasta:main",
            "virus-features=scripts.generate_features:main",
            "virus-split=scripts.split_dataset:main",
            "virus-train=scripts.train_models:main",
            "virus-tsne=scripts.visualize_tsne:main",
        ],
    },
)
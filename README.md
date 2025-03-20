# Virus Host Classification

A machine learning pipeline for classifying virus hosts based on k-mer frequency features.

## Overview

This repository contains a complete pipeline for predicting virus hosts from genomic sequences. The pipeline processes raw FASTA files, calculates k-mer frequency features, and trains machine learning models to classify viruses as either human or non-human. It includes neural network, logistic regression, and random forest models for classification.

## Features

- **Preprocessing**: Filter FASTA sequences based on keywords and sequence quality
- **Feature extraction**: Calculate k-mer frequency features for various k values (3-8)
- **Dataset splitting**: Create train/validation/test splits preserving family and host distributions
- **Model training**: Train neural network, logistic regression, and random forest models
- **Visualization**: Generate t-SNE plots and model performance visualizations
- **Metadata processing**: Process virus metadata for additional insights

## Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/hiyata/host-virus-dataset.git
cd host-virus-dataset
pip install -e .
```

Or install with all dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

### Full Pipeline

Run the entire pipeline with default settings:

```bash
# 1. Preprocess FASTA files
python scripts/preprocess_fasta.py --input_dir data/raw_fastas --output_dir data/processed_fastas

# 2. Generate k-mer features
python scripts/generate_features.py --input_dir data/processed_fastas --output_dir data/kmer_features

# 3. Split dataset
python scripts/split_dataset.py --input_dir data/kmer_features --output_dir data/dataset_splits

# 4. Train models
python scripts/train_models.py --data_dir data/dataset_splits --output_dir results/models

# 5. Visualize results
python scripts/visualize_tsne.py --h5_path data/dataset_splits/k6/train.h5 \
                              --model_path results/models/k6/nn/model.pt \
                              --scaler_path results/models/k6/nn/scaler.pkl \
                              --output_dir results/visualizations
```

### Preprocessing

```bash
python scripts/preprocess_fasta.py --input_dir data/raw_fastas --output_dir data/processed_fastas \
                                --exclusion_keywords "partial,mutant,unverified,bac,clone" \
                                --length_filter --min_length 500 --max_length 50000
```

### Feature Generation

```bash
python scripts/generate_features.py --input_dir data/processed_fastas --output_dir data/kmer_features \
                                 --k_values 3,4,5,6,7,8 --config config/default_config.yml
```

### Dataset Splitting

```bash
python scripts/split_dataset.py --input_dir data/kmer_features --output_dir data/dataset_splits \
                             --k_values 3,4,5,6,7,8 --cluster_k 6
```

### Model Training

```bash
python scripts/train_models.py --data_dir data/dataset_splits --output_dir results/models \
                            --kmers 3 4 5 6 7 8 --model all --batch_size 64 \
                            --learning_rate 1e-5 --epochs 60 --n_estimators 100
```

### Visualization

```bash
python scripts/visualize_tsne.py --h5_path data/dataset_splits/k6/train.h5 \
                              --model_path results/models/k6/nn/model.pt \
                              --scaler_path results/models/k6/nn/scaler.pkl \
                              --output_dir results/visualizations \
                              --sample_size 2000 --perplexity 30 --save_coords
```

### Metadata Processing

```bash
python scripts/process_metadata.py --input metadata/virus_data.tsv \
                                --config config/host_patterns.yml \
                                --output_dir results/metadata
```

## Configuration

Configuration files are found in the `config/` directory:

- `default_config.yml`: Default parameters for the pipeline
- `host_patterns.yml`: Patterns for virus metadata processing

# Using Custom Data

## Quick Start with Your Own Data

```bash
# 1. Preprocess your FASTA files
python scripts/preprocess_fasta.py --input_dir path/to/your/fastas --output_dir processed_data

# 2. Generate k-mer features
python scripts/generate_features.py --input_dir processed_data --output_dir features_data

# 3. Split dataset into train/val/test
python scripts/split_dataset.py --input_dir features_data --output_dir split_data

# 4. Train models on your data
python scripts/train_models.py --data_dir split_data --output_dir your_results
```

For detailed information on data requirements and formats, see [DATA_GUIDE.md](DATA_GUIDE.md).

## Custom Classification Tasks

This pipeline is flexible and can be adapted for different classification tasks, not just human/non-human host classification. You can classify by:

- Different host types (human, bat, bird, etc.)
- Virus families
- Genome types
- Custom categories

See the [DATA_GUIDE.md](DATA_GUIDE.md) for instructions on configuring custom classification tasks.

## Project Structure

```
host-virus-dataset/
├── README.md               # Project overview
├── requirements.txt        # Dependencies
├── test_pipeline.py        # Run full pipeline
├── setup.py                # Package installation
├── .gitignore              # Git ignore file
├── config/                 # Configuration files
│   ├── default_config.yml  # Default configuration
│   └── host_patterns.yml   # Host pattern definitions
├── src/                    # Source code
│   ├── data/               # Data processing
│   ├── models/             # Model definitions
│   ├── training/           # Training pipelines
│   └── visualization/      # Visualization tools
├── scripts/                # Standalone scripts
└── notebooks/              # Jupyter notebooks for examples
```

## Requirements

- Python 3.8+
- PyTorch
- scikit-learn
- pandas
- numpy
- h5py
- matplotlib
- seaborn
- biopython
- umap-learn

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Citation

If you use this code in your research, please cite:

```
@software{,
  author = {},
  title = {},
  year = {},
  url = {https://github.com/hiyata/virus-host-classification}
}
```
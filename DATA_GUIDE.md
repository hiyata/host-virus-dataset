# Data Guide for Virus Host Classification

This document explains how to use your own data with this pipeline.

## Data Requirements

### FASTA Files

The pipeline expects FASTA files organized by virus family:

```
data/
├── flaviviridae.fasta
├── adenoviridae.fasta
├── herpesviridae.fasta
└── ... (other virus families)
```

Alternatively, you can use a single FASTA file with properly formatted headers that include the family information.

#### FASTA Header Format

The pipeline works best with headers that follow this format:
```
>SequenceID|Family|Host|Other_metadata
```

Example:
```
>NC_001802.1|lentiviridae|Homo sapiens|HIV-1
GCCTGGGAGCTCTCTGGCTAACTAGGGAACCCACTGCTTAAGCCTCAATAAAGCTTGCCT...
```

If your headers don't follow this format, you can provide a separate metadata file.

### Metadata File

If your FASTA headers don't contain all necessary information, provide a tab-separated metadata file:

```
sequence_id  virus_name  strain_name  family  host  location  isolation_source
NC_001802.1  HIV-1       HXB2         lentiviridae  Homo sapiens  USA  blood
...
```

## Using Your Own Data

### Option 1: Use the Directory Structure

1. Organize your FASTA files by family in a directory:
   ```
   your_virus_data_folder/
   ├── family1_viruses.fasta
   ├── family2_viruses.fasta
   └── ...
   ```

2. Run the pipeline with your data directory:
   ```bash
   python scripts/preprocess_fasta.py --input_dir my_virus_data --output_dir preprocessed_data
   ```

### Option 2: Provide a Metadata Mapping

1. Keep your FASTA files however they're organized
2. Create a metadata CSV or TSV file with the following columns:
   - `sequence_id`: Unique identifier matching FASTA headers
   - `family`: Virus family
   - `host`: Host organism
   - Other optional columns (location, isolation_source, etc.)

3. Run the pipeline with metadata:
   ```bash
   python scripts/preprocess_fasta.py --input_dir my_virus_data --output_dir preprocessed_data --metadata metadata.tsv
   ```

## Custom Classification Tasks

To classify viruses by categories other than human/non-human hosts:

1. Modify the `config/default_config.yml` file:
   ```yaml
   classification:
     # Change to your classification task
     task: "host_specificity"  # Options: "host_type", "family", "genome_type", etc.
     
     # Define your classes
     classes:
       - "human"
       - "non-human"
       # Or for family classification:
       # - "flaviviridae"
       # - "herpesviridae"
       # etc.
   ```

2. Run the pipeline with your configuration:
   ```bash
   python scripts/train_models.py --data_dir data/dataset_splits --output_dir results/models --config my_config.yml
   ```

## Example Data

This repository includes small example datasets in `data/examples/`. For full datasets, see the Resources section below.

## Resources

- [Virus-Host DB](https://www.genome.jp/virushostdb/): A database of virus-host relationships
- [NCBI Virus](https://www.ncbi.nlm.nih.gov/labs/virus/): Comprehensive virus sequence database
- [ViPR](https://www.viprbrc.org/): Virus Pathogen Resource
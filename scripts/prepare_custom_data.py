#!/usr/bin/env python3
"""
Custom data preparation script.

This script helps users prepare their own virus sequence data for use with
the virus host classification pipeline. It processes a directory of FASTA files
and optionally a metadata file to create a standardized dataset.
"""

import os
import argparse
import pandas as pd
import numpy as np
import re
from pathlib import Path
from Bio import SeqIO
from tqdm import tqdm
import yaml
import shutil


def identify_virus_family(filename, description, metadata=None, seq_id=None):
    """
    Identify virus family from filename, description or metadata.
    
    Args:
        filename (str): FASTA filename
        description (str): Sequence description
        metadata (pd.DataFrame): Optional metadata DataFrame
        seq_id (str): Sequence identifier
        
    Returns:
        str: Identified family or None
    """
    # Try to get from metadata first
    if metadata is not None and seq_id is not None:
        if seq_id in metadata.index and not pd.isna(metadata.loc[seq_id, 'family']):
            return metadata.loc[seq_id, 'family'].lower()
    
    # Try to extract from filename
    file_match = re.search(r'([a-z]+viridae)', filename.lower())
    if file_match:
        return file_match.group(1)
    
    # Try to extract from description
    desc_match = re.search(r'([a-z]+viridae)', description.lower())
    if desc_match:
        return desc_match.group(1)
    
    # Common virus name to family mapping
    virus_to_family = {
        'influenza': 'orthomyxoviridae',
        'hepatitis c': 'flaviviridae',
        'dengue': 'flaviviridae',
        'zika': 'flaviviridae',
        'herpes': 'herpesviridae',
        'hsv': 'herpesviridae',
        'cmv': 'herpesviridae',
        'ebv': 'herpesviridae',
        'adenovirus': 'adenoviridae',
        'papilloma': 'papillomaviridae',
        'hpv': 'papillomaviridae',
        'polyoma': 'polyomaviridae',
        'corona': 'coronaviridae',
        'sars': 'coronaviridae',
        'mers': 'coronaviridae',
        'covid': 'coronaviridae',
        'hiv': 'retroviridae',
        'htlv': 'retroviridae',
        'rota': 'reoviridae',
        'hepadna': 'hepadnaviridae',
        'hbv': 'hepadnaviridae',
        'parvo': 'parvoviridae',
        'rabies': 'rhabdoviridae'
    }
    
    for virus, family in virus_to_family.items():
        if virus in description.lower() or virus in filename.lower():
            return family
    
    return None


def identify_host(description, metadata=None, seq_id=None):
    """
    Identify host from description or metadata.
    
    Args:
        description (str): Sequence description
        metadata (pd.DataFrame): Optional metadata DataFrame
        seq_id (str): Sequence identifier
        
    Returns:
        str: Identified host or None
    """
    # Try to get from metadata first
    if metadata is not None and seq_id is not None:
        if seq_id in metadata.index and not pd.isna(metadata.loc[seq_id, 'host']):
            return metadata.loc[seq_id, 'host']
    
    # Try to extract from description
    host_patterns = [
        (r'host[=:]\s*([^,;|]+)', lambda m: m.group(1).strip()),
        (r'isolated from\s+([^,;|]+)', lambda m: m.group(1).strip()),
        (r'from\s+([^,;|]+)', lambda m: m.group(1).strip()),
        (r'human|homo sapiens', lambda m: 'Homo sapiens'),
        (r'chicken|gallus', lambda m: 'Gallus gallus'),
        (r'duck|anas', lambda m: 'Anas platyrhynchos'),
        (r'pig|swine|sus', lambda m: 'Sus scrofa'),
        (r'bat|pteropus', lambda m: 'Chiroptera'),
        (r'mosquito|aedes|culex', lambda m: 'Culicidae'),
        (r'mouse|mus', lambda m: 'Mus musculus'),
        (r'monkey|macaque|primate', lambda m: 'Primates')
    ]
    
    for pattern, host_func in host_patterns:
        match = re.search(pattern, description.lower())
        if match:
            return host_func(match)
    
    return None


def prepare_metadata(input_files, output_file, existing_metadata=None):
    """
    Prepare metadata from FASTA files.
    
    Args:
        input_files (list): List of input FASTA files
        output_file (str): Path to output metadata file
        existing_metadata (str): Path to existing metadata file
        
    Returns:
        pd.DataFrame: Metadata DataFrame
    """
    # Load existing metadata if provided
    metadata_df = None
    if existing_metadata:
        if existing_metadata.endswith('.tsv'):
            metadata_df = pd.read_csv(existing_metadata, sep='\t', index_col='sequence_id')
        else:
            metadata_df = pd.read_csv(existing_metadata, index_col='sequence_id')
    
    # Create metadata dictionary
    metadata = []
    
    # Process each FASTA file
    for input_file in tqdm(input_files, desc="Processing FASTA files for metadata"):
        filename = os.path.basename(input_file)
        
        for record in SeqIO.parse(input_file, "fasta"):
            seq_id = record.id
            description = record.description
            
            # Check if metadata already exists
            if metadata_df is not None and seq_id in metadata_df.index:
                # Use existing metadata
                row = metadata_df.loc[seq_id].to_dict()
                row['sequence_id'] = seq_id
                metadata.append(row)
                continue
            
            # Identify family and host
            family = identify_virus_family(filename, description, metadata_df, seq_id)
            host = identify_host(description, metadata_df, seq_id)
            
            # Create metadata entry
            entry = {
                'sequence_id': seq_id,
                'description': description,
                'source_file': filename,
                'family': family,
                'host': host,
                'sequence_length': len(record.seq)
            }
            
            metadata.append(entry)
    
    # Create DataFrame
    result_df = pd.DataFrame(metadata)
    
    # Save to file
    if output_file:
        if output_file.endswith('.tsv'):
            result_df.to_csv(output_file, sep='\t', index=False)
        else:
            result_df.to_csv(output_file, index=False)
    
    return result_df


def prepare_dataset(input_files, output_dir, metadata_file=None, organize_by='family'):
    """
    Prepare dataset from FASTA files.
    
    Args:
        input_files (list): List of input FASTA files
        output_dir (str): Output directory
        metadata_file (str): Optional metadata file
        organize_by (str): How to organize files ('family', 'host', or 'none')
        
    Returns:
        dict: Summary of processed sequences
    """
    # Load metadata if provided
    metadata_df = None
    if metadata_file:
        if metadata_file.endswith('.tsv'):
            metadata_df = pd.read_csv(metadata_file, sep='\t')
        else:
            metadata_df = pd.read_csv(metadata_file)
        metadata_df.set_index('sequence_id', inplace=True)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dictionary to track processed sequences
    summary = {}
    
    # Process each FASTA file
    for input_file in tqdm(input_files, desc="Processing FASTA files"):
        filename = os.path.basename(input_file)
        
        for record in SeqIO.parse(input_file, "fasta"):
            seq_id = record.id
            description = record.description
            
            # Determine organization category
            category = None
            if organize_by == 'family':
                category = identify_virus_family(filename, description, metadata_df, seq_id)
                if category is None:
                    category = 'unknown_family'
            elif organize_by == 'host':
                category = identify_host(description, metadata_df, seq_id)
                if category is None:
                    category = 'unknown_host'
            else:  # 'none'
                category = 'all_sequences'
            
            # Create output file path
            output_file = os.path.join(output_dir, f"{category}.fasta")
            
            # Append to output file
            with open(output_file, 'a') as f:
                SeqIO.write(record, f, "fasta")
            
            # Update summary
            if category not in summary:
                summary[category] = 0
            summary[category] += 1
    
    return summary


def main(args):
    """
    Main function to prepare custom data.
    
    Args:
        args: Command-line arguments
    """
    print(f"Preparing custom data from {args.input_dir or args.input_files}")
    
    # Get list of input files
    input_files = []
    if args.input_dir:
        for root, _, files in os.walk(args.input_dir):
            for file in files:
                if file.endswith(('.fasta', '.fa', '.fna')):
                    input_files.append(os.path.join(root, file))
    else:
        input_files = args.input_files
    
    print(f"Found {len(input_files)} FASTA files")
    
    # Prepare metadata
    if args.prepare_metadata:
        print("Preparing metadata...")
        metadata_df = prepare_metadata(input_files, args.output_metadata, args.existing_metadata)
        print(f"Created metadata with {len(metadata_df)} entries")
        
        # Display metadata summary
        print("\nMetadata Summary:")
        
        # Family summary
        if 'family' in metadata_df.columns:
            family_counts = metadata_df['family'].value_counts()
            print("Top families:")
            for family, count in family_counts.head(10).items():
                print(f"  {family}: {count}")
        
        # Host summary
        if 'host' in metadata_df.columns:
            host_counts = metadata_df['host'].value_counts()
            print("\nTop hosts:")
            for host, count in host_counts.head(10).items():
                print(f"  {host}: {count}")
    
    # Prepare dataset
    if args.prepare_dataset:
        print("\nPreparing dataset...")
        metadata_file = args.output_metadata if args.prepare_metadata else args.existing_metadata
        summary = prepare_dataset(input_files, args.output_dir, metadata_file, args.organize_by)
        
        print("\nDataset Summary:")
        total_sequences = sum(summary.values())
        print(f"Total sequences: {total_sequences}")
        print("Sequences by category:")
        for category, count in summary.items():
            print(f"  {category}: {count} ({count/total_sequences:.2%})")
    
    # Create pipeline running script
    if args.create_pipeline_script:
        script_path = os.path.join(args.output_dir, "run_pipeline.sh")
        with open(script_path, 'w') as f:
            f.write("#!/bin/bash\n\n")
            f.write("# Auto-generated pipeline script\n\n")
            
            # Preprocess step
            f.write("# Step 1: Preprocess FASTA files\n")
            f.write(f"python ../../scripts/preprocess_fasta.py --input_dir . --output_dir preprocessed_data")
            if args.existing_metadata or args.output_metadata:
                metadata_path = args.output_metadata if args.prepare_metadata else args.existing_metadata
                f.write(f" --metadata {os.path.basename(metadata_path)}")
            f.write(" --exclusion_keywords partial,mutant,unverified,bac,clone --length_filter\n\n")
            
            # Feature generation step
            f.write("# Step 2: Generate k-mer features\n")
            f.write("python ../../scripts/generate_features.py --input_dir preprocessed_data --output_dir features_data --k_values 3,4,5,6\n\n")
            
            # Split dataset step
            f.write("# Step 3: Split dataset\n")
            f.write("python ../../scripts/split_dataset.py --input_dir features_data --output_dir split_data --k_values 3,4,5,6\n\n")
            
            # Train models step
            f.write("# Step 4: Train models\n")
            f.write("python ../../scripts/train_models.py --data_dir split_data --output_dir results --kmers 3 4 5 6\n\n")
            
            # Visualization step
            f.write("# Step 5: Visualize results\n")
            f.write("python ../../scripts/visualize_tsne.py --h5_path split_data/k6/train.h5 --model_path results/k6/nn/model.pt --scaler_path results/k6/nn/scaler.pkl --output_dir visualizations\n")
        
        # Make the script executable
        os.chmod(script_path, 0o755)
        print(f"\nCreated pipeline script: {script_path}")
    
    print("\nCustom data preparation complete!")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Prepare custom data for virus host classification")
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_dir", type=str,
                          help="Directory containing input FASTA files")
    input_group.add_argument("--input_files", type=str, nargs='+',
                          help="List of input FASTA files")
    
    # Output options
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory to save prepared data")
    
    # Metadata options
    parser.add_argument("--prepare_metadata", action="store_true",
                      help="Prepare metadata from FASTA files")
    parser.add_argument("--existing_metadata", type=str,
                      help="Path to existing metadata file (CSV or TSV)")
    parser.add_argument("--output_metadata", type=str,
                      help="Path to save generated metadata (CSV or TSV)")
    
    # Dataset options
    parser.add_argument("--prepare_dataset", action="store_true",
                      help="Prepare dataset from FASTA files")
    parser.add_argument("--organize_by", type=str, default="family",
                      choices=["family", "host", "none"],
                      help="How to organize output files")
    
    # Other options
    parser.add_argument("--create_pipeline_script", action="store_true",
                      help="Create a shell script to run the entire pipeline")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Validate arguments
    if args.prepare_metadata and not args.output_metadata:
        args.output_metadata = os.path.join(args.output_dir, "metadata.tsv")
    
    main(args)
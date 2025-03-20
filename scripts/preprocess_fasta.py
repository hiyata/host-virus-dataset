#!/usr/bin/env python3
"""
FASTA preprocessing script.

This script preprocesses FASTA files by filtering sequences based on
description keywords and nucleotide composition.
"""

import os
import argparse
from Bio import SeqIO
from tqdm import tqdm
from pathlib import Path


def process_fasta_files(input_dir, output_dir):
    """
    Process FASTA files by filtering sequences based on description and composition.
    
    Args:
        input_dir (str): Directory containing input FASTA files
        output_dir (str): Directory to write filtered FASTA files
        
    Returns:
        dict: Dictionary with family names as keys and counts of filtered sequences as values
    """
    exclusion_keywords = ['partial', 'mutant', 'unverified', 'bac', 'clone']
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Use the provided input_dir instead of hardcoded 'raw_fastas'
    fasta_files = [f for f in os.listdir(input_dir) if f.endswith('.fasta')]
    results = {}
    
    # Process each file
    for fasta_file in tqdm(fasta_files, desc="Processing FASTA files"):
        family_name = fasta_file.split('_')[0]
        input_path = os.path.join(input_dir, fasta_file)
        output_path = os.path.join(output_dir, f"{family_name}_filtered.fasta")
        
        # Store filtered records
        filtered_records = []
        
        # Filter sequences
        for record in SeqIO.parse(input_path, "fasta"):
            # Check for exclusion keywords
            if not any(keyword in record.description.lower() for keyword in exclusion_keywords):
                # Check sequence composition
                sequence = str(record.seq).upper()
                if set(sequence).issubset({'A', 'C', 'G', 'T'}):
                    filtered_records.append(record)
        
        # Write filtered sequences
        if filtered_records:
            SeqIO.write(filtered_records, output_path, "fasta")
            results[family_name] = len(filtered_records)
            print(f"Processed {family_name}: {len(filtered_records)} sequences")
    
    return results


def filter_sequence_length(input_dir, output_dir, min_length=500, max_length=50000):
    """
    Further filter FASTA files based on sequence length.
    
    Args:
        input_dir (str): Directory containing input FASTA files
        output_dir (str): Directory to write length-filtered FASTA files
        min_length (int): Minimum sequence length
        max_length (int): Maximum sequence length
        
    Returns:
        dict: Dictionary with file names as keys and counts of length-filtered sequences as values
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fasta_files = [f for f in os.listdir(input_dir) if f.endswith('.fasta')]
    results = {}
    
    for fasta_file in tqdm(fasta_files, desc="Length filtering"):
        input_path = os.path.join(input_dir, fasta_file)
        output_path = os.path.join(output_dir, fasta_file)
        
        length_filtered = []
        
        for record in SeqIO.parse(input_path, "fasta"):
            seq_len = len(record.seq)
            if min_length <= seq_len <= max_length:
                length_filtered.append(record)
        
        if length_filtered:
            SeqIO.write(length_filtered, output_path, "fasta")
            results[fasta_file] = len(length_filtered)
    
    return results


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Preprocess FASTA files for virus host classification")
    parser.add_argument("--input_dir", type=str, required=True,
                      help="Directory containing input FASTA files")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory to save preprocessed FASTA files")
    parser.add_argument("--metadata", type=str,
                      help="Metadata file (optional)")
    parser.add_argument("--exclusion_keywords", type=str, default="partial,mutant,unverified,bac,clone",
                      help="Comma-separated list of keywords to exclude")
    parser.add_argument("--length_filter", action="store_true",
                      help="Apply sequence length filtering")
    parser.add_argument("--min_length", type=int, default=500,
                      help="Minimum sequence length")
    parser.add_argument("--max_length", type=int, default=50000,
                      help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process FASTA files
    print(f"Preprocessing FASTA files from {args.input_dir}")
    results = process_fasta_files(args.input_dir, args.output_dir)
    
    # Print summary
    print("\nPreprocessing Summary:")
    total_sequences = sum(results.values())
    print(f"Total processed files: {len(results)}")
    print(f"Total filtered sequences: {total_sequences}")
    
    # Apply length filtering if requested
    if args.length_filter:
        length_filtered_dir = os.path.join(args.output_dir, "length_filtered")
        length_results = filter_sequence_length(
            args.output_dir,
            length_filtered_dir,
            min_length=args.min_length,
            max_length=args.max_length
        )
        
        print("\nLength Filtering Summary:")
        total_length_filtered = sum(length_results.values())
        print(f"Total length-filtered files: {len(length_results)}")
        print(f"Total length-filtered sequences: {total_length_filtered}")
        print(f"Sequences retained after length filtering: {total_length_filtered / total_sequences:.2%}")
    
    print(f"\nAll preprocessing complete. Results saved to {args.output_dir}")
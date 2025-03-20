"""
Preprocessing module for FASTA files.

This module provides functions to preprocess virus FASTA files by:
1. Filtering out sequences with unwanted keywords
2. Ensuring sequences contain only valid nucleotides (A, C, G, T)
"""

import os
from Bio import SeqIO
from tqdm import tqdm
from pathlib import Path


def process_fasta_files(input_dir, output_dir, exclusion_keywords=None):
    """
    Process FASTA files by filtering sequences based on description and composition.
    
    Args:
        input_dir (str): Directory containing input FASTA files
        output_dir (str): Directory to write filtered FASTA files
        exclusion_keywords (list): List of keywords to exclude; if None, defaults 
                                  to ['partial', 'mutant', 'unverified', 'bac', 'clone']
    
    Returns:
        dict: Dictionary with family names as keys and counts of filtered sequences as values
    """
    if exclusion_keywords is None:
        exclusion_keywords = ['partial', 'mutant', 'unverified', 'bac', 'clone']
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
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
    # Example usage
    input_directory = "raw_fastas"
    output_directory = "filtered_fastas"
    process_fasta_files(input_directory, output_directory)
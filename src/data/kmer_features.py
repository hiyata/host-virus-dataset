"""
K-mer feature extraction module.

This module provides functions to calculate k-mer frequencies from virus sequences
and store them in HDF5 format.
"""

import os
import h5py
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from itertools import product


def generate_kmer_dict(k):
    """
    Generate a dictionary mapping k-mers to indices.
    
    Args:
        k (int): K-mer size
        
    Returns:
        dict: Dictionary mapping k-mers to indices
    """
    return {''.join(kmer): i for i, kmer in enumerate(product('ACGT', repeat=k))}


def calculate_kmer_freq(seq, k, kmer_dict):
    """
    Calculate k-mer frequency for a given sequence.
    
    Args:
        seq (str): DNA sequence
        k (int): K-mer size
        kmer_dict (dict): Dictionary mapping k-mers to indices
        
    Returns:
        numpy.ndarray: Array of k-mer frequencies
    """
    freq = np.zeros(4**k)
    for i in range(len(seq)-k+1):
        kmer = seq[i:i+k]
        if 'N' not in kmer:
            freq[kmer_dict[kmer]] += 1
    return freq/(len(seq)-k+1) if len(seq)-k+1 > 0 else freq


def process_fasta_to_h5(input_dir, output_dir, k_values=None, genome_types=None):
    """
    Process FASTA files to extract k-mer features and store in HDF5 format.
    
    Args:
        input_dir (str): Directory containing filtered FASTA files
        output_dir (str): Directory to write HDF5 files
        k_values (list): List of k values to process; if None, defaults to [3,4,5,6,7,8]
        genome_types (dict): Dictionary mapping virus families to genome types
        
    Returns:
        dict: Dictionary with k values as keys and counts of processed sequences as values
    """
    if k_values is None:
        k_values = [3, 4, 5, 6, 7, 8]
    
    if genome_types is None:
        genome_types = {
            'astroviridae': 'ssRNA(+)',
            'picornaviridae': 'ssRNA(+)',
            'flaviviridae': 'ssRNA(+)',
            'sedoreoviridae': 'dsRNA',
            'spinareoviridae': 'dsRNA',
            'parvoviridae': 'ssDNA',
            'togaviridae': 'ssRNA(+)',
            'adenoviridae': 'dsDNA',
            'orthoherpesviridae': 'dsDNA',
            'orthomyxoviridae': 'ssRNA(-)',
            'papillomaviridae': 'dsDNA',
            'polyomaviridae': 'dsDNA',
            'poxviridae': 'dsDNA',
            'hepadnaviridae': 'dsDNA-RT',
            'rhabdoviridae': 'ssRNA(-)'
        }
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create kmer dictionaries
    kmer_dicts = {k: generate_kmer_dict(k) for k in k_values}
    results = {}
    
    for k in k_values:
        output_file = os.path.join(output_dir, f'viral_dataset_k{k}.h5')
        
        with h5py.File(output_file, 'w') as h5f:
            # Create groups for different data types
            sequences_group = h5f.create_group('sequences')
            metadata_group = h5f.create_group('metadata')
            features_group = h5f.create_group('features')
            
            total_seqs = 0
            seq_ids = []
            families = []
            hosts = []
            genome_types_list = []
            
            # First pass to get total sequences
            fasta_files = [f for f in os.listdir(input_dir)
                          if f.endswith('_filtered.fasta')]
            
            for fasta_file in fasta_files:
                total_seqs += sum(1 for _ in SeqIO.parse(
                    os.path.join(input_dir, fasta_file), "fasta"))
            
            # Create datasets
            kmer_features = features_group.create_dataset(
                f'kmer_{k}', shape=(total_seqs, 4**k), dtype=np.float32)
            
            current_idx = 0
            
            # Process each file
            for fasta_file in tqdm(fasta_files, desc=f"Processing k={k}"):
                family_name = fasta_file.split('_')[0].lower()
                file_path = os.path.join(input_dir, fasta_file)
                
                for record in SeqIO.parse(file_path, "fasta"):
                    sequence = str(record.seq).upper()
                    seq_id = record.id
                    
                    # Calculate k-mer frequencies
                    freq = calculate_kmer_freq(sequence, k, kmer_dicts[k])
                    kmer_features[current_idx] = freq
                    
                    # Store metadata
                    seq_ids.append(seq_id)
                    families.append(family_name)
                    host = 'human' if any(h in record.description.lower()
                           for h in ['homo sapiens', 'human']) else 'non-human'
                    hosts.append(host)
                    genome_types_list.append(genome_types.get(family_name, 'unknown'))
                    
                    current_idx += 1
            
            # Store metadata
            metadata_group.create_dataset('sequence_ids',
                data=np.array(seq_ids, dtype='S'))
            metadata_group.create_dataset('families',
                data=np.array(families, dtype='S'))
            metadata_group.create_dataset('hosts',
                data=np.array(hosts, dtype='S'))
            metadata_group.create_dataset('genome_types',
                data=np.array(genome_types_list, dtype='S'))
            
            # Store kmer dict as attributes
            for kmer, idx in kmer_dicts[k].items():
                features_group.attrs[f'kmer_{kmer}'] = idx
                
        results[k] = current_idx
    
    return results


if __name__ == "__main__":
    input_directory = "filtered_fastas"
    output_directory = "kmer_features"
    process_fasta_to_h5(input_directory, output_directory)
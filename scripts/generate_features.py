#!/usr/bin/env python3
"""
Feature generation script.

This script generates k-mer frequency features from preprocessed FASTA files
and stores them in HDF5 format. Optionally, if a metadata file is provided,
it will use the metadata to set the host label rather than relying solely on
simple keyword extraction.
"""

import os
import argparse
import yaml
from pathlib import Path
import h5py
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
from itertools import product
import re
import pandas as pd

def canonicalize(seq_id):
    """
    Return a canonical version of a sequence ID by converting to lower-case,
    stripping whitespace, and removing trailing version numbers (e.g. ".1").
    """
    return re.sub(r'\.\d+$', '', str(seq_id).lower().strip())

def generate_kmer_dict(k):
    """Generate a dictionary mapping k-mers to indices."""
    return {''.join(kmer): i for i, kmer in enumerate(product('ACGT', repeat=k))}

def calculate_kmer_freq(seq, k, kmer_dict):
    """Calculate k-mer frequency for a given sequence."""
    freq = np.zeros(4**k)
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if 'N' not in kmer:
            freq[kmer_dict[kmer]] += 1
    return freq / (len(seq) - k + 1) if (len(seq) - k + 1) > 0 else freq

def load_genome_types(config_file):
    """Load genome types from configuration file."""
    if not config_file or not os.path.exists(config_file):
        print("No config file provided or file not found. Using default genome types.")
        return {
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
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('genome_types', {})
    except Exception as e:
        print(f"Error loading config file: {str(e)}")
        return {}

def load_metadata(metadata_file):
    """
    Load processed metadata to build an accession-to-host mapping.
    
    Returns:
        dict: Mapping from canonical accession (without version) to host label.
    """
    if not os.path.exists(metadata_file):
        print(f"Metadata file {metadata_file} not found.")
        return {}
    df = pd.read_csv(metadata_file, sep="\t")
    if "accession" not in df.columns or "host" not in df.columns:
        print("Metadata file must contain 'accession' and 'host' columns.")
        return {}
    mapping = {}
    for _, row in df.iterrows():
        acc = str(row["accession"]).strip()
        m = re.match(r'^([A-Z]{1,2}\d{5,8})(\.\d+)?$', acc)
        if m:
            mapping[m.group(1).lower()] = str(row["host"]).strip().lower()
        else:
            mapping[acc.lower()] = str(row["host"]).strip().lower()
    return mapping

def extract_host(record, metadata_mapping=None):
    """
    Extract the host for a FASTA record.
    If a metadata mapping is provided, use the canonical accession from the record ID.
    Otherwise, use simple keyword matching of the description.
    """
    m = re.match(r'^([A-Z]{1,2}\d{5,8})(\.\d+)?', record.id)
    if m:
        accession = m.group(1).lower()
        if metadata_mapping and accession in metadata_mapping:
            return metadata_mapping[accession]
    description = record.description.lower()
    if 'human' in description or 'homo sapiens' in description:
        return 'human'
    else:
        # For training, we force binary classification: non-human if not human.
        return 'non-human'

def process_fasta_to_h5(input_dir, output_dir, k_values=None, genome_types=None, metadata_mapping=None):
    """
    Process FASTA files to extract k-mer features and store them in HDF5 format.
    """
    if k_values is None:
        k_values = [3, 4, 5, 6, 7, 8]
    if genome_types is None:
        genome_types = {}
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    kmer_dicts = {k: generate_kmer_dict(k) for k in k_values}
    results = {}
    
    print("Scanning FASTA files...")
    total_seqs = 0
    fasta_files = [f for f in os.listdir(input_dir) if f.endswith('.fasta')]
    for fasta_file in fasta_files:
        file_path = os.path.join(input_dir, fasta_file)
        total_seqs += sum(1 for _ in SeqIO.parse(file_path, "fasta"))
    print(f"Found {total_seqs} sequences in {len(fasta_files)} files")
    
    for k in k_values:
        print(f"Processing k={k}...")
        output_file = os.path.join(output_dir, f'viral_dataset_k{k}.h5')
        with h5py.File(output_file, 'w') as h5f:
            features = h5f.create_dataset('features', shape=(total_seqs, 4**k), dtype=np.float32)
            seq_ids = h5f.create_dataset('sequence_ids', shape=(total_seqs,), dtype='S100')
            families = h5f.create_dataset('families', shape=(total_seqs,), dtype='S50')
            hosts = h5f.create_dataset('hosts', shape=(total_seqs,), dtype='S50')
            genome_types_list = h5f.create_dataset('genome_types', shape=(total_seqs,), dtype='S20')
            current_idx = 0
            for fasta_file in tqdm(fasta_files, desc=f"Processing k={k}"):
                family_name = fasta_file.split('_')[0].lower()
                file_path = os.path.join(input_dir, fasta_file)
                for record in SeqIO.parse(file_path, "fasta"):
                    sequence = str(record.seq).upper()
                    if len(sequence) < k:
                        continue
                    freq = calculate_kmer_freq(sequence, k, kmer_dicts[k])
                    features[current_idx] = freq
                    seq_ids[current_idx] = canonicalize(record.id).encode('utf-8')
                    families[current_idx] = family_name.encode('utf-8')
                    host = extract_host(record, metadata_mapping)
                    hosts[current_idx] = host.encode('utf-8')
                    genome_types_list[current_idx] = genome_types.get(family_name, 'unknown').encode('utf-8')
                    current_idx += 1
            if current_idx < total_seqs:
                print(f"Resizing datasets from {total_seqs} to {current_idx} (some sequences may have been skipped)")
                features.resize((current_idx, 4**k))
                seq_ids.resize((current_idx,))
                families.resize((current_idx,))
                hosts.resize((current_idx,))
                genome_types_list.resize((current_idx,))
            for kmer, idx in kmer_dicts[k].items():
                h5f.attrs[f'kmer_{kmer}'] = idx
        results[k] = current_idx
        print(f"Completed k={k}: {current_idx} sequences processed")
    return results

def main(args):
    genome_types = load_genome_types(args.config)
    k_values = [int(k) for k in args.k_values.split(',')]
    print(f"Generating k-mer features for k = {k_values}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    metadata_mapping = None
    if args.metadata:
        metadata_mapping = load_metadata(args.metadata)
        if metadata_mapping:
            print(f"Loaded metadata mapping with {len(metadata_mapping)} entries.")
    results = process_fasta_to_h5(
        args.input_dir,
        args.output_dir,
        k_values=k_values,
        genome_types=genome_types,
        metadata_mapping=metadata_mapping
    )
    print("\nFeature Generation Summary:")
    for k, count in results.items():
        print(f"k={k}: {count} sequences processed")
    print(f"\nOutput HDF5 files saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate k-mer features for virus host classification")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing preprocessed FASTA files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save HDF5 feature files")
    parser.add_argument("--k_values", type=str, default="3,4,5,6,7,8", help="Comma-separated list of k-mer sizes")
    parser.add_argument("--config", type=str, help="Path to configuration file with genome types")
    parser.add_argument("--metadata", type=str, help="(Optional) Path to processed metadata TSV file")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)

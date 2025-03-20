#!/usr/bin/env python3
"""
Dataset splitting script.

This script splits the dataset into train, validation, and test sets
while preserving stratification by family and host.
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import h5py
from sklearn.model_selection import train_test_split
import re

def canonicalize(seq_id):
    """
    Return a canonical version of a sequence ID by converting to lower-case,
    stripping whitespace, and removing trailing version numbers.
    """
    if isinstance(seq_id, bytes):
        seq_id = seq_id.decode('utf-8')
    return re.sub(r'\.\d+$', '', str(seq_id).lower().strip())

def load_h5_data(h5_path):
    """
    Load data from an HDF5 file.
    
    Returns:
        tuple: (features, sequence_ids, families, hosts)
    """
    with h5py.File(h5_path, 'r') as h5f:
        # Find features dataset
        if 'features' in h5f:
            features = h5f['features'][:]
        else:
            # Look in features group
            if 'features' in h5f:
                features_group = h5f['features']
                feature_keys = list(features_group.keys())
                if feature_keys:
                    features = features_group[feature_keys[0]][:]
                else:
                    raise ValueError(f"Could not find features dataset in {h5_path}")
            else:
                # Try to find features at the root level
                feature_keys = [k for k in h5f.keys() if 'feature' in k.lower() or 'kmer' in k.lower()]
                if feature_keys:
                    features = h5f[feature_keys[0]][:]
                else:
                    raise ValueError(f"Could not find features dataset in {h5_path}")
        
        # Load metadata
        if 'metadata' in h5f and 'sequence_ids' in h5f['metadata']:
            seq_ids = [s.decode('utf-8') if isinstance(s, bytes) else s for s in h5f['metadata']['sequence_ids'][:]]
            families = [s.decode('utf-8') if isinstance(s, bytes) else s for s in h5f['metadata']['families'][:]]
            hosts = [s.decode('utf-8') if isinstance(s, bytes) else s for s in h5f['metadata']['hosts'][:]]
        else:
            # Direct in root
            if 'sequence_ids' in h5f:
                seq_ids = [s.decode('utf-8') if isinstance(s, bytes) else s for s in h5f['sequence_ids'][:]]
            else:
                seq_ids = [f"seq_{i}" for i in range(features.shape[0])]
                
            if 'families' in h5f:
                families = [s.decode('utf-8') if isinstance(s, bytes) else s for s in h5f['families'][:]]
            else:
                families = ["unknown"] * features.shape[0]
                
            if 'hosts' in h5f:
                hosts = [s.decode('utf-8') if isinstance(s, bytes) else s for s in h5f['hosts'][:]]
            else:
                hosts = ["unknown"] * features.shape[0]
    
    # Ensure strings
    seq_ids = [str(s).strip() for s in seq_ids]
    families = [str(f).strip() for f in families]
    hosts = [str(h).strip() for h in hosts]
    
    print(f"Loaded {len(seq_ids)} sequences from {h5_path}")
    
    return features, seq_ids, families, hosts

def load_metadata(metadata_file):
    """
    Load processed metadata and return a mapping from canonical accession to host.
    """
    if not os.path.exists(metadata_file):
        print(f"Metadata file {metadata_file} not found.")
        return {}
    
    df = pd.read_csv(metadata_file, sep="\t")
    
    if "accession" not in df.columns or "host" not in df.columns:
        print("Metadata file must contain 'accession' and 'host' columns.")
        return {}
    
    # Create both strict and flexible mappings
    mapping = {}
    for _, row in df.iterrows():
        acc = str(row["accession"]).strip()
        host = str(row["host"]).strip().lower()
        
        # Store multiple versions of the accession for flexible matching
        mapping[acc.lower()] = host
        
        # Remove version number if present
        base_acc = re.sub(r'\.\d+$', '', acc.lower())
        mapping[base_acc] = host
        
        # Extract base accession number if it follows standard pattern
        m = re.match(r'^([A-Z]{1,2}\d{5,8})(\.\d+)?$', acc)
        if m:
            mapping[m.group(1).lower()] = host
    
    print(f"Loaded {len(mapping)} accession-to-host mappings from metadata")
    return mapping

def stratified_split(seq_ids, families, hosts, proportions=None):
    """
    Split the dataset while preserving stratification by family and host.
    
    Returns:
        dict: Mapping from split name to list of sequence IDs.
    """
    if proportions is None:
        proportions = {'train': 0.8, 'validate': 0.1, 'test': 0.1}
    
    df = pd.DataFrame({
        'seq_id': seq_ids,
        'family': families,
        'host': hosts
    })
    
    # Standardize values
    df['family'] = df['family'].str.lower()
    df['host'] = df['host'].str.lower()
    
    # Create stratification groups
    df['group'] = df['family'] + '_' + df['host']
    
    # Print group statistics
    print(f"Found {len(df['group'].unique())} unique family-host groups")
    for group, count in df['group'].value_counts().items():
        print(f"  Group '{group}': {count} sequences")
    
    splits = {split: [] for split in proportions.keys()}
    
    # Handle each group separately
    for group, group_df in df.groupby('group'):
        group_ids = group_df['seq_id'].tolist()
        
        if len(group_ids) < 3:
            # Put all in training if too few samples
            print(f"Group '{group}' has only {len(group_ids)} sequences, adding all to training")
            splits['train'].extend(group_ids)
            continue
        
        # Calculate split sizes
        train_size = max(1, int(len(group_ids) * proportions['train']))
        val_size = max(1, int(len(group_ids) * proportions['validate']))
        
        # Split into train and remaining
        train_ids, rest_ids = train_test_split(
            group_ids, 
            train_size=train_size, 
            random_state=42
        )
        
        splits['train'].extend(train_ids)
        
        if len(rest_ids) > 1:
            # Split remaining into validate and test
            val_ids, test_ids = train_test_split(
                rest_ids,
                train_size=val_size/(val_size + len(rest_ids) - val_size),  # Convert to proportion
                random_state=42
            )
            splits['validate'].extend(val_ids)
            splits['test'].extend(test_ids)
        else:
            # Put all remaining in validate if only one left
            splits['validate'].extend(rest_ids)
    
    # Ensure each split has at least 2 samples for batch normalization
    for split in splits:
        if len(splits[split]) < 2:
            print(f"WARNING: Split '{split}' has only {len(splits[split])} samples, which is not enough for batch normalization")
            # Duplicate the sample if only one exists
            if len(splits[split]) == 1:
                splits[split].append(splits[split][0])
                print(f"  Duplicated the single sample in '{split}' to ensure batch normalization works")
    
    print("Split sizes:")
    for split, ids in splits.items():
        print(f"  {split}: {len(ids)} sequences")
    
    return splits

def process_splits(input_dir, output_dir, k_values=None, proportions=None, metadata_file=None):
    """
    Process the dataset into train, validation, and test splits.
    
    If a metadata file is provided, update the host labels using the mapping so that
    every sequence is either "human" or "non-human".
    
    Returns:
        pd.DataFrame: Split metadata as a DataFrame.
    """
    if k_values is None:
        k_values = [3, 4, 5, 6, 7, 8]
    if proportions is None:
        proportions = {'train': 0.8, 'validate': 0.1, 'test': 0.1}
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    metadata_mapping = load_metadata(metadata_file) if metadata_file else {}

    # Find HDF5 files for each specified k
    h5_files = []
    for k in k_values:
        file_path = os.path.join(input_dir, f'viral_dataset_k{k}.h5')
        if os.path.exists(file_path):
            h5_files.append((k, file_path))
    
    if not h5_files:
        raise FileNotFoundError(f"No HDF5 files found in {input_dir} for k values {k_values}")
    
    # Use the first file for splitting
    k_first, file_first = h5_files[0]
    features, seq_ids, families, hosts = load_h5_data(file_first)
    
    # Update host labels with metadata mapping if provided
    if metadata_mapping:
        updated_hosts = []
        for seq_id, host in zip(seq_ids, hosts):
            # Try multiple ways of matching sequence IDs
            canon_id = canonicalize(seq_id)
            if canon_id in metadata_mapping:
                updated_hosts.append(metadata_mapping[canon_id])
            elif seq_id in metadata_mapping:
                updated_hosts.append(metadata_mapping[seq_id])
            else:
                # If no match found, use the original host
                updated_hosts.append(host)
        hosts = updated_hosts
    
    print(f"Creating splits for {len(seq_ids)} sequences...")
    splits = stratified_split(seq_ids, families, hosts, proportions)
    
    # Create a metadata CSV
    metadata = []
    for split_name, split_ids in splits.items():
        for seq_id in split_ids:
            # Find the index in the original data
            try:
                idx = seq_ids.index(seq_id)
                metadata.append({
                    'sequence_id': seq_id,
                    'split': split_name,
                    'family': families[idx],
                    'host': hosts[idx]
                })
            except ValueError:
                # If not found (shouldn't happen), log a warning
                print(f"Warning: Sequence ID {seq_id} not found in original data")
    
    metadata_df = pd.DataFrame(metadata)
    metadata_csv = os.path.join(output_dir, "split_metadata.csv")
    metadata_df.to_csv(metadata_csv, index=False)
    
    # Process each k-mer file
    for k, file_path in h5_files:
        print(f"Processing k={k}...")
        k_dir = Path(output_dir) / f"k{k}"
        k_dir.mkdir(exist_ok=True)
        
        # Load the data for this k-mer
        features, file_seq_ids, file_families, file_hosts = load_h5_data(file_path)
        
        # Create a mapping for fast lookup
        id_mapping = {}
        for i, seq_id in enumerate(file_seq_ids):
            canon = canonicalize(seq_id)
            id_mapping[canon] = i
            id_mapping[seq_id] = i
        
        # Update host labels if metadata mapping is provided
        if metadata_mapping:
            updated_hosts = []
            for seq_id, host in zip(file_seq_ids, file_hosts):
                canon_id = canonicalize(seq_id)
                if canon_id in metadata_mapping:
                    updated_hosts.append(metadata_mapping[canon_id])
                elif seq_id in metadata_mapping:
                    updated_hosts.append(metadata_mapping[seq_id])
                else:
                    updated_hosts.append(host)
            file_hosts = updated_hosts
        
        # Process each split
        for split_name, split_ids in splits.items():
            print(f"  Creating {split_name} split...")
            split_indices = []
            
            # Match sequence IDs to indices
            for seq_id in split_ids:
                # Try multiple ways to match
                if seq_id in id_mapping:
                    split_indices.append(id_mapping[seq_id])
                else:
                    canon = canonicalize(seq_id)
                    if canon in id_mapping:
                        split_indices.append(id_mapping[canon])
            
            # Ensure at least 2 samples for batch normalization
            if len(split_indices) < 2 and len(split_indices) > 0:
                split_indices.append(split_indices[0])  # Duplicate first sample
            
            if not split_indices:
                print(f"  Warning: No sequences for {split_name} split")
                # Add at least 2 samples to ensure batch normalization works
                if len(file_seq_ids) >= 2:
                    split_indices = [0, 1]  # Use first two samples
                    print(f"  Added first two samples to ensure split is not empty")
                elif len(file_seq_ids) == 1:
                    split_indices = [0, 0]  # Duplicate the single sample
                    print(f"  Added duplicate of single sample to ensure split is not empty")
                else:
                    continue  # Skip if no data available
            
            # Create HDF5 file for this split
            with h5py.File(k_dir / f"{split_name}.h5", 'w') as h5f:
                h5f.create_dataset('features', data=features[split_indices])
                
                # Convert to bytes for HDF5 compatibility
                seq_ids_bytes = np.array([str(file_seq_ids[i]).encode('utf-8') for i in split_indices])
                families_bytes = np.array([str(file_families[i]).encode('utf-8') for i in split_indices])
                hosts_bytes = np.array([str(file_hosts[i]).encode('utf-8') for i in split_indices])
                
                h5f.create_dataset('sequence_ids', data=seq_ids_bytes)
                h5f.create_dataset('families', data=families_bytes)
                h5f.create_dataset('hosts', data=hosts_bytes)
                
                print(f"    Created {split_name}.h5 with {len(split_indices)} sequences")
    
    print("\nSplit Summary:")
    for split_name, split_ids in splits.items():
        print(f"{split_name}: {len(split_ids)} sequences ({len(split_ids)/len(seq_ids):.2%})")
    
    return metadata_df

def main(args):
    k_values = [int(k) for k in args.k_values.split(',')]
    proportions = {
        'train': args.train_ratio,
        'validate': args.val_ratio,
        'test': args.test_ratio
    }
    total = sum(proportions.values())
    proportions = {k: v / total for k, v in proportions.items()}
    
    print(f"Splitting dataset for k = {k_values}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Split ratios: {proportions}")
    
    metadata_df = process_splits(
        args.input_dir, 
        args.output_dir, 
        k_values=k_values, 
        proportions=proportions, 
        metadata_file=args.metadata
    )
    
    print(f"\nSplit metadata saved to {os.path.join(args.output_dir, 'split_metadata.csv')}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset for virus host classification")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing HDF5 feature files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save split files")
    parser.add_argument("--k_values", type=str, default="3,4,5,6,7,8", help="Comma-separated list of k-mer sizes")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training data")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Ratio of validation data")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Ratio of test data")
    parser.add_argument("--metadata", type=str, help="Path to processed metadata file (optional)")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
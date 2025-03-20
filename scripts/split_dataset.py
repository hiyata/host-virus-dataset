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


def load_h5_data(h5_path):
    """
    Load data from HDF5 file.
    
    Args:
        h5_path (str): Path to the HDF5 file
        
    Returns:
        tuple: Tuple containing (features, sequence IDs, families, hosts)
    """
    with h5py.File(h5_path, 'r') as h5f:
        # Get features (check for different formats)
        if 'features' in h5f and isinstance(h5f['features'], h5py.Dataset):
            # Direct features dataset
            features = h5f['features'][:]
        elif 'features' in h5f and isinstance(h5f['features'], h5py.Group):
            # Features in a group
            feature_group = h5f['features']
            if 'kmer_5' in feature_group:
                # If kmer_5 exists, use it
                features = feature_group['kmer_5'][:]
            else:
                # Otherwise use the first dataset
                feature_keys = list(feature_group.keys())
                if feature_keys:
                    features = feature_group[feature_keys[0]][:]
                else:
                    raise ValueError(f"No datasets found in features group in {h5_path}")
        else:
            # Look for any feature-like datasets at the root
            feature_keys = [k for k in h5f.keys() if any(term in k.lower() for term in ['feature', 'kmer'])]
            if feature_keys:
                features = h5f[feature_keys[0]][:]
            else:
                raise ValueError(f"Could not find features dataset in {h5_path}")
            
        # Get metadata - check both root and metadata group
        if 'metadata' in h5f and 'sequence_ids' in h5f['metadata']:
            # Get from metadata group
            metadata = h5f['metadata']
            seq_ids = [s.decode('utf-8') if isinstance(s, bytes) else s for s in metadata['sequence_ids'][:]]
            families = [f.decode('utf-8') if isinstance(f, bytes) else f for f in metadata['families'][:]]
            hosts = [h.decode('utf-8') if isinstance(h, bytes) else h for h in metadata['hosts'][:]]
        else:
            # Look in root
            if 'sequence_ids' in h5f:
                seq_ids = [s.decode('utf-8') if isinstance(s, bytes) else s for s in h5f['sequence_ids'][:]]
            else:
                seq_ids = [f"seq_{i}" for i in range(features.shape[0])]
                
            if 'families' in h5f:
                families = [f.decode('utf-8') if isinstance(f, bytes) else f for f in h5f['families'][:]]
            else:
                families = ["unknown"] * features.shape[0]
                
            if 'hosts' in h5f:
                hosts = [h.decode('utf-8') if isinstance(h, bytes) else h for h in h5f['hosts'][:]]
            else:
                hosts = ["unknown"] * features.shape[0]
    
    print(f"Loaded {len(seq_ids)} sequences from {h5_path}")
    return features, seq_ids, families, hosts


def stratified_split(seq_ids, families, hosts, proportions):
    """
    Split the dataset while preserving family and host distributions.
    
    Args:
        seq_ids (list): List of sequence IDs
        families (list): List of family labels
        hosts (list): List of host labels
        proportions (dict): Dictionary with split proportions
        
    Returns:
        dict: Dictionary with sequence IDs by split
    """
    # Create DataFrame with metadata
    df = pd.DataFrame({
        'seq_id': seq_ids,
        'family': families,
        'host': hosts
    })
    
    # Create stratification groups
    df['group'] = df['family'] + '_' + df['host']
    
    # Print group statistics
    print(f"Found {len(df['group'].unique())} unique family-host groups")
    for group, count in df['group'].value_counts().head(10).items():
        print(f"  Group '{group}': {count} sequences")
    
    if len(df['group'].unique()) > 10:
        print(f"  ... and {len(df['group'].unique()) - 10} more groups")
    
    # Initialize split dictionaries
    splits = {split: [] for split in proportions.keys()}
    
    # Process each group separately to maintain stratification
    for group, group_df in df.groupby('group'):
        group_ids = group_df['seq_id'].tolist()
        group_size = len(group_ids)
        
        # Calculate sizes for each split
        train_size = max(int(group_size * proportions['train']), 1)
        val_size = max(int(group_size * proportions['validate']), 1)
        
        # Make sure we don't exceed group size
        if train_size + val_size > group_size:
            # Adjust sizes
            if group_size == 1:
                # If only one sample, assign to all splits
                train_ids = val_ids = test_ids = group_ids
            elif group_size == 2:
                # If two samples, assign to train and validate, duplicate for test
                train_ids = [group_ids[0]]
                val_ids = [group_ids[1]]
                test_ids = [group_ids[0]]  # Duplicate for test
            else:
                # Otherwise, adjust proportionally
                train_size = max(int(group_size * 0.6), 1)
                val_size = max(int(group_size * 0.2), 1)
                if train_size + val_size >= group_size:
                    train_size = max(group_size - 1, 1)
                    val_size = 1
                
                # Split into train, val, and test
                remaining = list(range(group_size))
                train_indices = remaining[:train_size]
                val_indices = remaining[train_size:train_size+val_size]
                test_indices = remaining[train_size+val_size:]
                
                train_ids = [group_ids[i] for i in train_indices]
                val_ids = [group_ids[i] for i in val_indices]
                test_ids = [group_ids[i] for i in test_indices]
        else:
            # Normal case - enough samples for all splits
            # Shuffle the group with a fixed seed for reproducibility
            np.random.seed(42)
            np.random.shuffle(group_ids)
            
            # Split the group
            train_ids = group_ids[:train_size]
            val_ids = group_ids[train_size:train_size+val_size]
            test_ids = group_ids[train_size+val_size:]
        
        # Add to splits
        splits['train'].extend(train_ids)
        splits['validate'].extend(val_ids)
        splits['test'].extend(test_ids)
    
    # Ensure each split has at least 2 samples (for batch normalization)
    for split_name, split_ids in splits.items():
        if len(split_ids) < 2:
            print(f"Warning: {split_name} split has only {len(split_ids)} sequences")
            if len(split_ids) == 1:
                # Duplicate the single sample
                splits[split_name].append(split_ids[0])
                print(f"  Duplicated the sample to ensure at least 2 sequences")
            elif len(split_ids) == 0:
                # Borrow from train split if available
                if len(splits['train']) >= 2:
                    splits[split_name].extend(splits['train'][:2])
                    print(f"  Borrowed 2 sequences from train split")
    
    # Print split sizes
    for split_name, split_ids in splits.items():
        print(f"{split_name} split: {len(split_ids)} sequences")
    
    return splits


def process_splits(input_dir, output_dir, k_values=None, proportions=None, metadata_file=None):
    """
    Process the dataset into train, validation, and test splits.
    
    Args:
        input_dir (str): Directory containing HDF5 files
        output_dir (str): Directory to write split files
        k_values (list): List of k values to process
        proportions (dict): Dictionary with split proportions
        metadata_file (str): Optional path to metadata file
        
    Returns:
        pd.DataFrame: DataFrame with split metadata
    """
    if k_values is None:
        k_values = [3, 4, 5, 6, 7, 8]
    
    if proportions is None:
        proportions = {'train': 0.8, 'validate': 0.1, 'test': 0.1}
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Find HDF5 files
    h5_files = []
    for k in k_values:
        file_path = os.path.join(input_dir, f'viral_dataset_k{k}.h5')
        if os.path.exists(file_path):
            h5_files.append((k, file_path))
    
    if not h5_files:
        raise FileNotFoundError(f"No HDF5 files found in {input_dir} for k values {k_values}")
    
    # Load the first file for splitting
    k_first, file_first = h5_files[0]
    print(f"Loading data from {file_first} for splitting...")
    features, seq_ids, families, hosts = load_h5_data(file_first)
    
    # Load metadata if provided
    if metadata_file and os.path.exists(metadata_file):
        print(f"Loading metadata from {metadata_file}...")
        try:
            metadata_df = pd.read_csv(metadata_file, sep='\t')
            # Use metadata to update hosts if possible
            if 'accession' in metadata_df.columns and 'host' in metadata_df.columns:
                # Create mapping from accession to host
                acc_to_host = {
                    str(row['accession']).lower(): str(row['host']).lower()
                    for _, row in metadata_df.iterrows()
                }
                # Update hosts
                updated_hosts = []
                for seq_id, host in zip(seq_ids, hosts):
                    seq_id_lower = str(seq_id).lower()
                    if seq_id_lower in acc_to_host:
                        updated_hosts.append(acc_to_host[seq_id_lower])
                    else:
                        updated_hosts.append(host)
                hosts = updated_hosts
                print(f"Updated host information using metadata file")
        except Exception as e:
            print(f"Error loading metadata: {e}")
    
    # Create splits
    print(f"Creating splits for {len(seq_ids)} sequences...")
    splits = stratified_split(seq_ids, families, hosts, proportions)
    
    # Create metadata CSV
    metadata = []
    for split_name, split_ids in splits.items():
        for seq_id in split_ids:
            try:
                idx = seq_ids.index(seq_id)
                metadata.append({
                    'sequence_id': seq_id,
                    'split': split_name,
                    'family': families[idx],
                    'host': hosts[idx]
                })
            except ValueError:
                # If the sequence ID is not found, use available information
                print(f"Warning: Sequence ID {seq_id} not found in original data")
    
    metadata_df = pd.DataFrame(metadata)
    metadata_csv = os.path.join(output_dir, "split_metadata.csv")
    metadata_df.to_csv(metadata_csv, index=False)
    
    # Process each k-mer file
    for k, file_path in h5_files:
        print(f"Processing k={k}...")
        k_dir = Path(output_dir) / f"k{k}"
        k_dir.mkdir(exist_ok=True)
        
        features, file_seq_ids, file_families, file_hosts = load_h5_data(file_path)
        
        # Create sequence ID to index mapping
        seq_id_to_idx = {seq_id: i for i, seq_id in enumerate(file_seq_ids)}
        
        # Process each split
        for split_name, split_ids in splits.items():
            print(f"  Creating {split_name} split...")
            
            # Find indices of sequences in this split
            split_indices = []
            for seq_id in split_ids:
                if seq_id in seq_id_to_idx:
                    split_indices.append(seq_id_to_idx[seq_id])
            
            # Ensure at least 2 samples for batch normalization
            if 0 < len(split_indices) < 2:
                split_indices.append(split_indices[0])  # Duplicate
            
            if not split_indices:
                print(f"  Warning: No sequences for {split_name} split")
                # Use the first few sequences if none match
                if len(file_seq_ids) >= 2:
                    split_indices = [0, 1]
                elif len(file_seq_ids) == 1:
                    split_indices = [0, 0]
                else:
                    continue
            
            # Create HDF5 file
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
    
    # Print summary
    print("\nFinal Split Summary:")
    for split_name, split_ids in splits.items():
        print(f"{split_name}: {len(split_ids)} sequences ({len(split_ids)/len(seq_ids):.2%})")
    
    return metadata_df


def main(args):
    """
    Main function to split the dataset.
    
    Args:
        args: Command-line arguments
    """
    # Parse k-mer sizes
    k_values = [int(k) for k in args.k_values.split(',')]
    
    # Parse proportions
    proportions = {
        'train': args.train_ratio,
        'validate': args.val_ratio,
        'test': args.test_ratio
    }
    
    # Normalize proportions
    total = sum(proportions.values())
    proportions = {k: v / total for k, v in proportions.items()}
    
    print(f"Splitting dataset for k = {k_values}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Split ratios: {proportions}")
    
    # Process splits
    metadata_df = process_splits(
        args.input_dir,
        args.output_dir,
        k_values=k_values,
        proportions=proportions,
        metadata_file=args.metadata
    )
    
    print(f"\nSplit metadata saved to {os.path.join(args.output_dir, 'split_metadata.csv')}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Split dataset for virus host classification")
    parser.add_argument("--input_dir", type=str, required=True,
                      help="Directory containing HDF5 feature files")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory to save split files")
    parser.add_argument("--k_values", type=str, default="3,4,5,6,7,8",
                      help="Comma-separated list of k-mer sizes")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                      help="Ratio of training data")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                      help="Ratio of validation data")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                      help="Ratio of test data")
    parser.add_argument("--metadata", type=str, 
                      help="Path to processed metadata file (optional)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)
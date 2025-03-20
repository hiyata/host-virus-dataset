"""
Dataset module for PyTorch.

This module provides custom PyTorch datasets for virus classification.
"""

import torch
from torch.utils.data import Dataset


class VirusDataset(Dataset):
    """Custom Dataset for virus sequences."""
    def __init__(self, features, labels):
        """
        Initialize the dataset.
        
        Args:
            features (numpy.ndarray): Array of features
            labels (numpy.ndarray): Array of labels
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.labels)
    
    def __getitem__(self, idx):
        """Return a sample from the dataset."""
        return self.features[idx], self.labels[idx]


class VirusDatasetWithIDs(Dataset):
    """Custom Dataset for virus sequences with sequence IDs."""
    def __init__(self, features, labels, seq_ids):
        """
        Initialize the dataset.
        
        Args:
            features (numpy.ndarray): Array of features
            labels (numpy.ndarray): Array of labels
            seq_ids (list): List of sequence IDs
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.seq_ids = seq_ids
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.labels)
    
    def __getitem__(self, idx):
        """Return a sample from the dataset."""
        return self.features[idx], self.labels[idx], self.seq_ids[idx]
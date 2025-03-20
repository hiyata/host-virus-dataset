"""
Neural network model for virus host classification.

This module provides the VirusClassifier model for classifying virus hosts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VirusClassifier(nn.Module):
    """Neural network for virus host classification."""
    def __init__(self, input_shape: int, dropout_rate: float = 0.3):
        """
        Initialize the model.
        
        Args:
            input_shape (int): Number of input features
            dropout_rate (float): Dropout rate
        """
        super(VirusClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 32),
            nn.GELU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout_rate),
            
            nn.Linear(32, 32),
            nn.GELU(),
            
            nn.Linear(32, 2)
        )
    
    def forward(self, x):
        """Forward pass."""
        return self.network(x)
    
    def predict_proba(self, x):
        """
        Predict class probabilities.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Class probabilities
        """
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)
    
    def get_embeddings(self, x):
        """
        Get embeddings from the penultimate layer.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Embeddings
        """
        with torch.no_grad():
            # Extract all but the last layer
            penultimate = nn.Sequential(*list(self.network.children())[:-1])
            return penultimate(x)


class EnhancedVirusClassifier(nn.Module):
    """Enhanced neural network for virus host classification with multiple heads."""
    def __init__(self, input_shape: int, num_classes: int = 2, dropout_rate: float = 0.3):
        """
        Initialize the model.
        
        Args:
            input_shape (int): Number of input features
            num_classes (int): Number of output classes
            dropout_rate (float): Dropout rate
        """
        super(EnhancedVirusClassifier, self).__init__()
        
        # Feature extraction backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 32),
            nn.GELU(),
            nn.BatchNorm1d(32),
        )
        
        # Classification head
        self.classifier = nn.Linear(32, num_classes)
        
    def forward(self, x):
        """Forward pass."""
        features = self.backbone(x)
        return self.classifier(features)
    
    def get_embeddings(self, x):
        """Get embeddings from the backbone."""
        return self.backbone(x)
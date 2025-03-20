"""
Classical machine learning models for virus host classification.

This module provides wrapper classes for scikit-learn models for consistency
with the PyTorch models.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib


class ClassicalModel:
    """Base class for classical ML models."""
    def __init__(self, model=None):
        """Initialize with a scikit-learn model."""
        self.model = model
    
    def fit(self, X, y):
        """
        Fit the model.
        
        Args:
            X (numpy.ndarray): Features
            y (numpy.ndarray): Labels
            
        Returns:
            self: Fitted model
        """
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X (numpy.ndarray): Features
            
        Returns:
            numpy.ndarray: Predicted labels
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X (numpy.ndarray): Features
            
        Returns:
            numpy.ndarray: Class probabilities
        """
        return self.model.predict_proba(X)
    
    def save(self, path):
        """
        Save the model.
        
        Args:
            path (str): Path to save the model
        """
        joblib.dump(self.model, path)
    
    @classmethod
    def load(cls, path):
        """
        Load a saved model.
        
        Args:
            path (str): Path to the saved model
            
        Returns:
            ClassicalModel: Loaded model
        """
        model = joblib.load(path)
        return cls(model)


class LogisticRegressionModel(ClassicalModel):
    """Logistic regression model for virus host classification."""
    def __init__(self, max_iter=1000, random_state=12041997, **kwargs):
        """
        Initialize the model.
        
        Args:
            max_iter (int): Maximum number of iterations
            random_state (int): Random state for reproducibility
            **kwargs: Additional arguments for LogisticRegression
        """
        super().__init__(LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            **kwargs
        ))


class RandomForestModel(ClassicalModel):
    """Random forest model for virus host classification."""
    def __init__(self, n_estimators=100, random_state=12041997, n_jobs=-1, **kwargs):
        """
        Initialize the model.
        
        Args:
            n_estimators (int): Number of trees
            random_state (int): Random state for reproducibility
            n_jobs (int): Number of jobs to run in parallel
            **kwargs: Additional arguments for RandomForestClassifier
        """
        super().__init__(RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs
        ))
    
    def feature_importances(self):
        """
        Get feature importances.
        
        Returns:
            numpy.ndarray: Feature importances
        """
        return self.model.feature_importances_
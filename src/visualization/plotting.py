"""
Plotting utilities module.

This module provides common plotting functions for visualizing
model performance and results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve


def set_plotting_style():
    """Set a consistent style for all plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10


def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix", 
                          output_path=None, normalize=False, cmap='Blues'):
    """
    Plot a confusion matrix.
    
    Args:
        y_true (numpy.ndarray): True labels
        y_pred (numpy.ndarray): Predicted labels
        labels (list): Class labels
        title (str): Plot title
        output_path (str): Path to save the plot; if None, the plot is displayed
        normalize (bool): Whether to normalize the confusion matrix
        cmap (str): Colormap name
    """
    set_plotting_style()
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap)
    
    plt.title(title, pad=15)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if labels:
        plt.xticks(np.arange(len(labels)) + 0.5, labels)
        plt.yticks(np.arange(len(labels)) + 0.5, labels)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_roc_curve(y_true, y_proba, output_path=None, title="ROC Curve"):
    """
    Plot a ROC curve.
    
    Args:
        y_true (numpy.ndarray): True labels
        y_proba (numpy.ndarray): Predicted probabilities
        output_path (str): Path to save the plot; if None, the plot is displayed
        title (str): Plot title
    """
    set_plotting_style()
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title, pad=15)
    plt.legend(loc="lower right")
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_precision_recall_curve(y_true, y_proba, output_path=None, title="Precision-Recall Curve"):
    """
    Plot a precision-recall curve.
    
    Args:
        y_true (numpy.ndarray): True labels
        y_proba (numpy.ndarray): Predicted probabilities
        output_path (str): Path to save the plot; if None, the plot is displayed
        title (str): Plot title
    """
    set_plotting_style()
    
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.axhline(y=np.sum(y_true) / len(y_true), color='navy', linestyle='--', 
                label=f'Baseline (= {np.sum(y_true) / len(y_true):.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title, pad=15)
    plt.legend(loc="lower left")
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_history(train_losses, val_losses, train_metrics=None, val_metrics=None,
                         metric_name='Accuracy', output_path=None):
    """
    Plot training history.
    
    Args:
        train_losses (list): Training losses
        val_losses (list): Validation losses
        train_metrics (list): Training metrics
        val_metrics (list): Validation metrics
        metric_name (str): Name of the metric
        output_path (str): Path to save the plot; if None, the plot is displayed
    """
    set_plotting_style()
    
    fig, axes = plt.subplots(1, 2 if train_metrics is not None else 1, figsize=(12, 5))
    
    # Plot losses
    ax = axes[0] if train_metrics is not None else axes
    ax.plot(train_losses, label='Train', linewidth=2)
    ax.plot(val_losses, label='Validation', linewidth=2)
    ax.set_title('Training and Validation Loss', pad=15)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    # Plot metrics if provided
    if train_metrics is not None and val_metrics is not None:
        axes[1].plot(train_metrics, label='Train', linewidth=2)
        axes[1].plot(val_metrics, label='Validation', linewidth=2)
        axes[1].set_title(f'Training and Validation {metric_name}', pad=15)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel(metric_name)
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_feature_importances(importances, feature_names=None, n_top=20, output_path=None,
                           title="Feature Importances"):
    """
    Plot feature importances.
    
    Args:
        importances (numpy.ndarray): Feature importances
        feature_names (list): Feature names
        n_top (int): Number of top features to show
        output_path (str): Path to save the plot; if None, the plot is displayed
        title (str): Plot title
    """
    set_plotting_style()
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    indices = indices[:n_top]
    
    # Create names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importances))]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.title(title, pad=15)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_model_comparison(results_dict, metric='accuracy', output_path=None):
    """
    Plot model comparison bar chart.
    
    Args:
        results_dict (dict): Dictionary with model names as keys and metric values as values
        metric (str): Metric name
        output_path (str): Path to save the plot; if None, the plot is displayed
    """
    set_plotting_style()
    
    models = list(results_dict.keys())
    values = [results_dict[model][metric] if isinstance(results_dict[model], dict) 
              else results_dict[model] for model in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, values, color='skyblue')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.ylim(0, max(values) * 1.15)  # Add some space at the top
    plt.ylabel(metric.capitalize())
    plt.title(f'Model Comparison - {metric.capitalize()}', pad=15)
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
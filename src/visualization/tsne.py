"""
t-SNE visualization module.

This module provides functions to generate t-SNE visualizations of virus features
and model embeddings.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import joblib
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D

from src.models.neural_network import VirusClassifier


def load_h5_for_tsne(h5_path):
    """
    Load data from HDF5 file for t-SNE visualization.
    
    Args:
        h5_path (str): Path to the HDF5 file
        
    Returns:
        tuple: Tuple containing (features, labels, families, sequence IDs)
    """
    import h5py
    
    with h5py.File(h5_path, 'r') as f:
        X_all = f['features'][:]
        hosts_all = np.array([h.decode() for h in f['hosts']])
        seq_ids_all = np.array([s.decode() for s in f['sequence_ids']])
        
        families_all = None
        if 'families' in f:
            families_all = np.array([fam.decode() for fam in f['families']])
        
    valid_mask = (hosts_all == 'human') | (hosts_all == 'non-human')
    X = X_all[valid_mask]
    hosts = hosts_all[valid_mask]
    seq_ids = seq_ids_all[valid_mask]
    if families_all is not None:
        families = families_all[valid_mask]
    else:
        families = np.array(["NA"]*len(hosts), dtype=object)
    
    y = np.array([1 if h == 'human' else 0 for h in hosts], dtype=np.int64)
    return X, y, families, seq_ids


def get_embeddings_and_probabilities(model, X_tensor, device='cpu'):
    """
    Get embeddings, probabilities, and predictions from a model.
    
    Args:
        model (VirusClassifier): Model to use
        X_tensor (torch.Tensor): Input tensor
        device (str): Device to use
        
    Returns:
        tuple: Tuple containing (embeddings, probabilities, predictions)
    """
    import torch.nn.functional as F
    
    # Define a sub-network that excludes the final linear layer
    penultimate = torch.nn.Sequential(*list(model.network.children())[:-1])
    
    model.eval()
    with torch.no_grad():
        # Get penultimate-layer embeddings
        feats = penultimate(X_tensor.to(device))
        # Get final layer outputs
        logits = model(X_tensor.to(device))
        probs = F.softmax(logits, dim=1)[:,1]  # probability of 'human' class
        preds = logits.argmax(dim=1)
    
    return feats.cpu().numpy(), probs.cpu().numpy(), preds.cpu().numpy()


def scatter_discrete(ax, points2d, labels, title, legend_title, cmap='tab10'):
    """
    Create a scatter plot with discrete colors.
    
    Args:
        ax (matplotlib.axes.Axes): Axes to plot on
        points2d (numpy.ndarray): 2D points to plot
        labels (numpy.ndarray): Labels for coloring
        title (str): Plot title
        legend_title (str): Legend title
        cmap (str): Colormap name
    """
    scatter = ax.scatter(points2d[:,0], points2d[:,1], c=labels, cmap=cmap, s=15)
    ax.set_title(title, pad=15)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    
    if legend_title == "Host":
        custom_legend = [
            Line2D([0], [0], marker='o', color='w', label='Non-Human',
                   markerfacecolor=scatter.cmap(0), markersize=6),
            Line2D([0], [0], marker='o', color='w', label='Human',
                   markerfacecolor=scatter.cmap(1), markersize=6),
        ]
        ax.legend(handles=custom_legend, title=legend_title, loc='best')
    
    ax.grid(True)


def scatter_continuous(ax, points2d, values, title, cbar_label):
    """
    Create a scatter plot with continuous colors.
    
    Args:
        ax (matplotlib.axes.Axes): Axes to plot on
        points2d (numpy.ndarray): 2D points to plot
        values (numpy.ndarray): Values for coloring
        title (str): Plot title
        cbar_label (str): Colorbar label
    """
    sc = ax.scatter(points2d[:,0], points2d[:,1], c=values, cmap='viridis', s=15)
    ax.set_title(title, pad=15)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label)
    ax.grid(True)


def tsne_visualization(
    h5_path, model_path, scaler_path, output_dir=None,
    sample_size=2000, perplexity=30, device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Generate t-SNE visualizations for raw features and model embeddings.
    
    Args:
        h5_path (str): Path to the HDF5 file
        model_path (str): Path to the model file
        scaler_path (str): Path to the scaler file
        output_dir (str): Directory to save plots; if None, plots are shown
        sample_size (int): Maximum number of samples to use
        perplexity (float): Perplexity parameter for t-SNE
        device (str): Device to use
        
    Returns:
        dict: Dictionary with t-SNE results
    """
    print(f"Using device: {device}")
    
    # Load the data
    X, y, families, seq_ids = load_h5_for_tsne(h5_path)
    print("Loaded data shape:", X.shape)
    
    # Scale
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)
    
    # Load model
    input_shape = X.shape[1]
    model = VirusClassifier(input_shape=input_shape).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Compute embeddings + probabilities for entire set
    X_torch = torch.FloatTensor(X_scaled)
    embeddings, probs, preds = get_embeddings_and_probabilities(model, X_torch, device=device)
    
    # Possibly subsample if data is huge (for plotting)
    N = X.shape[0]
    if N > sample_size:
        idx = np.random.choice(N, size=sample_size, replace=False)
        X_sub = X_scaled[idx]
        embeddings_sub = embeddings[idx]
        probs_sub = probs[idx]
        y_sub = y[idx]
        families_sub = families[idx]
    else:
        X_sub = X_scaled
        embeddings_sub = embeddings
        probs_sub = probs
        y_sub = y
        families_sub = families
    
    # t-SNE on raw input
    print("Running t-SNE on raw features...")
    tsne_raw = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    raw_2d = tsne_raw.fit_transform(X_sub)
    
    # t-SNE on embeddings
    print("Running t-SNE on embeddings...")
    tsne_embed = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embed_2d = tsne_embed.fit_transform(embeddings_sub)
    
    # Create plots
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
    axes = axes.ravel()
    
    # 1) RAW features colored by predicted probabilities
    scatter_continuous(
        ax=axes[0],
        points2d=raw_2d,
        values=probs_sub,
        title="Raw (colored by Probabilities)",
        cbar_label="P(Human)"
    )
    
    # 2) RAW features colored by families (discrete)
    unique_fams = list(set(families_sub))
    fam_to_idx = {f:i for i, f in enumerate(unique_fams)}
    family_ints = np.array([fam_to_idx[f] for f in families_sub])
    scatter_discrete(
        ax=axes[1],
        points2d=raw_2d,
        labels=family_ints,
        title="Raw (colored by Families)",
        legend_title="Families"
    )
    
    # 3) RAW features colored by true host
    scatter_discrete(
        ax=axes[2],
        points2d=raw_2d,
        labels=y_sub,
        title="Raw (colored by True Host)",
        legend_title="Host"
    )
    
    # 4) EMBEDDINGS colored by predicted probabilities
    scatter_continuous(
        ax=axes[3],
        points2d=embed_2d,
        values=probs_sub,
        title="Embeddings (colored by Probabilities)",
        cbar_label="P(Human)"
    )
    
    # 5) EMBEDDINGS colored by families
    scatter_discrete(
        ax=axes[4],
        points2d=embed_2d,
        labels=family_ints,
        title="Embeddings (colored by Families)",
        legend_title="Families"
    )
    
    # 6) EMBEDDINGS colored by true host
    scatter_discrete(
        ax=axes[5],
        points2d=embed_2d,
        labels=y_sub,
        title="Embeddings (colored by True Host)",
        legend_title="Host"
    )
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'tsne_visualization.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return {
        'raw_2d': raw_2d,
        'embed_2d': embed_2d,
        'families': families_sub,
        'y': y_sub,
        'probs': probs_sub
    }


if __name__ == "__main__":
    # Example usage
    h5_path = "dataset_splits/k4/train.h5"
    model_path = "results/k4/nn/model.pt"
    scaler_path = "results/k4/nn/scaler.pkl"
    tsne_visualization(h5_path, model_path, scaler_path, "results/visualizations")
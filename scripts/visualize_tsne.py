#!/usr/bin/env python3
"""
t-SNE visualization script.

This script generates t-SNE visualizations for raw features and model embeddings.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.visualization.tsne import tsne_visualization


def main(args):
    """
    Main function to generate t-SNE visualizations.
    
    Args:
        args: Command-line arguments
    """
    # Get file paths
    h5_path = args.h5_path
    model_path = args.model_path
    scaler_path = args.scaler_path
    
    print(f"Generating t-SNE visualizations")
    print(f"HDF5 file: {h5_path}")
    print(f"Model file: {model_path}")
    print(f"Scaler file: {scaler_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sample size: {args.sample_size}")
    print(f"Perplexity: {args.perplexity}")
    
    # Run t-SNE visualization
    results = tsne_visualization(
        h5_path=h5_path,
        model_path=model_path,
        scaler_path=scaler_path,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        perplexity=args.perplexity
    )
    
    # Save t-SNE coordinates to CSV
    if args.save_coords:
        raw_df = pd.DataFrame(results['raw_2d'], columns=['tsne_1', 'tsne_2'])
        raw_df['family'] = results['families']
        raw_df['host'] = results['y']
        raw_df['probability'] = results['probs']
        raw_df.to_csv(os.path.join(args.output_dir, 'raw_tsne_coordinates.csv'), index=False)
        
        embed_df = pd.DataFrame(results['embed_2d'], columns=['tsne_1', 'tsne_2'])
        embed_df['family'] = results['families']
        embed_df['host'] = results['y']
        embed_df['probability'] = results['probs']
        embed_df.to_csv(os.path.join(args.output_dir, 'embed_tsne_coordinates.csv'), index=False)
        
        print(f"t-SNE coordinates saved to {args.output_dir}")
    
    print(f"\nVisualization complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate t-SNE visualizations for virus features")
    parser.add_argument("--h5_path", type=str, required=True,
                      help="Path to HDF5 file")
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to model file")
    parser.add_argument("--scaler_path", type=str, required=True,
                      help="Path to scaler file")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory to save visualizations")
    parser.add_argument("--sample_size", type=int, default=2000,
                      help="Maximum number of samples to use")
    parser.add_argument("--perplexity", type=float, default=30,
                      help="Perplexity parameter for t-SNE")
    parser.add_argument("--save_coords", action="store_true",
                      help="Save t-SNE coordinates to CSV")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)
#!/usr/bin/env python3
"""
Main script for training virus host classification models.

This script trains neural network, logistic regression, and random forest models
on the dataset for each k-mer size.
"""

import os
import argparse
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    matthews_corrcoef
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Set a specific style for figures
plt.style.use('seaborn-v0_8-whitegrid')

# -------------------
# Dataset & Model
# -------------------

class VirusDataset(Dataset):
    """Custom Dataset for virus sequences."""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class VirusClassifier(nn.Module):
    """Neural network for virus host classification."""
    def __init__(self, input_shape: int):
        super(VirusClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.GELU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),

            nn.Linear(32, 32),
            nn.GELU(),

            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.network(x)

# ------------------------
# Data Loading Utilities
# ------------------------

def load_h5_data(h5_path: str):
    """
    Loads the .h5 file, filters to 'human' / 'non-human',
    and returns (X, y, seq_ids).

    - y = 1 for human, 0 for non-human
    - seq_ids are the IDs for the sequences, used for tracing predictions in CSV
    """
    try:
        with h5py.File(h5_path, 'r') as h5f:
            # Make sure all needed datasets exist
            for needed_key in ['features', 'hosts', 'sequence_ids']:
                if needed_key not in h5f:
                    raise KeyError(f"Required dataset '{needed_key}' not found in HDF5 file")

            X_all = h5f['features'][:]
            hosts_all = np.array([h.decode() for h in h5f['hosts']])
            seq_ids_all = np.array([s.decode() for s in h5f['sequence_ids']])

            # Filter only human/non-human
            valid_mask = (hosts_all == 'human') | (hosts_all == 'non-human')
            X_filtered = X_all[valid_mask]
            hosts_filtered = hosts_all[valid_mask]
            seq_ids_filtered = seq_ids_all[valid_mask]

            y_binary = np.array([1 if h == 'human' else 0 for h in hosts_filtered], dtype=np.int64)

            return X_filtered, y_binary, seq_ids_filtered

    except (OSError, KeyError) as e:
        print(f"Error loading data from {h5_path}: {str(e)}")
        raise

# ------------------------
# Neural Network Training
# ------------------------

def train_and_evaluate_neural_network(
    X_train, y_train, X_val, y_val, X_test, y_test,
    seq_ids_test, output_directory,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Trains and evaluates the neural network, saves model.pt, scaler.pkl,
    and outputs confusion matrix, training curves, and predictions CSV.
    """
    # -----------------------
    # Scale Data
    # -----------------------
    print("[NN] Scaling data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    # -----------------------
    # Create Datasets/Loaders
    # -----------------------
    train_dataset = VirusDataset(X_train_scaled, y_train)
    val_dataset   = VirusDataset(X_val_scaled,   y_val)
    test_dataset  = VirusDataset(X_test_scaled,  y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=64)
    test_loader  = DataLoader(test_dataset,  batch_size=64)

    # -----------------------
    # Initialize Model
    # -----------------------
    model = VirusClassifier(X_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

    # Training parameters
    epochs = 60
    best_val_acc = 0
    best_model_state = None
    patience = 3
    patience_counter = 0

    # Tracking
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    print("[NN] === Training Neural Network ===")
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"[NN] Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[NN] Early stopping at epoch {epoch+1}")
                break

    # -----------------------
    # Load best model
    # -----------------------
    model.load_state_dict(best_model_state)

    # -----------------------
    # Final evaluation on test
    # -----------------------
    model.eval()
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = outputs.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    test_predictions = np.array(all_predictions)
    test_true_labels = np.array(all_true_labels)

    # -----------------------
    # Metrics
    # -----------------------
    accuracy    = accuracy_score(test_true_labels, test_predictions)
    f1_macro    = f1_score(test_true_labels, test_predictions, average='macro')
    f1_weighted = f1_score(test_true_labels, test_predictions, average='weighted')
    mcc         = matthews_corrcoef(test_true_labels, test_predictions)

    print("\n[NN] Test Set Metrics:")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"F1 (macro):  {f1_macro:.4f}")
    print(f"F1 (weighted): {f1_weighted:.4f}")
    print(f"MCC:         {mcc:.4f}")

    print("\n[NN] Detailed Classification Report:")
    print(classification_report(test_true_labels, test_predictions, digits=3))

    # -----------------------
    # Save training plots
    # -----------------------
    plt.figure(figsize=(12, 5))
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train', linewidth=2)
    plt.plot(val_losses,   label='Validation', linewidth=2)
    plt.title('Training and Validation Loss', pad=15)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train', linewidth=2)
    plt.plot(val_accs,   label='Validation', linewidth=2)
    plt.title('Training and Validation Accuracy', pad=15)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'nn_training_history.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # -----------------------
    # Confusion matrix
    # -----------------------
    cm = confusion_matrix(test_true_labels, test_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('NN Test Set Confusion Matrix', pad=15)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(output_directory, 'nn_confusion_matrix.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # -----------------------
    # Save predictions to CSV
    # -----------------------
    df_preds = pd.DataFrame({
        'seq_id': seq_ids_test,
        'true_label': test_true_labels,
        'predicted_label': test_predictions
    })
    csv_path = os.path.join(output_directory, 'nn_predictions.csv')
    df_preds.to_csv(csv_path, index=False)

    # -----------------------
    # Save model & scaler
    # -----------------------
    torch.save(model.state_dict(), os.path.join(output_directory, 'model.pt'))
    joblib.dump(scaler, os.path.join(output_directory, 'scaler.pkl'))

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'mcc': mcc
    }

# ---------------------------------
# Logistic Regression
# ---------------------------------

def train_and_evaluate_logistic_regression(
    X_train, y_train, X_val, y_val, X_test, y_test,
    seq_ids_test, output_directory
):
    """
    Trains and evaluates a Logistic Regression model, saves logreg_model.pkl, scaler.pkl,
    and outputs confusion matrix, predictions CSV.
    """
    print("[LOGREG] Scaling data and training Logistic Regression...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    # Initialize and fit
    logreg = LogisticRegression(max_iter=1000, random_state=12041997)
    logreg.fit(X_train_scaled, y_train)

    # Evaluate on test
    test_predictions = logreg.predict(X_test_scaled)
    accuracy    = accuracy_score(y_test, test_predictions)
    f1_macro    = f1_score(y_test, test_predictions, average='macro')
    f1_weighted = f1_score(y_test, test_predictions, average='weighted')
    mcc         = matthews_corrcoef(y_test, test_predictions)

    print("\n[LOGREG] Test Set Metrics:")
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"F1 (macro):   {f1_macro:.4f}")
    print(f"F1 (weighted): {f1_weighted:.4f}")
    print(f"MCC:          {mcc:.4f}")

    print("\n[LOGREG] Detailed Classification Report:")
    print(classification_report(y_test, test_predictions, digits=3))

    # Confusion Matrix
    cm = confusion_matrix(y_test, test_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Logistic Regression Confusion Matrix', pad=15)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(output_directory, 'logreg_confusion_matrix.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Save predictions to CSV
    df_preds = pd.DataFrame({
        'seq_id': seq_ids_test,
        'true_label': y_test,
        'predicted_label': test_predictions
    })
    csv_path = os.path.join(output_directory, 'logreg_predictions.csv')
    df_preds.to_csv(csv_path, index=False)

    # Save model and scaler
    joblib.dump(logreg, os.path.join(output_directory, 'logreg_model.pkl'))
    joblib.dump(scaler, os.path.join(output_directory, 'scaler.pkl'))

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'mcc': mcc
    }

# ---------------------------------
# Random Forest
# ---------------------------------

def train_and_evaluate_random_forest(
    X_train, y_train, X_val, y_val, X_test, y_test,
    seq_ids_test, output_directory
):
    """
    Trains and evaluates a Random Forest model, saves rf_model.pkl, scaler.pkl,
    and outputs confusion matrix, predictions CSV.
    """
    print("[RF] Scaling data and training Random Forest...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    # Initialize and fit
    rf = RandomForestClassifier(
        n_estimators=100, random_state=12041997,
        max_depth=None, n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)

    # Evaluate on test
    test_predictions = rf.predict(X_test_scaled)
    accuracy    = accuracy_score(y_test, test_predictions)
    f1_macro    = f1_score(y_test, test_predictions, average='macro')
    f1_weighted = f1_score(y_test, test_predictions, average='weighted')
    mcc         = matthews_corrcoef(y_test, test_predictions)

    print("\n[RF] Test Set Metrics:")
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"F1 (macro):   {f1_macro:.4f}")
    print(f"F1 (weighted): {f1_weighted:.4f}")
    print(f"MCC:          {mcc:.4f}")

    print("\n[RF] Detailed Classification Report:")
    print(classification_report(y_test, test_predictions, digits=3))

    # Confusion Matrix
    cm = confusion_matrix(y_test, test_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Random Forest Confusion Matrix', pad=15)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(output_directory, 'rf_confusion_matrix.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Save predictions to CSV
    df_preds = pd.DataFrame({
        'seq_id': seq_ids_test,
        'true_label': y_test,
        'predicted_label': test_predictions
    })
    csv_path = os.path.join(output_directory, 'rf_predictions.csv')
    df_preds.to_csv(csv_path, index=False)

    # Save model and scaler
    joblib.dump(rf, os.path.join(output_directory, 'rf_model.pkl'))
    joblib.dump(scaler, os.path.join(output_directory, 'scaler.pkl'))

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'mcc': mcc
    }

# -------------
# Main Script
# -------------

def main():
    """
    Main script to run ablation for k=3,4,5,6 on:
        - Neural Network (PyTorch)
        - Logistic Regression
        - Random Forest

    Each model is saved in a subdirectory, and per-sequence predictions are
    also saved in a CSV for easy inspection.
    """
    parser = argparse.ArgumentParser(description="Train virus host classification models")
    parser.add_argument("--data_dir", type=str, required=True,
                      help="Directory containing dataset splits")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory to save results")
    parser.add_argument("--kmers", type=int, nargs='+', default=[3, 4, 5, 6, 7, 8],
                      help="K-mer sizes to process")
    parser.add_argument("--model", type=str, default="all",
                      choices=["all", "nn", "logreg", "rf"],
                      help="Model type to train")
    parser.add_argument("--epochs", type=int, default=60,
                      help="Number of epochs for neural network training")
    parser.add_argument("--n_estimators", type=int, default=100,
                      help="Number of trees for random forest")
    
    args = parser.parse_args()
    
    # Create the main results dir if not exists
    os.makedirs(args.output_dir, exist_ok=True)

    for k in args.kmers:
        print(f"\n\n========== Processing k={k} ==========")

        # Paths to HDF5 splits
        kmer_dir    = os.path.join(args.data_dir, f"k{k}")
        train_path  = os.path.join(kmer_dir, "train.h5")
        val_path    = os.path.join(kmer_dir, "validate.h5")
        test_path   = os.path.join(kmer_dir, "test.h5")

        # Load data
        print(f"[k={k}] Loading data splits...")
        X_train, y_train, _train_ids = load_h5_data(train_path)
        X_val,   y_val,   _val_ids   = load_h5_data(val_path)
        X_test,  y_test,  test_ids   = load_h5_data(test_path)

        print(f"[k={k}] Data shapes:")
        print(f"   Train: X={X_train.shape}, y={y_train.shape}")
        print(f"   Val:   X={X_val.shape},   y={y_val.shape}")
        print(f"   Test:  X={X_test.shape},  y={y_test.shape}")

        # Create an output directory for this k
        k_output_dir = os.path.join(args.output_dir, f"k{k}")
        os.makedirs(k_output_dir, exist_ok=True)

        # ==============================
        # 1) Neural Network
        # ==============================
        if args.model == "all" or args.model == "nn":
            nn_output_dir = os.path.join(k_output_dir, "nn")
            os.makedirs(nn_output_dir, exist_ok=True)
            nn_results = train_and_evaluate_neural_network(
                X_train, y_train,
                X_val,   y_val,
                X_test,  y_test,
                seq_ids_test=test_ids,
                output_directory=nn_output_dir
            )

        # ==============================
        # 2) Logistic Regression
        # ==============================
        if args.model == "all" or args.model == "logreg":
            logreg_output_dir = os.path.join(k_output_dir, "logreg")
            os.makedirs(logreg_output_dir, exist_ok=True)
            logreg_results = train_and_evaluate_logistic_regression(
                X_train, y_train,
                X_val,   y_val,
                X_test,  y_test,
                seq_ids_test=test_ids,
                output_directory=logreg_output_dir
            )

        # ==============================
        # 3) Random Forest
        # ==============================
        if args.model == "all" or args.model == "rf":
            rf_output_dir = os.path.join(k_output_dir, "rf")
            os.makedirs(rf_output_dir, exist_ok=True)
            rf_results = train_and_evaluate_random_forest(
                X_train, y_train,
                X_val,   y_val,
                X_test,  y_test,
                seq_ids_test=test_ids,
                output_directory=rf_output_dir
            )

        # Print summary of all
        print(f"\n=== [k={k}] SUMMARY OF RESULTS ===")
        if args.model == "all" or args.model == "nn":
            print("[NN]:", nn_results)
        if args.model == "all" or args.model == "logreg":
            print("[LogReg]:", logreg_results)
        if args.model == "all" or args.model == "rf":
            print("[RF]:", rf_results)
        print("==================================\n")

    print("\nAll done! Check your ablation results at:", args.output_dir)


if __name__ == "__main__":
    main()
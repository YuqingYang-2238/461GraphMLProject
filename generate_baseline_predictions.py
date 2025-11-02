"""
Generate predictions from baseline GCN models for LLM boosting.

This script loads trained baseline models and generates predictions on test sets,
along with the necessary SMILES and label information for LLM-based boosting.
"""

import torch
import numpy as np
import pandas as pd
import os
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from loadMoleculeData import load_datasets
from BaselineGCNModeling.baselineGCN import GCN, device, load_model_with_config, get_logits
from BaselineGCNModeling.model_configs import ConfigManager


def extract_smiles_and_labels_from_subset(subset):
    """
    Extract SMILES and labels from a Subset object.
    
    Args:
        subset: torch_geometric Subset containing dataset indices
        
    Returns:
        smiles_list: list of SMILES strings
        labels_list: list of labels
    """
    base_ds = subset.dataset
    indices = getattr(subset, 'indices', list(range(len(subset))))
    
    smiles_list = [base_ds[i].smiles for i in indices]
    labels_list = [float(base_ds[i].y.item()) for i in indices]
    
    return smiles_list, labels_list


def calculate_auc(labels, predictions):
    """Calculate ROC AUC score."""
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(labels, predictions)
    except:
        return float('nan')


def generate_predictions_for_dataset(dataset, dataset_name, model_path, config_manager, output_dir):
    """
    Generate predictions for a single dataset and save results.
    
    Args:
        dataset: Dictionary containing train_set, val_set, test_set
        dataset_name: Name of the dataset (e.g., 'BBBP', 'BACE', 'HIV')
        model_path: Path to the trained model checkpoint
        config_manager: ConfigManager instance
        output_dir: Directory to save output files
    """
    
    # Get configuration for this dataset
    config = config_manager.get_config(dataset_name)
    
    # Get datasets
    train_set = dataset['train_set']
    val_set = dataset['val_set']
    test_set = dataset['test_set']
    
    # Load model using existing function from baselineGCN.py
    in_channels = train_set.dataset.num_features
    print(f"Loading model from: {model_path}")
    model = load_model_with_config(model_path, config, in_channels)
    
    # Get predictions using existing get_probs function
    test_logits = get_logits(model, dataset, split_name = "test_set")
    test_probs = 1 / (1 + np.exp(-test_logits))  # sigmoid
    
    val_logits = get_logits(model, dataset, split_name = "val_set")
    val_probs = 1 / (1 + np.exp(-val_logits))  # sigmoid
    
    # Extract SMILES and labels
    test_smiles, test_labels = extract_smiles_and_labels_from_subset(test_set)
    train_smiles, train_labels = extract_smiles_and_labels_from_subset(train_set)
    val_smiles, val_labels = extract_smiles_and_labels_from_subset(val_set)
    
    print(f"\nDataset: {dataset_name}")
    print(f"  Train set size: {len(train_set)}")
    print(f"  Val set size: {len(val_set)}")
    print(f"  Test set size: {len(test_set)}")
    print(f"  Val AUC: {calculate_auc(val_labels, val_probs):.4f}")
    print(f"  Test AUC: {calculate_auc(test_labels, test_probs):.4f}")
    
    # Save results as CSV files
    
    # 1. Save test results as CSV
    test_df = pd.DataFrame({
        'SMILES': test_smiles,
        'True_Label': test_labels,
        'GNN_Prediction': test_probs
    })
    test_csv_path = os.path.join(output_dir, f'{dataset_name}_test_predictions.csv')
    test_df.to_csv(test_csv_path, index=False)
    print(f"  Saved: {test_csv_path}")
    
    # 2. Save train data as CSV
    train_df = pd.DataFrame({
        'SMILES': train_smiles,
        'Label': train_labels
    })
    train_csv_path = os.path.join(output_dir, f'{dataset_name}_train_data.csv')
    train_df.to_csv(train_csv_path, index=False)
    print(f"  Saved: {train_csv_path}")
    
    # 3. Save validation results as CSV
    val_df = pd.DataFrame({
        'SMILES': val_smiles,
        'True_Label': val_labels,
        'GNN_Prediction': val_probs
    })
    val_csv_path = os.path.join(output_dir, f'{dataset_name}_val_predictions.csv')
    val_df.to_csv(val_csv_path, index=False)
    print(f"  Saved: {val_csv_path}")
    
    return {
        'dataset_name': dataset_name,
        'val_auc': calculate_auc(val_labels, val_probs),
        'test_auc': calculate_auc(test_labels, test_probs),
        'val_size': len(val_set),
        'test_size': len(test_set),
        'train_size': len(train_set)
    }


def main():
    """Main function to generate predictions for all datasets."""
    
    # Setup
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baseline_predictions')
    os.makedirs(output_dir, exist_ok=True)
    
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    
    # Initialize configuration manager
    config_manager = ConfigManager()
    config_manager.initialize_default_configs()
    
    # Load all datasets
    datasets = load_datasets(names=["BBBP", "BACE", "HIV"], root="data/moleculenet", scaffold="standard")
    
    # Process each dataset
    all_results = {}
    for dataset in datasets:
        dataset_name = dataset['name']
        model_path = os.path.join(models_dir, f'baselineGCN_{dataset_name}.pt')
        
        if not os.path.exists(model_path):
            print(f"\nWARNING: Model file not found: {model_path}")
            print(f"Skipping {dataset_name}")
            continue
        
        results = generate_predictions_for_dataset(
            dataset, 
            dataset_name, 
            model_path, 
            config_manager,
            output_dir
        )
        all_results[dataset_name] = results
    
    print(f"\n{'='*60}")
    print("All predictions generated successfully!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")
    print("\nFiles created for each dataset:")
    print("  - <DATASET>_val_predictions.csv: Validation predictions and labels")
    print("  - <DATASET>_test_predictions.csv: Test predictions and labels")
    print("  - <DATASET>_train_data.csv: Training SMILES and labels")
    print("\nUse these CSV files in llm_boosting_example.ipynb for LLM boosting!")
    print("Tip: Use validation set for modeling decisions and test set only for final evaluation.")


if __name__ == "__main__":
    main()


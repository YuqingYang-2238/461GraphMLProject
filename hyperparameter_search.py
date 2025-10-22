"""
Hyperparameter grid search implementation for GNN models.
Supports grid search with comprehensive evaluation and result saving.
"""

import itertools
import json
import time
import os
from copy import deepcopy
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

from baselineGCN import GCN, do_epoch
from loadMoleculeData import load_datasets
from model_configs import ModelConfig, ConfigManager


class HyperparameterSearch:
    """Grid search hyperparameter optimization for GNN models."""
    
    def __init__(self, device=None, results_dir="hyperparameter_results"):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(f"{results_dir}/models", exist_ok=True)
        
    def train_single_config(self, config: ModelConfig, dataset_name: str, 
                          max_epochs: int = 100, patience: int = 15) -> Dict[str, Any]:
        """Train a single model configuration and return results."""
        
        # Load dataset
        datasets = load_datasets([dataset_name])
        if not datasets:
            raise ValueError(f"Dataset {dataset_name} not found")
        
        dataset = datasets[0]
        train_loader = DataLoader(dataset['train_set'], batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(dataset['val_set'], batch_size=config.batch_size, shuffle=False)
        test_loader = DataLoader(dataset['test_set'], batch_size=config.batch_size, shuffle=False)
        
        # Get original dataset for num_features
        original_dataset = dataset['train_set'].dataset
        model = GCN(in_channels=original_dataset.num_features, config=config)
        model = model.to(self.device)
        
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )
        criterion = nn.BCEWithLogitsLoss()
        
        best_val_auc = -1.0
        best_state = None
        best_epoch = -1
        epochs_without_improvement = 0
        
        # Track training history
        history = {
            'epochs': [],
            'train_losses': [],
            'val_losses': [],
            'train_aucs': [],
            'val_aucs': [],
            'test_aucs': []
        }
        
        for epoch in range(1, max_epochs + 1):
            train_loss, train_auc = do_epoch(train_loader, training=True, model=model, 
                                           optimizer=optimizer, criterion=criterion)
            val_loss, val_auc = do_epoch(val_loader, training=False, model=model, 
                                       criterion=criterion)
            test_loss, test_auc = do_epoch(test_loader, training=False, model=model, 
                                         criterion=criterion)
            
            # Store history
            history['epochs'].append(epoch)
            history['train_losses'].append(float(train_loss))
            history['val_losses'].append(float(val_loss))
            history['train_aucs'].append(float(train_auc))
            history['val_aucs'].append(float(val_auc))
            history['test_aucs'].append(float(test_auc))
            
            # Check for improvement
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = deepcopy(model.state_dict())
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # Early stopping
            if epochs_without_improvement >= patience:
                break
        
        # Load best state and compute final test AUC
        if best_state is not None:
            model.load_state_dict(best_state)
        _, final_test_auc = do_epoch(test_loader, training=False, model=model, 
                                   criterion=criterion)
        
        return {
            'config': config.to_dict(),
            'best_val_auc': float(best_val_auc),
            'best_epoch': int(best_epoch),
            'final_test_auc': float(final_test_auc),
            'total_epochs': len(history['epochs']),
            'history': history,
            'state_dict': best_state
        }
    
    def grid_search(self, param_grid: Dict[str, List], dataset_name: str, 
                   max_epochs: int = 100, patience: int = 15) -> Dict[str, Any]:
        """Perform grid search over parameter combinations."""
        
        print(f"Starting grid search for {dataset_name}")
        print(f"Parameter grid: {param_grid}")
        print(f"Total combinations: {np.prod([len(v) for v in param_grid.values()])}")
        
        # Generate all parameter combinations
        keys = list(param_grid.keys())
        all_results = []
        best_overall = None
        best_auc = -1.0
        start_time = time.time()
        
        for i, values in enumerate(itertools.product(*(param_grid[k] for k in keys))):
            config_dict = {k: v for k, v in zip(keys, values)}
            config = ModelConfig.from_dict(config_dict)
            
            print(f"\n[{i+1}/{np.prod([len(v) for v in param_grid.values()])}] "
                  f"Testing config: {config_dict}")
            
            try:
                result = self.train_single_config(config, dataset_name, max_epochs, patience)
                all_results.append(result)
                
                if result['best_val_auc'] > best_auc:
                    best_auc = result['best_val_auc']
                    best_overall = result
                    
                    # Save best model
                    model_path = f"{self.results_dir}/models/best_{dataset_name}_grid.pt"
                    torch.save(result['state_dict'], model_path)
                    
                    print(f"New best AUC: {best_auc:.4f} at epoch {result['best_epoch']}")
                
            except Exception as e:
                print(f"Error training config {config_dict}: {e}")
                continue
        
        # Save results
        results = {
            'dataset_name': dataset_name,
            'search_type': 'grid',
            'param_grid': param_grid,
            'total_combinations': len(all_results),
            'best_config': best_overall['config'] if best_overall else None,
            'best_val_auc': best_auc,
            'best_test_auc': best_overall['final_test_auc'] if best_overall else None,
            'all_results': all_results,
            'search_time': time.time() - start_time
        }
        
        # Save to files
        self._save_results(results, dataset_name, 'grid')
        
        return results
    
    
    def _save_results(self, results: Dict[str, Any], dataset_name: str, search_type: str):
        """Save search results to files."""
        
        # Save JSON results
        json_path = f"{self.results_dir}/{dataset_name}_{search_type}_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save CSV results
        csv_data = []
        for result in results['all_results']:
            row = result['config'].copy()
            row.update({
                'best_val_auc': result['best_val_auc'],
                'best_epoch': result['best_epoch'],
                'final_test_auc': result['final_test_auc'],
                'total_epochs': result['total_epochs']
            })
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        csv_path = f"{self.results_dir}/{dataset_name}_{search_type}_results.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"Results saved to {json_path} and {csv_path}")
    
    def plot_results(self, results: Dict[str, Any], save_plots: bool = True):
        """Create visualization plots for search results."""
        
        dataset_name = results['dataset_name']
        search_type = results['search_type']
        
        # Create results DataFrame
        df_data = []
        for result in results['all_results']:
            row = result['config'].copy()
            row.update({
                'best_val_auc': result['best_val_auc'],
                'final_test_auc': result['final_test_auc'],
                'best_epoch': result['best_epoch']
            })
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Plot 1: Validation AUC distribution
        plt.figure(figsize=(10, 6))
        plt.hist(df['best_val_auc'], bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(results['best_val_auc'], color='red', linestyle='--', 
                   label=f'Best: {results["best_val_auc"]:.4f}')
        plt.xlabel('Validation AUC')
        plt.ylabel('Frequency')
        plt.title(f'{dataset_name} - {search_type.title()} Search: Validation AUC Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        if save_plots:
            plt.savefig(f"{self.results_dir}/{dataset_name}_{search_type}_auc_distribution.png", 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 2: Parameter importance (if enough data)
        if len(df) > 10:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col not in ['best_val_auc', 'final_test_auc', 'best_epoch']]
            
            if len(numeric_cols) > 0:
                plt.figure(figsize=(12, 8))
                corr_matrix = df[numeric_cols + ['best_val_auc']].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                           square=True, fmt='.3f')
                plt.title(f'{dataset_name} - {search_type.title()} Search: Parameter Correlations')
                if save_plots:
                    plt.savefig(f"{self.results_dir}/{dataset_name}_{search_type}_correlations.png", 
                               dpi=300, bbox_inches='tight')
                plt.show()
        
        # Plot 3: Training curves for top 5 models
        top_5_indices = df.nlargest(5, 'best_val_auc').index
        plt.figure(figsize=(12, 8))
        
        for i, idx in enumerate(top_5_indices):
            result = results['all_results'][idx]
            history = result['history']
            label = f"Config {idx+1} (AUC: {result['best_val_auc']:.3f})"
            plt.plot(history['epochs'], history['val_aucs'], label=label, alpha=0.7)
        
        plt.xlabel('Epoch')
        plt.ylabel('Validation AUC')
        plt.title(f'{dataset_name} - {search_type.title()} Search: Top 5 Training Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        if save_plots:
            plt.savefig(f"{self.results_dir}/{dataset_name}_{search_type}_training_curves.png", 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def load_best_config(self, dataset_name: str) -> Optional[ModelConfig]:
        """Load the best configuration from a previous grid search."""
        
        json_path = f"{self.results_dir}/{dataset_name}_grid_results.json"
        if not os.path.exists(json_path):
            print(f"No results found for {dataset_name} grid search")
            return None
        
        with open(json_path, 'r') as f:
            results = json.load(f)
        
        if results['best_config'] is None:
            print(f"No best config found in {dataset_name} grid search")
            return None
        
        return ModelConfig.from_dict(results['best_config'])


def main():
    """Example usage of hyperparameter grid search."""
    
    # Initialize search
    search = HyperparameterSearch()
    
    # Define parameter grid for grid search
    param_grid = {
        'hidden_channels': [64, 128, 256],
        'num_layers': [2, 3, 4, 5],
        'dropout': [0.1],
        'learning_rate': [1e-2, 1e-3],
        'weight_decay': [1e-5, 1e-7]
    }
    
    # Run grid search for each dataset
    datasets = ['BBBP', 'BACE', 'HIV']
    
    for dataset_name in datasets:
        print(f"\n{'='*50}")
        print(f"Running grid search for {dataset_name}")
        print(f"{'='*50}")
        
        # Grid search
        results = search.grid_search(param_grid, dataset_name, max_epochs=80, patience=12)
        search.plot_results(results)
    
    print("\nGrid search completed!")


if __name__ == "__main__":
    main()

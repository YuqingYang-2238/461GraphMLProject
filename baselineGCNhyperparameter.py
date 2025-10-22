import optuna
from loadMoleculeData import load_datasets
from model_configs import ModelConfig, ConfigManager

from baselineGCN import GCN, do_epoch
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define datasets to optimize
datasets_to_optimize = ["BBBP", "BACE", "HIV"]

def objective(trial, dataset_name, train_loader, val_loader, original_dataset):
    """Objective function for Optuna optimization."""
    hidden_channels = trial.suggest_categorical('hidden_channels', [64, 128, 256])
    num_layers = trial.suggest_categorical('num_layers', [2, 3, 4, 5])
    dropout = trial.suggest_categorical('dropout', [0.1])
    learning_rate = trial.suggest_categorical('learning_rate', [1e-2, 1e-3])
    weight_decay = trial.suggest_categorical('weight_decay', [1e-5, 1e-7])
    
    # Create model using the pre-loaded dataset info
    model = GCN(in_channels=original_dataset.num_features, 
               config=ModelConfig(
                   hidden_channels=hidden_channels,
                   num_layers=num_layers,
                   dropout=dropout,
                   learning_rate=learning_rate,
                   weight_decay=weight_decay
               ))
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training with early stopping
    best_val_auc = -1.0
    for epoch in range(1, 81):  # 80 epochs max
        train_loss, train_auc = do_epoch(train_loader, training=True, model=model, 
                                       optimizer=optimizer, criterion=criterion)
        val_loss, val_auc = do_epoch(val_loader, training=False, model=model, criterion=criterion)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
        
        # Report intermediate result for pruning
        trial.report(val_auc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return best_val_auc

# Initialize config manager
config_manager = ConfigManager()

# Loop over all datasets
for dataset_name in datasets_to_optimize:
    print(f"\n{'='*60}")
    print(f"Starting hyperparameter optimization for {dataset_name}")
    print(f"{'='*60}")
    
    # Load dataset
    print(f"Loading {dataset_name} dataset...")
    datasets = load_datasets([dataset_name])
    dataset = datasets[0]
    train_loader = DataLoader(dataset['train_set'], batch_size=256, shuffle=True)
    val_loader = DataLoader(dataset['val_set'], batch_size=256, shuffle=False)
    test_loader = DataLoader(dataset['test_set'], batch_size=256, shuffle=False)
    original_dataset = dataset['train_set'].dataset
    print(f"Dataset loaded: {len(dataset['train_set'])} train, {len(dataset['val_set'])} val, {len(dataset['test_set'])} test samples")
    
    # Run optimization
    print(f"Starting hyperparameter optimization for {dataset_name}...")
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, dataset_name, train_loader, val_loader, original_dataset), 
        n_trials=48
    )
    
    print(f"Best parameters: {study.best_params}")
    print(f"Best validation AUC: {study.best_value:.4f}")
    
    # Save the best configuration
    print(f"\nSaving best configuration for {dataset_name}...")
    best_config = ModelConfig(
        hidden_channels=study.best_params['hidden_channels'],
        num_layers=study.best_params['num_layers'],
        dropout=study.best_params['dropout'],
        learning_rate=study.best_params['learning_rate'],
        weight_decay=study.best_params['weight_decay'],
        batch_size=256,  # Use the same batch size as in training
        epochs=100
    )
    
    config_manager.save_config(dataset_name, best_config)
    print(f"Best configuration saved for {dataset_name} dataset:")
    print(f"  Hidden channels: {best_config.hidden_channels}")
    print(f"  Number of layers: {best_config.num_layers}")
    print(f"  Dropout: {best_config.dropout}")
    print(f"  Learning rate: {best_config.learning_rate}")
    print(f"  Weight decay: {best_config.weight_decay}")
    print(f"  Batch size: {best_config.batch_size}")
    print(f"  Epochs: {best_config.epochs}")
    
    # Verify the saved configuration
    loaded_config = config_manager.load_config(dataset_name)
    print(f"Verification - loaded config matches: {loaded_config.to_dict() == best_config.to_dict()}")

print(f"\n{'='*60}")
print("All hyperparameter optimizations completed!")
print(f"{'='*60}")

# Print summary of all saved configurations
print("\nSummary of all saved configurations:")
for dataset_name in datasets_to_optimize:
    config = config_manager.load_config(dataset_name)
    print(f"\n{dataset_name}:")
    print(f"  Hidden channels: {config.hidden_channels}")
    print(f"  Number of layers: {config.num_layers}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Weight decay: {config.weight_decay}")
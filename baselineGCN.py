import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
from sklearn.metrics import roc_auc_score

from torch_geometric.loader import DataLoader

from loadMoleculeData import load_datasets
from model_configs import ConfigManager, ModelConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCHS = 100

class GCN(torch.nn.Module):
    def __init__(self, in_channels, config: ModelConfig):
        super().__init__()
        self.config = config
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, config.hidden_channels))
        for _ in range(config.num_layers-2):
            self.convs.append(GCNConv(config.hidden_channels, config.hidden_channels))
        self.convs.append(GCNConv(config.hidden_channels, config.hidden_channels))

        self.dropout = config.dropout
        self.lin = nn.Sequential(
            nn.Linear(config.hidden_channels, config.hidden_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(config.hidden_channels, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.float()
        
        # Apply GCN layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        g = global_mean_pool(x, batch)

        return self.lin(g).view(-1)

def do_epoch(loader, training: bool, model, optimizer = None, criterion = None):
    if training:
        model.train()
    else:
        model.eval()

    all_logits, all_labels, total_loss, n = [], [], 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        
        logits = logits.view(-1)
        labels = batch.y.view(-1).float() 
        loss = criterion(logits, labels)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    try:
        auc = roc_auc_score(all_labels, 1/(1+np.exp(-all_logits))) # TODO: Fix overflow
    except ValueError:
        auc = float("nan")
    return total_loss / n, auc

def train(datasets, config_manager: ConfigManager = None):
    if config_manager is None:
        config_manager = ConfigManager()
        config_manager.initialize_default_configs()
    
    for dataset in datasets:
        # Get dataset-specific configuration
        config = config_manager.get_config(dataset['name'])
        print(f"\nTraining {dataset['name']} with config: {config.to_dict()}")
        
        # Create data loaders with config batch size
        train_loader = DataLoader(dataset['train_set'], batch_size=config.batch_size, shuffle=True)
        val_loader   = DataLoader(dataset['val_set'],   batch_size=config.batch_size, shuffle=False)
        test_loader  = DataLoader(dataset['test_set'],  batch_size=config.batch_size, shuffle=False)

        # Get the original dataset to access num_features
        original_dataset = dataset['train_set'].dataset
        model = GCN(in_channels=original_dataset.num_features, config=config)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        criterion = nn.BCEWithLogitsLoss()
        
        best_val_auc = 0.0

        for epoch in range(1, config.epochs+1):
            train_loss, train_auc = do_epoch(train_loader, training=True, model=model, optimizer=optimizer, criterion=criterion)
            val_loss,   val_auc   = do_epoch(val_loader,   training=False, model=model, criterion=criterion)
            test_loss,  test_auc  = do_epoch(test_loader,  training=False, model=model, criterion=criterion)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), f"models/baselineGCN_{dataset['name']}.pt")

            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:02d} | "
                    f"Train loss {train_loss:.4f} AUC {train_auc:.3f} | "
                    f"Val AUC {val_auc:.3f}")

        print(f"\nBest Val AUC: {best_val_auc:.3f}")
        ckpt = torch.load(f"models/baselineGCN_{dataset['name']}.pt", map_location=device)
        model.load_state_dict(ckpt)
        _, test_auc = do_epoch(test_loader, training=False, model=model, criterion=criterion)
        print(f"Test AUC (best-val checkpoint): {test_auc:.3f}")

def get_probs(model, dataset):
    test_loader = DataLoader(dataset['test_set'], batch_size=256, shuffle=False)
    model.eval()
    all_logits = []
    for batch in test_loader:
        batch = batch.to(device)
        logits = model(batch)
        all_logits.append(logits.detach().cpu())
    return torch.cat(all_logits).numpy()

def load_model(model_path, dataset_name, config_manager: ConfigManager = None):
    """Load a trained model with its configuration."""
    if config_manager is None:
        config_manager = ConfigManager()
    
    config = config_manager.get_config(dataset_name)
    # Note: This requires the dataset to be available to get num_features
    # In practice, you might want to store num_features in the config or model file
    raise NotImplementedError("load_model requires dataset to get num_features. Consider storing it in config.")
    
def load_model_with_config(model_path, config: ModelConfig, in_channels: int):
    """Load a trained model with explicit configuration and input channels."""
    model = GCN(in_channels=in_channels, config=config)
    model = model.to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt)
    return model

def train_single_dataset(dataset_name: str, config_manager: ConfigManager = None):
    """Train a single dataset with its specific configuration."""
    if config_manager is None:
        config_manager = ConfigManager()
        config_manager.initialize_default_configs()
    
    datasets = load_datasets([dataset_name])
    if not datasets:
        raise ValueError(f"Dataset {dataset_name} not found")
    
    train(datasets, config_manager)

def update_dataset_config(dataset_name: str, **kwargs):
    """Update configuration for a specific dataset."""
    config_manager = ConfigManager()
    config = config_manager.update_config(dataset_name, **kwargs)
    print(f"Updated config for {dataset_name}: {config.to_dict()}")
    return config

def list_available_configs():
    """List all available dataset configurations."""
    config_manager = ConfigManager()
    configs = config_manager.list_configs()
    for dataset_name, config in configs.items():
        print(f"{dataset_name}: {config.to_dict()}")
    return configs
    
if __name__ == "__main__":
    # Initialize configuration manager
    config_manager = ConfigManager()
    config_manager.initialize_default_configs()
    
    # Load and train datasets
    datasets = load_datasets()
    train(datasets, config_manager)


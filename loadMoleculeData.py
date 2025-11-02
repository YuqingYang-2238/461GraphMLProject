import numpy as np
import os
import pandas as pd
from torch.utils.data import Subset
from torch_geometric.datasets import MoleculeNet

from deepchem.splits import ScaffoldSplitter
from deepchem.data import NumpyDataset

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def load_datasets(names = ["BBBP", "BACE", "HIV"], root = "data/moleculenet", scaffold = "standard"): # default is BBBP, BACE, and HIV which are single task classification datasets
    if scaffold == "standard":
        datasets = []
        for name in names:
            dataset = MoleculeNet(root=root, name=name)
            ids = [d.smiles for d in dataset]
            y = None
            X = ids  
            dc_ds = NumpyDataset(X=X, y=y, ids=ids)
            splitter = ScaffoldSplitter()   # Bemis–Murcko scaffold splitter
            train_idx, valid_idx, test_idx = splitter.split(dc_ds, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=42)
            train_set = Subset(dataset, train_idx)
            val_set = Subset(dataset, valid_idx)
            test_set = Subset(dataset, test_idx)
            datasets.append({
                'name': name,
                'train_set': train_set,
                'val_set': val_set,
                'test_set': test_set
            })
    return datasets


def load_datasets_csv(names = ["BBBP", "BACE", "HIV"], root = "data/moleculenet", scaffold = "standard", output_dir = "data/moleculenet"):
    datasets = load_datasets(names=names, root=root, scaffold=scaffold)
    
    # Extract SMILES and outcomes for each split
    def create_csv_data(subset):
        smiles_list = []
        outcome_list = []
        for item in subset:
            smiles_list.append(item.smiles)
            outcome_list.append(float(item.y.item()))
        
        df = pd.DataFrame({
            'smiles': smiles_list,
            'outcome': outcome_list
        })
        return df
    
    # Process each dataset
    for dataset_dict in datasets:
        name = dataset_dict['name']
        train_set = dataset_dict['train_set']
        val_set = dataset_dict['val_set']
        test_set = dataset_dict['test_set']
        
        # Create DataFrames for each split
        train_df = create_csv_data(train_set)
        val_df = create_csv_data(val_set)
        test_df = create_csv_data(test_set)
        
        # Save to CSV files
        dataset_dir = os.path.join(output_dir, name.lower())
        os.makedirs(dataset_dir, exist_ok=True)
        
        train_path = os.path.join(dataset_dir, f"{name.lower()}_train.csv")
        val_path = os.path.join(dataset_dir, f"{name.lower()}_val.csv")
        test_path = os.path.join(dataset_dir, f"{name.lower()}_test.csv")
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"Saved {name} datasets:")
        print(f"  Train: {train_path} ({len(train_df)} samples)")
        print(f"  Val:   {val_path} ({len(val_df)} samples)")
        print(f"  Test:  {test_path} ({len(test_df)} samples)")


if __name__ == "__main__":
    # datasets = load_datasets()
    load_datasets_csv()
    # print(datasets)
import os
import random
import numpy as np
import warnings

import torch
from torch.utils.data import Subset
from torch_geometric.datasets import MoleculeNet

from deepchem.splits import ScaffoldSplitter
from deepchem.data import NumpyDataset


warnings.warn(
    "loadBBBP.py is no longer used. Please use loadMoleculeData.py instead.",
    DeprecationWarning,
    stacklevel=2
)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(461)

root = "data/moleculenet"
dataset = MoleculeNet(root=root, name="BBBP")


ids = [d.smiles for d in dataset]
y = None
X = ids  
dc_ds = NumpyDataset(X=X, y=y, ids=ids)

splitter = ScaffoldSplitter()   # Bemis–Murcko scaffold splitter
train_idx, valid_idx, test_idx = splitter.split(dc_ds, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=42)

# Create train, validation, and test sets
train_set = Subset(dataset, train_idx)
val_set = Subset(dataset, valid_idx)
test_set = Subset(dataset, test_idx)


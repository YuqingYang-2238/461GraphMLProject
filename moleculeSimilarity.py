from typing import List, Literal, Optional, Iterable, Dict, Any
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, RDKFingerprint

# SMILES to molecule fingerprint
def smiles_to_fingerprint(
    smiles: str,
    kind: Literal["morgan", "rdk"] = "morgan",
    radius: int = 2,
    nBits: int = 2048
):
    """Covert each SMILES to RDkit fingerprint, return None if fails"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if kind == "morgan":
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    elif kind == "rdk":
        return RDKFingerprint(mol, fpSize=nBits)
    else:
        raise ValueError("Unknown fingerprint kind")

def batch_smiles_to_fps(
    smiles_list: Iterable[str],
    kind: Literal["morgan", "rdk"] = "morgan",
    radius: int = 2,
    nBits: int = 2048
):
    return [smiles_to_fingerprint(s, kind=kind, radius=radius, nBits=nBits) for s in smiles_list]

# K-most similar
def find_k_most_similar_molecules(
    test_smiles: Iterable[str],
    train_smiles: Iterable[str],
    y_train: Iterable[Any],
    k_neighbors: int = 5,
    fp_kind: Literal["morgan", "rdk"] = "morgan",
    fp_radius: int = 2,
    fp_nBits: int = 2048,
    return_indices: bool = True) -> List[Dict[str, Any]]:
    """
    For each test molecule, find the K most similar molecules in the training set based on Tanimoto similarity.
    Return a list, where each element is a dictionary containing the information of the K nearest neighbors for that test sample.
    """
    # 1) Precompute fingerprints of the training set (done once, reused for all test samples)
    train_smiles = list(train_smiles)
    y_train = np.asarray(list(y_train))
    train_fps = batch_smiles_to_fps(train_smiles, kind=fp_kind, radius=fp_radius, nBits=fp_nBits)

    # Record which training samples are invalid (SMILES parsing failed)
    valid_train = np.array([fp is not None for fp in train_fps])
    if not valid_train.any():
        raise ValueError("No valid fingerprints in training set.")
    valid_train_fps = [fp for fp in train_fps if fp is not None]
    valid_train_smiles = [s for s, v in zip(train_smiles, valid_train) if v]
    valid_y_train = y_train[valid_train]

    results = []

    # 2) Iterate through the test set, compute BulkTanimoto similarity with all training samples, then take Top-K
    for q_idx, q_smiles in enumerate(test_smiles):
        q_fp = smiles_to_fingerprint(q_smiles, kind=fp_kind, radius=fp_radius, nBits=fp_nBits)
        if q_fp is None:
            results.append({
                "test_index": q_idx,
                "test_smiles": q_smiles,
                "neighbors_smiles": [],
                "neighbors_labels": [],
                "neighbors_similarity": [],
                "neighbors_indices_in_train": [] if return_indices else None,
                "note": "invalid test SMILES"
            })
            continue

        # The larger the value, the more similar
        sims = np.array(DataStructs.BulkTanimotoSimilarity(q_fp, valid_train_fps), dtype=float)

        # Select the top K indices (Argpartition first, then sort by similarity in descending order)
        k = min(k_neighbors, sims.size)
        topk_idx = np.argpartition(sims, -k)[-k:]
        topk_idx = topk_idx[np.argsort(sims[topk_idx])[::-1]]

        results.append({
            "test_index": q_idx,
            "test_smiles": q_smiles,
            "neighbors_smiles": [valid_train_smiles[i] for i in topk_idx],
            "neighbors_labels":  [valid_y_train[i]    for i in topk_idx],
            "neighbors_similarity": sims[topk_idx].tolist(),
            "neighbors_indices_in_train": (
                np.nonzero(valid_train)[0][topk_idx].tolist() if return_indices else None
            )
        })

    return results
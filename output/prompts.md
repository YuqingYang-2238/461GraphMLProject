data named like:
..._{DATASET}_{SPLIT}_{K}K_{LOCAL_LLM}_prompt{PROMPT}.csv

V2"""
[INST]You are assisting with molecular property prediction. 
Your goal is to refine a baseline GNN prediction using information from K similar labeled molecules.

Instructions:
1) Consider the GNN’s initial predicted probability for BBB penetration.
2) Use the K most similar molecules (each with a known label y in {0,1} and similarity score in [0,1]) 
   to adjust this prediction toward values consistent with their outcomes.
3) Return a single real number between 0 and 1, representing the final probability 
   that the current molecule has the property.
   

Input:
- Current molecule (SMILES): {current_molecule_smiles}
- GNN initial prediction (probability): {initial_pred_gnn}
- K similar molecules (SMILES | label y | similarity):
  {similar_historical_molecules}

Output: one float in [0,1]. No text or explanation.[/INST]
"""


V2K0"""
[INST]You are assisting with molecular property prediction. 
Your goal is to refine a baseline GNN prediction.

Instructions:
1) Consider the GNN’s initial predicted probability for BBB penetration.
2) Adjust this prediction toward values consistent with their outcomes.
3) Return a single real number between 0 and 1, representing the final probability 
   that the current molecule has the property.
   

Input:
- Current molecule (SMILES): {current_molecule_smiles}
- GNN initial prediction (probability): {initial_pred_gnn}

Output: one float in [0,1]. No text or explanation.[/INST]
"""
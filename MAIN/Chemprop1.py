#!/usr/bin/env python
"""
Full Antibiotic Recommendation Pipeline

This script implements an end-to-end pipeline for recommending an antibiotic based on a bacterial cell wall SMILES input.
It performs the following steps:
1. Accepts a bacterial cell wall SMILES string as input.
2. Loads an antibiotic library (CSV) containing candidate antibiotic SMILES.
3. Uses a pre-trained Chemprop model to predict inhibition (activity) values for the candidate antibiotics.
4. Computes the Tanimoto similarity between the bacterial input and each candidate antibiotic.
5. (Optionally) Combines the predicted inhibition and similarity into a final score.
6. Outputs a final CSV file with antibiotic SMILES and similarity scores.
"""

import subprocess
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

def run_chemprop_prediction(library_csv, model_path, output_csv, smiles_col="smiles"):
    """
    Run Chemprop prediction on the antibiotic library CSV to obtain predicted inhibition values.
    
    Parameters:
      library_csv (str): Path to the antibiotic library CSV.
      model_path (str): Path to the trained Chemprop model file (e.g., best.pt).
      output_csv (str): Where to save the predictions.
      smiles_col (str): Name of the SMILES column.
    
    Returns:
      None (predictions are saved to output_csv)
    """
    cmd = [
        "chemprop_predict",  # Use the chemprop_predict command
        "--test_path", library_csv,
        "--checkpoint_path", model_path,
        "--preds_path", output_csv,
        "--smiles_columns", smiles_col
    ]
    print("Running Chemprop prediction...")
    subprocess.run(cmd, check=True)
    print("Chemprop prediction complete. Predictions saved to", output_csv)

def compute_similarity_for_library(bacteria_smiles, library_df, smiles_col="smiles"):
    """
    Compute Tanimoto similarity between the bacterial input and each candidate antibiotic.
    
    Parameters:
      bacteria_smiles (str): SMILES string for the bacterial cell wall component.
      library_df (pd.DataFrame): DataFrame of candidate antibiotics.
      smiles_col (str): Column name for antibiotic SMILES.
      
    Returns:
      pd.Series: A series of similarity scores.
    """
    bacteria_mol = Chem.MolFromSmiles(bacteria_smiles)
    if bacteria_mol is None:
        raise ValueError("Invalid bacterial SMILES input!")
    bacteria_fp = AllChem.GetMorganFingerprintAsBitVect(bacteria_mol, radius=2, nBits=2048)
    
    def similarity(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return 0.0
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        return DataStructs.TanimotoSimilarity(bacteria_fp, fp)
    
    return library_df[smiles_col].apply(similarity)
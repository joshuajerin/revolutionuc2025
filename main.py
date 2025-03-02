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
        "chemprop", "predict",
        "--test-path", library_csv,
        "--model-path", model_path,
        "--preds-path", output_csv,
        "--smiles-columns", smiles_col
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

def main():
    # ----- Step 1: Get Bacterial Input -----
    bacteria_smiles = input("Enter the SMILES for the bacterial cell wall component: ").strip()
    
    # ----- Step 2: Load the Antibiotic Library -----
    library_csv = "antibiotic_library.csv"  # Ensure this file is in your working directory
    try:
        df_library = pd.read_csv(library_csv)
    except Exception as e:
        print(f"Error reading '{library_csv}': {e}")
        return
    
    if "smiles" not in df_library.columns:
        print("Error: 'smiles' column not found in the antibiotic library CSV.")
        return

    # ----- Step 3: Run Chemprop Prediction -----
    # This will add a predicted inhibition value for each antibiotic candidate.
    chemprop_model_path = "chemprop_inhibition_model/model_0/best.pt"  # Update if needed
    temp_preds_csv = "temp_predictions.csv"
    run_chemprop_prediction(library_csv, chemprop_model_path, temp_preds_csv, smiles_col="smiles")
    
    # Load the predictions
    try:
        df_preds = pd.read_csv(temp_preds_csv)
    except Exception as e:
        print(f"Error reading predictions file '{temp_preds_csv}': {e}")
        return

    df_preds.rename(columns={"activity": "prediction"}, inplace=True)
    
    # Assume the predictions file has a column "prediction" corresponding to predicted inhibition.
    if "prediction" not in df_preds.columns:
        print("Error: 'prediction' column not found in the Chemprop predictions CSV.")
        return
    
    # Merge the predictions with the original antibiotic library based on SMILES
    df_library = df_library.merge(df_preds, on="smiles", how="left")
    
    # ----- Step 4: Compute Similarity Scores -----
    print("Computing similarity scores between bacterial input and candidate antibiotics...")
    df_library["similarity_score"] = compute_similarity_for_library(bacteria_smiles, df_library, smiles_col="smiles")
    
    # ----- Step 5: Combine Metrics (Optional) -----
    # For this example, we'll calculate a final score = predicted_inhibition * similarity_score.
    # You can modify this formula as desired.
    df_library["final_score"] = df_library["prediction"] * df_library["similarity_score"]
    
    # ----- Step 6: Rank Candidates and Save Final Output -----
    # The final output file will include only antibiotic SMILES and similarity scores, as requested.
    df_ranked = df_library.sort_values(by="final_score", ascending=False).reset_index(drop=True)
    
    # Save a final CSV with the columns "smiles" and "similarity_score"
    final_output_csv = "final_antibiotic_recommendations.csv"
    df_ranked[["smiles", "similarity_score"]].to_csv(final_output_csv, index=False)
    
    # Also print the top candidate for convenience.
    top_candidate = df_ranked.iloc[0]
    print("\nTop Recommended Antibiotic Candidate:")
    print("SMILES:", top_candidate["smiles"])
    print("Predicted Inhibition:", top_candidate["prediction"])
    print("Similarity Score:", top_candidate["similarity_score"])
    print("Final Score:", top_candidate["final_score"])
    
    print(f"\nFinal recommendations saved to '{final_output_csv}'.")

if __name__ == "__main__":
    main()

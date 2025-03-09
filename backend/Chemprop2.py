from Chemprop1 import run_chemprop_prediction, compute_similarity_for_library
import pandas as pd
import os
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    try:
        # ----- Step 1: Get Bacterial Input from environment variable -----
        bacteria_smiles = os.environ.get('SMILES_INPUT')
        if not bacteria_smiles:
            raise ValueError("No SMILES input provided in environment variable SMILES_INPUT")
        
        logger.info(f"Using SMILES input: {bacteria_smiles}")
        
        # ----- Step 2: Load the Antibiotic Library -----
        library_csv = "antibiotic_library.csv"
        if not os.path.exists(library_csv):
            raise FileNotFoundError(f"Antibiotic library not found: {library_csv}")
        
        logger.info(f"Loading antibiotic library from {library_csv}")
        df_library = pd.read_csv(library_csv)
        
        if "smiles" not in df_library.columns:
            raise ValueError("'smiles' column not found in the antibiotic library CSV")

        # ----- Step 3: Run Chemprop Prediction -----
        chemprop_model_path = "chemprop_inhibition_model/model_0/best.pt"
        if not os.path.exists(chemprop_model_path):
            raise FileNotFoundError(f"Chemprop model not found: {chemprop_model_path}")
        
        temp_preds_csv = "temp_predictions.csv"
        logger.info("Running Chemprop prediction...")
        run_chemprop_prediction(library_csv, chemprop_model_path, temp_preds_csv, smiles_col="smiles")
        
        # Load the predictions
        logger.info("Loading predictions...")
        df_preds = pd.read_csv(temp_preds_csv)
        df_preds.rename(columns={"activity": "prediction"}, inplace=True)
        
        if "prediction" not in df_preds.columns:
            raise ValueError("'prediction' column not found in Chemprop predictions")
        
        # Merge predictions with library
        df_library = df_library.merge(df_preds, on="smiles", how="left")
        
        # ----- Step 4: Compute Similarity Scores -----
        logger.info("Computing similarity scores...")
        df_library["similarity_score"] = compute_similarity_for_library(bacteria_smiles, df_library, smiles_col="smiles")
        
        # ----- Step 5: Calculate Final Scores -----
        df_library["final_score"] = df_library["prediction"] * df_library["similarity_score"]
        
        # ----- Step 6: Save and Output Results -----
        df_ranked = df_library.sort_values(by="final_score", ascending=False).reset_index(drop=True)
        
        final_output_csv = "final_antibiotic_recommendations.csv"
        df_ranked[["smiles", "similarity_score", "prediction", "final_score"]].to_csv(final_output_csv, index=False)
        
        # Print top candidate
        #top_candidate = df_ranked.iloc[0]
       # print("\nTop Recommended Antibiotic Candidate:")
        #print(f"SMILES: {top_candidate['smiles']}")
        #print(f"Predicted Inhibition: {top_candidate['prediction']:.4f}")
        #print(f"Similarity Score: {top_candidate['similarity_score']:.4f}")
        #print(f"Final Score: {top_candidate['final_score']:.4f}")
        
        logger.info(f"Final recommendations saved to '{final_output_csv}'")
        
    except Exception as e:
        logger.error(f"Error in Chemprop2: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

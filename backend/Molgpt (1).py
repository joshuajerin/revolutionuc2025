import pandas as pd
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from rdkit import Chem
import logging
import os
import sys

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer():
    try:
        logger.info("Loading MolGPT tokenizer and model...")
        tokenizer = PreTrainedTokenizerFast.from_pretrained("jonghyunlee/MolGPT_pretrained-by-ZINC15")
        tokenizer.pad_token = "<pad>"
        tokenizer.bos_token = "<bos>"
        tokenizer.eos_token = "<eos>"
        model = GPT2LMHeadModel.from_pretrained("jonghyunlee/MolGPT_pretrained-by-ZINC15")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model and tokenizer: {str(e)}")
        raise

def preprocess_data(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")
            
        logger.info(f"Reading data from {file_path}")
        df = pd.read_csv(file_path)
        
        if df.empty:
            raise ValueError("Input file is empty")
        
        # Get SMILES column
        if "smiles" in df.columns:
            return df["smiles"]
        else:
            return df.iloc[:, 0]  # Use first column if no "smiles" column
            
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

def tokenize_smiles(smiles_list, tokenizer):
    try:
        logger.info("Tokenizing SMILES data...")
        return tokenizer(smiles_list, padding=True, truncation=True, max_length=128, return_tensors="pt")
    except Exception as e:
        logger.error(f"Error tokenizing SMILES: {str(e)}")
        raise

def generate_smiles(model, tokenizer, num_sequences=10, temperature=1.0):
    try:
        logger.info(f"Generating {num_sequences} SMILES sequences...")
        outputs = model.generate(
            max_length=128,
            num_return_sequences=num_sequences,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=temperature,
            return_dict_in_generate=True,
        )
        
        generated_smiles = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs.sequences]
        return generated_smiles
    except Exception as e:
        logger.error(f"Error generating SMILES: {str(e)}")
        raise

def validate_smiles(smiles):
    try:
        molecule = Chem.MolFromSmiles(smiles)
        return bool(molecule)  # Convert to bool for clarity
    except Exception as e:
        logger.error(f"Error validating SMILES: {str(e)}")
        return False

def main():
    try:
        # Step 1: Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer()
        
        # Step 2: Load and preprocess data
        input_file = 'final_antibiotic_recommendations.csv'
        df = preprocess_data(input_file)
        
        # Step 3: Tokenize SMILES
        smiles_data = df.tolist()
        tokenized_data = tokenize_smiles(smiles_data, tokenizer)
        
        # Step 4: Generate new SMILES
        generated_smiles = generate_smiles(model, tokenizer, num_sequences=10, temperature=1.0)
        
        # Step 5: Validate and save results
        valid_smiles = []
        for idx, smiles in enumerate(generated_smiles):
            is_valid = validate_smiles(smiles)
            if is_valid:
                valid_smiles.append(smiles)
            logger.info(f"Generated SMILES {idx + 1}: {smiles} (Valid: {is_valid})")
        
        if not valid_smiles:
            raise ValueError("No valid SMILES were generated")
            
        # Save the first valid SMILES
        output_file = "generated_smiles.txt"
        with open(output_file, "w") as f:
            f.write(valid_smiles[0])
        logger.info(f"Saved first valid SMILES to {output_file}")
        
    except Exception as e:
        logger.error(f"Error in MolGPT pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
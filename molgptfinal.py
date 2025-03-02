# Install necessary libraries
!pip install transformers
!pip install tokenizers
!pip install rdkit
import pandas as pd
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from rdkit import Chem
# Step 1: Load tokenizer and model directly from Hugging Face
tokenizer = PreTrainedTokenizerFast.from_pretrained("jonghyunlee/MolGPT_pretrained-by-ZINC15")
tokenizer.pad_token = "<pad>"
tokenizer.bos_token = "<bos>"
tokenizer.eos_token = "<eos>"
model = GPT2LMHeadModel.from_pretrained("jonghyunlee/MolGPT_pretrained-by-ZINC15")
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Remove the first row and use only the first column (SMILES)
    df = df.iloc[1:, 0]  # Skip the first row and select the first column
    
    return df
df = preprocess_data('inhibition_data.csv')
def tokenize_smiles(smiles_list):
    return tokenizer(smiles_list, padding=True, truncation=True, max_length=128, return_tensors="pt")
smiles_data = df.tolist()  # Convert the SMILES column to a list
tokenized_data = tokenize_smiles(smiles_data)
def generate_smiles(model, tokenizer, num_sequences=10, temperature=1.0):
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
generated_smiles = generate_smiles(model, tokenizer, num_sequences=10, temperature=1.0)
def validate_smiles(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule:
        return True  # Valid SMILES
    else:
        return False  # Invalid SMILES

# Step 6: Print generated SMILES and validation result
for idx, smiles in enumerate(generated_smiles):
    is_valid = validate_smiles(smiles)
    print(f"Generated SMILES {idx + 1}: {smiles}")
    print(f"Valid: {is_valid}\n")



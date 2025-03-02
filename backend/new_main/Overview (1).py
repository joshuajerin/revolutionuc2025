from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
import py3Dmol
import pandas as pd
import os
import logging
import sys

# Configure logging to use stdout instead of stderr
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # Use sys.stdout instead of os.stdout
)
logger = logging.getLogger(__name__)

def get_top_candidate_smiles():
    try:
        # Read the predictions file
        preds_file = "temp_predictions.csv"
        if not os.path.exists(preds_file):
            raise FileNotFoundError(f"Predictions file not found: {preds_file}")
            
        logger.info(f"Reading predictions from {preds_file}")
        df = pd.read_csv(preds_file)
        
        if df.empty:
            raise ValueError("Predictions file is empty")
            
        # Get the top candidate (first row)
        top_smiles = df.iloc[0]["smiles"]
        if not top_smiles:
            raise ValueError("No SMILES found in predictions")
            
        return top_smiles
        
    except Exception as e:
        logger.error(f"Error getting top candidate: {str(e)}")
        raise

def create_3d_model(smiles, output_path):
    try:
        logger.info("Creating 3D model...")
        # Create RDKit molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")
            
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate 3D coordinates
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        
        # Create HTML content with embedded 3D viewer
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <script src="https://3dmol.org/build/3Dmol-min.js"></script>
            <style>
                .viewer_3Dmoljs {
                    width: 100%;
                    height: 100%;
                    position: relative;
                }
            </style>
        </head>
        <body>
            <div id="container" class="viewer_3Dmoljs" style="width: 400px; height: 400px; position: relative;"></div>
            <script>
                let viewer = $3Dmol.createViewer("container", {backgroundColor: "white"});
                let pdb = `{}`;
                viewer.addModel(pdb, "pdb");
                viewer.setStyle({}, {stick: {}});
                viewer.zoomTo();
                viewer.render();
            </script>
        </body>
        </html>
        """.format(Chem.MolToPDBBlock(mol))
        
        # Save the HTML file
        with open(output_path, 'w') as f:
            f.write(html_content)
            
        logger.info(f"3D model saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating 3D model: {str(e)}")
        raise

def create_molecule(smiles):
    try:
        logger.info("Converting SMILES to molecule")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        return mol
    except Exception as e:
        logger.error(f"Error creating molecule: {str(e)}")
        raise

def save_molecule_image(mol, output_path):
    try:
        logger.info(f"Generating molecule image and saving to {output_path}")
        img = Draw.MolToImage(mol)
        img.save(output_path)
    except Exception as e:
        logger.error(f"Error saving molecule image: {str(e)}")
        raise

def calculate_properties(mol):
    try:
        logger.info("Calculating molecular properties")
        properties = {
            "molecular_weight": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "h_acceptors": Descriptors.NumHAcceptors(mol),
            "h_donors": Descriptors.NumHDonors(mol),
            "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol)
        }
        return properties
    except Exception as e:
        logger.error(f"Error calculating properties: {str(e)}")
        raise

def save_properties(properties, smiles, output_path):
    try:
        logger.info(f"Saving molecular properties to {output_path}")
        description = f"The molecule with SMILES: {smiles} has the following properties:\n"
        description += f"Molecular weight: {properties['molecular_weight']:.2f} g/mol\n"
        description += f"LogP (lipophilicity): {properties['logp']:.2f}\n"
        description += f"Number of hydrogen bond acceptors: {properties['h_acceptors']}\n"
        description += f"Number of hydrogen bond donors: {properties['h_donors']}\n"
        description += f"Number of rotatable bonds: {properties['num_rotatable_bonds']}\n"
        
        with open(output_path, "w") as f:
            f.write(description)
    except Exception as e:
        logger.error(f"Error saving properties: {str(e)}")
        raise

def main():
    try:
        # Step 1: Get SMILES from top candidate
        smiles = get_top_candidate_smiles()
        
        # Step 2: Create molecule
        mol = create_molecule(smiles)
        
        # Step 3: Save 2D molecule image
        image_file = "molecule.png"
        save_molecule_image(mol, image_file)
        
        # Step 4: Create and save 3D model
        model_file = "molecule_3d.html"
        create_3d_model(smiles, model_file)
        
        # Step 5: Calculate and save properties
        properties = calculate_properties(mol)
        properties_file = "molecule_properties.txt"
        save_properties(properties, smiles, properties_file)
        
        logger.info("Overview process completed successfully")
        
    except Exception as e:
        logger.error(f"Error in Overview pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stdout)  # Print to stdout instead of stderr
        sys.exit(1) 
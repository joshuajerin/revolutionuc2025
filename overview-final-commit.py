from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
import os
import logging
import sys
import py3Dmol  # You need to install py3Dmol using pip
from IPython.display import Image, display 

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # Use sys.stdout instead of os.stdout
)
logger = logging.getLogger(__name__)


def read_smiles_from_file(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"SMILES file not found: {file_path}")
            
        logger.info(f"Reading SMILES from {file_path}")
        with open(file_path, "r") as f:
            smiles = f.read().strip()
            
        if not smiles:
            raise ValueError("Empty SMILES string in file")
            
        return smiles
    except Exception as e:
        logger.error(f"Error reading SMILES: {str(e)}")
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


def display_2d_image(mol, output_path="molecule_2d.png"):
    try:
        logger.info("Displaying 2D visualization")
        img = Draw.MolToImage(mol)
        img.save(output_path)
        display(Image(filename=output_path))  # Display the image inline (useful for Jupyter Notebooks)
    except Exception as e:
        logger.error(f"Error displaying 2D visualization: {str(e)}")
        raise

def display_3d(mol):
    try:
        logger.info("Displaying 3D visualization")
        mol_block = Chem.MolToMolBlock(mol)
        viewer = py3Dmol.view(width=800, height=400)
        viewer.addModel(mol_block, "mol")
        viewer.setStyle({'stick': {}})
        viewer.zoomTo()
        viewer.show()
    except Exception as e:
        logger.error(f"Error displaying 3D visualization: {str(e)}")
        raise

def main():
    try:
        # Step 1: Read SMILES from file
        input_file = "generated_smiles.txt"
        smiles = read_smiles_from_file(input_file)
        
        # Step 2: Create molecule
        mol = create_molecule(smiles)
        
        # Step 3: Save and display 2D image
        image_file = "molecule_2d.png"
        display_2d_image(mol, image_file)
        
        # Step 4: Display 3D visualization
        display_3d(mol)
        
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
        print(f"Error: {str(e)}", file=sys.stdout)
        sys.exit(1)
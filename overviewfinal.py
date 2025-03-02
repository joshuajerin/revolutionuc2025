from rdkit import Chem
from rdkit.Chem import Draw

smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Example: Aspirin
mol = Chem.MolFromSmiles(smiles)
img = Draw.MolToImage(mol)
img.show()
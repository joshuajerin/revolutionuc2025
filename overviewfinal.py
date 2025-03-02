from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol
from IPython.display import display
from rdkit.Chem import Draw



# Example SMILES string (Aspirin)
smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"

# Convert SMILES to molecule
mol = Chem.MolFromSmiles(smiles)

img = Draw.MolToImage(mol)
display(img)

# Add hydrogen atoms to improve 3D structure
mol = Chem.AddHs(mol)

# Generate 3D coordinates
AllChem.EmbedMolecule(mol, AllChem.ETKDG())

# Convert to mol block format (needed for visualization)
mol_block = Chem.MolToMolBlock(mol)

# Display using Py3Dmol
viewer = py3Dmol.view(width=400, height=400)
viewer.addModel(mol_block, "mol")
viewer.setStyle({"stick": {}})
viewer.zoomTo()
viewer.show()

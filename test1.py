from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from rdkit import Chem

from synthemol.constants import (
    BUILDING_BLOCKS_PATH, FINGERPRINT_TYPES, MODEL_TYPES, OPTIMIZATION_TYPES, REAL_BUILDING_BLOCK_ID_COL, SCORE_COL, SMILES_COL
)
from synthemol.generate import Generator, Node
from synthemol.generate.utils import create_model_scoring_fn, save_generated_molecules
from synthemol.models import chemprop_load, chemprop_load_scaler, chemprop_predict_on_molecule_ensemble
from synthemol.reactions import Reaction, QueryMol, set_all_building_blocks, load_and_set_allowed_reaction_building_blocks
from synthemol.utils import convert_to_mol, random_choice

# Helper function to print test results
def print_test_result(test_name: str, passed: bool) -> None:
    status = "PASSED" if passed else "FAILED"
    print(f"{test_name}: {status}")

# Test convert_to_mol function
def test_convert_to_mol() -> None:
    smiles = "CCO"
    mol = convert_to_mol(smiles)
    passed = mol is not None and Chem.MolToSmiles(mol) == smiles
    print_test_result("test_convert_to_mol", passed)

# Test random_choice function
def test_random_choice() -> None:
    rng = np.random.default_rng(seed=0)
    array = [1, 2, 3, 4, 5]
    choice = random_choice(rng, array)
    passed = choice in array
    print_test_result("test_random_choice", passed)

# Test QueryMol class
def test_query_mol() -> None:
    smarts = "[OH1][C:1]([*:2])=[O:3]"
    query_mol = QueryMol(smarts)
    passed = query_mol.has_match("CCO")
    print_test_result("test_query_mol", passed)

# Test Reaction class
def test_reaction() -> None:
    reactants = [QueryMol("[OH1][C:1]([*:2])=[O:3]")]
    product = QueryMol("[OH1][C:1]([*:2])=[O:3]")
    reaction = Reaction(reactants, product, reaction_id=1)
    passed = reaction.num_reactants == 1
    print_test_result("test_reaction", passed)

# Test Node class
def test_node() -> None:
    scoring_fn = lambda x: 1.0
    node = Node(explore_weight=1.0, scoring_fn=scoring_fn, node_id=1)
    passed = node.P == 0.0
    print_test_result("test_node", passed)

# Test Generator class
def test_generator() -> None:
    building_block_smiles_to_id = {"CCO": 1}
    scoring_fn = lambda x: 1.0
    generator = Generator(
        building_block_smiles_to_id=building_block_smiles_to_id,
        max_reactions=1,
        scoring_fn=scoring_fn,
        explore_weight=1.0,
        num_expand_nodes=None,
        optimization="maximize",
        reactions=(),
        rng_seed=0,
        no_building_block_diversity=False,
        store_nodes=False,
        verbose=False
    )
    passed = generator is not None
    print_test_result("test_generator", passed)

# Test Chemprop model loading
def test_chemprop_load() -> None:
    model_path = Path("path_to_chemprop_model.pt")
    if model_path.exists():
        model = chemprop_load(model_path)
        passed = model is not None
        print_test_result("test_chemprop_load", passed)
    else:
        print_test_result("test_chemprop_load", False)

# Test Chemprop prediction
def test_chemprop_predict_on_molecule_ensemble() -> None:
    model_path = Path("path_to_chemprop_model.pt")
    if model_path.exists():
        model = chemprop_load(model_path)
        scaler = chemprop_load_scaler(model_path)
        smiles = "CCO"
        prediction = chemprop_predict_on_molecule_ensemble([model], smiles, scalers=[scaler])
        passed = isinstance(prediction, float)
        print_test_result("test_chemprop_predict_on_molecule_ensemble", passed)
    else:
        print_test_result("test_chemprop_predict_on_molecule_ensemble", False)

# Test full molecule generation pipeline
def test_generate_molecules() -> None:
    model_path = Path("path_to_model")
    save_dir = Path("test_output")
    save_dir.mkdir(exist_ok=True)

    generate(
        model_path=model_path,
        model_type="random_forest",
        save_dir=save_dir,
        building_blocks_path=BUILDING_BLOCKS_PATH,
        reaction_to_building_blocks_path=None,
        building_blocks_id_column=REAL_BUILDING_BLOCK_ID_COL,
        building_blocks_score_column=SCORE_COL,
        building_blocks_smiles_column=SMILES_COL,
        reactions=(),
        max_reactions=1,
        n_rollout=1,
        explore_weight=1.0,
        num_expand_nodes=None,
        optimization="maximize",
        rng_seed=0,
        no_building_block_diversity=False,
        store_nodes=False,
        verbose=False
    )

    passed = (save_dir / "molecules.csv").exists()
    print_test_result("test_generate_molecules", passed)

# Run all tests
def run_tests() -> None:
    test_convert_to_mol()
    test_random_choice()
    test_query_mol()
    test_reaction()
    test_node()
    test_generator()
    test_chemprop_load()
    test_chemprop_predict_on_molecule_ensemble()
    test_generate_molecules()

if __name__ == "__main__":
    run_tests()
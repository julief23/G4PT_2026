import numpy as np

# fix for Mordred with numpy >= 1.24
if not hasattr(np, "float"):
    np.float = float

from rdkit import Chem
from mordred import Calculator, descriptors
import pandas as pd

# Initialize once (very important)
calc = Calculator(descriptors, ignore_3D=True)

def compute_mordred_from_smiles_list(smiles_list):
    """
    Compute Mordred descriptors from a list of SMILES.
    Returns DataFrame aligned with valid molecules only.
    """

    mols = []
    valid_smiles = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi).strip())
        if mol is not None:
            mols.append(mol)
            valid_smiles.append(smi)

    if len(mols) == 0:
        raise ValueError("No valid SMILES provided.")

    desc_df = calc.pandas(mols)

    # Clean missing / infinite
    desc_df = desc_df.replace([float("inf"), -float("inf")], 0)
    desc_df = desc_df.fillna(0)

    desc_df.insert(0, "SMILES", valid_smiles)

    return desc_df
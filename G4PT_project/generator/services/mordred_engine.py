import numpy as np

# Fix for Mordred with numpy >= 1.24
if not hasattr(np, "float"):
    np.float = float

import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors

from .smiles_utils import canonicalize_smiles


# Initialize once
calc = Calculator(descriptors, ignore_3D=True)


def compute_mordred_from_smiles_list(smiles_list):
    """
    Compute Mordred descriptors from a list of SMILES.
    Invalid SMILES are skipped.
    Returned SMILES are canonicalized.
    """

    mols = []
    valid_smiles = []

    for smi in smiles_list:
        canonical = canonicalize_smiles(smi)

        if canonical is None:
            continue

        mol = Chem.MolFromSmiles(canonical)

        if mol is None:
            continue

        mols.append(mol)
        valid_smiles.append(canonical)

    if len(mols) == 0:
        raise ValueError("No valid SMILES provided.")

    desc_df = calc.pandas(mols)

    desc_df = desc_df.replace([np.inf, -np.inf], np.nan)
    desc_df = desc_df.fillna(0)

    desc_df.insert(0, "SMILES", valid_smiles)

    return desc_df
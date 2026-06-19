from rdkit import Chem


def canonicalize_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return Chem.MolToSmiles(mol, canonical=True)
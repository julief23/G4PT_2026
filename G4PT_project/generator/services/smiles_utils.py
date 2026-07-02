from rdkit import Chem

def canonicalize_smiles(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles).strip())

    if mol is None:
        return None

    return Chem.MolToSmiles(mol, canonical=True)


def clean_smiles_list(smiles_list):
    clean = []

    for smi in smiles_list:
        canonical = canonicalize_smiles(smi)

        if canonical is not None:
            clean.append(canonical)

    return sorted(set(clean))
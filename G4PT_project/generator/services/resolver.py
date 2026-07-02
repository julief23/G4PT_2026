import requests
from urllib.parse import quote
from rdkit import Chem
from rdkit import RDLogger

from .smiles_utils import canonicalize_smiles


RDLogger.DisableLog("rdApp.*")


def generate_name_variants(name):
    name = str(name).strip()

    variants = [
        name,
        name.lower(),
        name.upper(),
        name.title(),
        name.replace("-", ""),
        name.replace("-", " "),
        name.replace(" ", "-"),
    ]

    return list(dict.fromkeys(variants))


HEADERS = {"User-Agent": "G4PT/1.0"}


def get_pubchem_smiles_from_cid(cid, debug=False):
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/"
        f"{cid}/JSON"
    )

    try:
        response = requests.get(url, timeout=10)

        if debug:
            print("PubChem PUG-View URL:", url)
            print("PubChem PUG-View status:", response.status_code)

        if response.status_code != 200:
            return None

        data = response.json()

        def search_sections(section):
            if isinstance(section, dict):
                heading = section.get("TOCHeading", "")

                if heading == "SMILES":
                    infos = section.get("Information", [])
                    for info in infos:
                        value = info.get("Value", {})
                        strings = value.get("StringWithMarkup", [])
                        if strings:
                            smiles = strings[0].get("String")
                            return canonicalize_smiles(smiles)

                for subsection in section.get("Section", []):
                    found = search_sections(subsection)
                    if found:
                        return found

            return None

        return search_sections(data["Record"])

    except Exception as e:
        if debug:
            print("PubChem PUG-View error:", e)

        return None

def resolve_name_with_pubchem(name, debug=False):
    name = str(name).strip()
    encoded = quote(name, safe="")

    # ========================================================
    # 1. PubChem name -> CID
    # ========================================================

    cid_url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{encoded}/cids/JSON"
    )

    try:
        r = requests.get(cid_url, headers=HEADERS, timeout=10)

        if debug:
            print("PubChem CID URL:", cid_url)
            print("PubChem CID status:", r.status_code)
            print("PubChem CID response:", r.text[:300])

        if r.status_code == 200:
            data = r.json()
            cids = data.get("IdentifierList", {}).get("CID", [])

            for cid in cids:
                canonical = get_pubchem_smiles_from_cid(cid)

                if canonical:
                    return canonical

    except Exception as e:
        if debug:
            print("PubChem name->CID error:", e)

    # ========================================================
    # 2. Entrez fallback
    # ========================================================

    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    try:
        r = requests.get(
            search_url,
            headers=HEADERS,
            params={
                "db": "pccompound",
                "term": name,
                "retmode": "json",
                "retmax": 10
            },
            timeout=10
        )

        if debug:
            print("Entrez status:", r.status_code)
            print("Entrez response:", r.text[:300])

        if r.status_code == 200:
            data = r.json()
            cid_list = data.get("esearchresult", {}).get("idlist", [])

            for cid in cid_list:
                canonical = get_pubchem_smiles_from_cid(cid)

                if canonical:
                    return canonical

    except Exception as e:
        if debug:
            print("Entrez error:", e)

    return None


def resolve_name_with_chembl_exact(name):
    search_url = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"

    try:
        response = requests.get(
            search_url,
            params={"pref_name__iexact": name, "limit": 1},
            timeout=10
        )
    except requests.RequestException:
        return None

    if response.status_code != 200:
        return None

    try:
        data = response.json()
        molecules = data.get("molecules", [])

        if not molecules:
            return None

        structures = molecules[0].get("molecule_structures")

        if not structures:
            return None

        smiles = structures.get("canonical_smiles")

    except (KeyError, IndexError, ValueError, AttributeError, TypeError):
        return None

    return canonicalize_smiles(smiles)


def resolve_name_with_chembl_search(name):
    search_url = "https://www.ebi.ac.uk/chembl/api/data/molecule/search.json"

    try:
        response = requests.get(
            search_url,
            params={"q": name, "limit": 5},
            timeout=10
        )
    except requests.RequestException:
        return None

    if response.status_code != 200:
        return None

    try:
        data = response.json()
        molecules = data.get("molecules", [])

        for molecule in molecules:
            structures = molecule.get("molecule_structures")

            if not structures:
                continue

            smiles = structures.get("canonical_smiles")
            canonical = canonicalize_smiles(smiles)

            if canonical:
                return canonical

    except (KeyError, ValueError, AttributeError, TypeError):
        return None

    return None


def resolve_name_with_chembl(name):
    canonical = resolve_name_with_chembl_exact(name)

    if canonical:
        return canonical

    canonical = resolve_name_with_chembl_search(name)

    if canonical:
        return canonical

    return None


def resolve_molecule(reference_molecule=None, reference_smiles=None, debug=False):
    # 1. Direct SMILES has priority
    if reference_smiles:
        canonical = canonicalize_smiles(reference_smiles)

        if canonical:
            return {
                "status": "success",
                "source": "input_smiles",
                "canonical_smiles": canonical,
                "original_name": reference_molecule,
                "message": None,
            }

    # 2. Resolve molecule name using PubChem first, then ChEMBL
    if reference_molecule:
        for variant in generate_name_variants(reference_molecule):
            if debug:
                print(f"Trying PubChem with: {variant}")

            canonical = resolve_name_with_pubchem(variant, debug=debug)

            if canonical:
                if debug:
                    print(f"Found in PubChem with: {variant}")

                return {
                    "status": "success",
                    "source": "pubchem_name",
                    "canonical_smiles": canonical,
                    "original_name": reference_molecule,
                    "resolved_query": variant,
                    "message": None,
                }

        for variant in generate_name_variants(reference_molecule):
            if debug:
                print(f"Trying ChEMBL with: {variant}")

            canonical = resolve_name_with_chembl(variant)

            if canonical:
                if debug:
                    print(f"Found in ChEMBL with: {variant}")

                return {
                    "status": "success",
                    "source": "chembl_name",
                    "canonical_smiles": canonical,
                    "original_name": reference_molecule,
                    "resolved_query": variant,
                    "message": None,
                }

    return {
        "status": "not_found",
        "source": None,
        "canonical_smiles": None,
        "original_name": reference_molecule,
        "message": (
            "The molecule name could not be resolved using PubChem or ChEMBL. "
            "Please provide the request again using a valid SMILES string."
        ),
    }


if __name__ == "__main__":
    tests = [
        {"reference_molecule": "BRACO-19", "reference_smiles": None},
        {"reference_molecule": "Braco-19", "reference_smiles": None},
        {"reference_molecule": "BRACO19", "reference_smiles": None},
        {"reference_molecule": "pyridostatin", "reference_smiles": None},
        {"reference_molecule": None, "reference_smiles": "CCO"},
        {"reference_molecule": "notarealmolecule12345", "reference_smiles": None},
    ]

    for test in tests:
        print("\n" + "=" * 80)
        print(test)

        result = resolve_molecule(
            reference_molecule=test["reference_molecule"],
            reference_smiles=test["reference_smiles"],
            debug=True
        )

        print(result)
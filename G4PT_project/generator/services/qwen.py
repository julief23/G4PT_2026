import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from rdkit import Chem

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)


# ============================================================
# 1. SMILES utilities
# ============================================================

def canonicalize_smiles(text):
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def extract_reference_smiles(user_prompt):
    quoted_items = re.findall(r'"([^"]+)"|\'([^\']+)\'', user_prompt)
    quoted_items = [item[0] or item[1] for item in quoted_items]

    for item in quoted_items:
        canonical = canonicalize_smiles(item)
        if canonical is not None:
            cleaned_prompt = user_prompt.replace(item, "[REFERENCE_SMILES]")
            return canonical, cleaned_prompt

    return None, user_prompt


def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in model response:\n{text}")
    return json.loads(match.group(0))


# ============================================================
# 2. Intent detection
# ============================================================

def is_generation_request(user_prompt):
    prompt = user_prompt.lower()

    generation_keywords = [
        "generate molecule",
        "generate molecules",
        "generate candidate",
        "generate candidates",
        "design molecule",
        "design molecules",
        "create molecule",
        "create molecules",
        "propose molecule",
        "propose molecules",
        "scaffold hopping",
        "scaffold variants",
        "scaffold variant",
        "new scaffolds",
        "analogs of",
        "analogues of",
        "similar to",
        "based on",
    ]

    return any(keyword in prompt for keyword in generation_keywords)


# ============================================================
# 3. JSON normalization
# ============================================================

def normalize_g4pt_result(result):
    defaults = {
        "reference_molecule": None,
        "reference_smiles": None,
        "mode": "unconditional",
        "num_candidates": 5,
        "lower_molecular_weight": False,
        "improve_sa_score": False,
        "high_agape_confidence": False,
        "agape_threshold": 0.6,
        "max_imputed_features": 8
    }

    normalized = defaults.copy()
    normalized.update(result)

    return normalized


# ============================================================
# 4. Strict JSON parser for molecular generation
# ============================================================

def parse_g4pt_prompt(user_prompt):
    detected_smiles, cleaned_prompt = extract_reference_smiles(user_prompt)

    system_prompt = """
You are a strict JSON parser for the molecular generation platform G4PT.

Your only task is to convert the user's generation request into valid JSON.

Return JSON only.
Do not explain anything.
Do not use markdown.
Do not output any text outside the JSON object.

Allowed JSON keys:
- reference_molecule
- reference_smiles
- mode
- num_candidates
- lower_molecular_weight
- improve_sa_score
- high_agape_confidence
- agape_threshold
- max_mw
- max_logp
- max_tpsa
- max_rotatable_bonds
- max_sa_score

Generation modes:

- DE_NOVO
  Generate completely new G4-like molecules without any reference.

- ANALOGUE
  Generate molecules similar to a reference ligand.

- EXTEND
  Extend or grow a provided scaffold.

- HOP
  Perform scaffold hopping by generating molecules with alternative cores while maintaining similar chemical characteristics.

Rules:

- If the user provides a SMILES string, set reference_smiles.
- Otherwise set reference_molecule.

- If both are absent:
    mode = "DE_NOVO"

- Expressions such as:
    similar
    analogue
    analog
    based on
    close to
    inspired by

    -> mode = "ANALOGUE"

- Expressions such as:
    extend
    grow
    complete scaffold
    elaborate scaffold

    -> mode = "EXTEND"

- Expressions such as:
    scaffold hopping
    scaffold hop
    scaffold variants
    new scaffold
    different scaffold
    alternative scaffold
    new core
    different core

    -> mode = "HOP"

- Words such as
    diverse
    varied
    explore
    novel

DO NOT change the generation mode.
They simply describe the desired diversity.

- If the user requests:
    smaller
    lighter
    lower molecular weight

    lower_molecular_weight = true

- If the user requests:
    easier to synthesize
    better SA
    higher SA
    improved synthetic accessibility
    simpler

    improve_sa_score = true

- If the user requests:
    high AGAPE confidence
    highly confident
    confident G4 stabilizers

    high_agape_confidence = true

Defaults:

num_candidates = 20
agape_threshold = 0.6

max_mw = 900
max_logp = 10
max_tpsa = 300
max_rotatable_bonds = 25
max_sa_score = 8

If a field is not mentioned, use the default value or false.

Return valid JSON only.

Example 1

User:
Generate 5 molecules similar to pyridostatin but smaller and easier to synthesize.

{
  "reference_molecule":"pyridostatin",
  "reference_smiles":null,
  "mode":"ANALOGUE",
  "num_candidates":5,
  "lower_molecular_weight":true,
  "improve_sa_score":true,
  "high_agape_confidence":false,
  "agape_threshold":0.6,
  "max_mw":900,
  "max_logp":10,
  "max_tpsa":300,
  "max_rotatable_bonds":25,
  "max_sa_score":8
}

Example 2

User:
Generate scaffold hopping candidates based on BRACO-19.

{
  "reference_molecule":"BRACO-19",
  "reference_smiles":null,
  "mode":"HOP",
  "num_candidates":20,
  "lower_molecular_weight":false,
  "improve_sa_score":false,
  "high_agape_confidence":false,
  "agape_threshold":0.6,
  "max_mw":900,
  "max_logp":10,
  "max_tpsa":300,
  "max_rotatable_bonds":25,
  "max_sa_score":8
}

Example 3

User:
Extend the scaffold "c1ccc2ccccc2c1".

{
  "reference_molecule":null,
  "reference_smiles":"c1ccc2ccccc2c1",
  "mode":"EXTEND",
  "num_candidates":20,
  "lower_molecular_weight":false,
  "improve_sa_score":false,
  "high_agape_confidence":false,
  "agape_threshold":0.6,
  "max_mw":900,
  "max_logp":10,
  "max_tpsa":300,
  "max_rotatable_bonds":25,
  "max_sa_score":8
}

Example 4

User:
Generate 100 novel G4 ligands with high AGAPE confidence.

{
  "reference_molecule":null,
  "reference_smiles":null,
  "mode":"DE_NOVO",
  "num_candidates":100,
  "lower_molecular_weight":false,
  "improve_sa_score":false,
  "high_agape_confidence":true,
  "agape_threshold":0.6,
  "max_mw":900,
  "max_logp":10,
  "max_tpsa":300,
  "max_rotatable_bonds":25,
  "max_sa_score":8
}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": cleaned_prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        do_sample=False
    )

    generated = outputs[0][inputs.input_ids.shape[-1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)

    result = extract_json(response)

    if detected_smiles is not None:
        result["reference_smiles"] = detected_smiles
        result["reference_molecule"] = None

    return normalize_g4pt_result(result)


# ============================================================
# 5. Conversational assistant mode
# ============================================================

def chat_with_g4pt(user_prompt):
    system_prompt = """
You are G4PT, a helpful scientific assistant specialized in:
- G-quadruplex ligands
- molecular generation
- AGAPE predictions
- cheminformatics
- molecular descriptors
- drug discovery
- generative AI for molecules
- synthetic accessibility is the SA score
- retrosynthetic accessibility is the RA score
- AGAPE confidence is a measure of how likely a molecule is to be a G4 stabilizer based on a machine learning model.
- AGAPE is used internally to prioritize generated molecular candidates.

You answer normally and conversationally.
- Be concise but informative.
- Avoid jargon when possible, but you can use technical terms if they are relevant and explained.

Important rule:
Only output JSON when the user explicitly asks to generate, design, create,
or propose molecules or molecular candidates.
Otherwise, explain in natural language.

Allowed modes:
- DE_NOVO: generate new G4-like molecules without a reference.
- ANALOGUE: generate molecules similar to a reference ligand.
- EXTEND: grow or complete a provided scaffold.
- HOP: scaffold hopping, meaning generating candidates with alternative molecular cores.

Example 1:
User: Generate 5 molecules similar to pyridostatin but smaller and easier to synthesize.
JSON:
{
  "reference_molecule": "pyridostatin",
  "reference_smiles": null,
  "mode": "ANALOGUE",
  "num_candidates": 5,
  "lower_molecular_weight": true,
  "improve_sa_score": true,
  "high_agape_confidence": false,
  "agape_threshold": 0.6,
  "max_mw": 900,
  "max_logp": 10,
  "max_tpsa": 300,
  "max_rotatable_bonds": 25,
  "max_sa_score": 8
}

Example 2:
User: Generate molecules similar to "CCO" but simpler.
JSON:
{
  "reference_molecule": null,
  "reference_smiles": "CCO",
  "mode": "ANALOGUE",
  "num_candidates": 20,
  "lower_molecular_weight": false,
  "improve_sa_score": true,
  "high_agape_confidence": false,
  "agape_threshold": 0.6,
  "max_mw": 900,
  "max_logp": 10,
  "max_tpsa": 300,
  "max_rotatable_bonds": 25,
  "max_sa_score": 8
}

Example 3:
User: Generate scaffold hopping candidates based on BRACO-19.
JSON:
{
  "reference_molecule": "BRACO-19",
  "reference_smiles": null,
  "mode": "HOP",
  "num_candidates": 20,
  "lower_molecular_weight": false,
  "improve_sa_score": false,
  "high_agape_confidence": false,
  "agape_threshold": 0.6,
  "max_mw": 900,
  "max_logp": 10,
  "max_tpsa": 300,
  "max_rotatable_bonds": 25,
  "max_sa_score": 8
}

Example 4:
User: Generate scaffold variants similar to "CC[N+](CC)(CC)Cc1ccc(O)c(/C=N\\c2ccccc2/N=C\\c2cc(Cl)cc(Cl)c2O)c1.[Cu]" but easier to synthesize.
JSON:
{
  "reference_molecule": null,
  "reference_smiles": "[REFERENCE_SMILES]",
  "mode": "HOP",
  "num_candidates": 20,
  "lower_molecular_weight": false,
  "improve_sa_score": true,
  "high_agape_confidence": false,
  "agape_threshold": 0.6,
  "max_mw": 900,
  "max_logp": 10,
  "max_tpsa": 300,
  "max_rotatable_bonds": 25,
  "max_sa_score": 8
}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=False
    )

    generated = outputs[0][inputs.input_ids.shape[-1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)

    return response.strip()


# ============================================================
# 6. Main router
# ============================================================

def respond_to_user(user_prompt):
    if is_generation_request(user_prompt):
        result = parse_g4pt_prompt(user_prompt)
        return json.dumps(result, indent=2)

    return chat_with_g4pt(user_prompt)


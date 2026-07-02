from pathlib import Path
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_model" / "molgpt_final_g4_multimode"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_GENERATION_SETTINGS = {
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.95,
    "repetition_penalty": 1.05,
    "max_new_tokens": 120,
    "batch_size": 50,
}


def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(str(smiles).strip())

    if mol is None:
        return None

    return Chem.MolToSmiles(mol, canonical=True)


def clean_generated_text(text):
    text = text.strip()

    if ">>" in text:
        text = text.split(">>")[-1]

    parts = text.split()

    if not parts:
        return None

    return parts[0].strip()


def build_prompt(mode, reference_smiles=None):
    mode = mode.upper()

    if mode == "DE_NOVO":
        return "<MODE=DE_NOVO> >>"

    if reference_smiles is None:
        raise ValueError(f"Mode {mode} requires reference_smiles.")

    if mode == "ANALOGUE":
        return f"<MODE=ANALOGUE> {reference_smiles} >>"

    if mode == "EXTEND":
        return f"<MODE=EXTEND> {reference_smiles} >>"

    if mode == "HOP":
        return f"<MODE=HOP> {reference_smiles} >>"

    raise ValueError(f"Unknown generation mode: {mode}")


class MolGPTGenerator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(DEVICE)
        self.model.eval()

    def generate_molecules(
        self,
        mode,
        reference_smiles=None,
        num_candidates=100,
        settings=None,
    ):
        settings = settings or {}

        generation_settings = DEFAULT_GENERATION_SETTINGS.copy()
        generation_settings.update(settings)

        prompt = build_prompt(mode, reference_smiles)

        generated_smiles = []
        invalid_count = 0
        generated_total = 0

        while generated_total < num_candidates:
            current_batch = min(
                generation_settings["batch_size"],
                num_candidates - generated_total,
            )

            inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
            input_len = inputs.input_ids.shape[-1]

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    num_return_sequences=current_batch,
                    max_new_tokens=generation_settings["max_new_tokens"],
                    do_sample=True,
                    temperature=generation_settings["temperature"],
                    top_k=generation_settings["top_k"],
                    top_p=generation_settings["top_p"],
                    repetition_penalty=generation_settings["repetition_penalty"],
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            for output in outputs:
                generated_tokens = output[input_len:]
                generated_text = self.tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True,
                )

                raw_smiles = clean_generated_text(generated_text)

                if raw_smiles is None:
                    invalid_count += 1
                    continue

                canonical = canonicalize_smiles(raw_smiles)

                if canonical is None:
                    invalid_count += 1
                    continue

                generated_smiles.append(canonical)

            generated_total += current_batch

        unique_smiles = sorted(set(generated_smiles))

        return {
            "mode": mode,
            "prompt": prompt,
            "requested": num_candidates,
            "valid": len(generated_smiles),
            "invalid": invalid_count,
            "unique": len(unique_smiles),
            "smiles": unique_smiles,
        }


molgpt_generator = MolGPTGenerator()
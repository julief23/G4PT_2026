from django.views.generic import TemplateView
from django.shortcuts import render

from django.utils.decorators import method_decorator
from django.views.decorators.clickjacking import xframe_options_sameorigin
from django.contrib import messages
from django.http import HttpResponse
from django.http import JsonResponse

from generator.services.mordred_engine import compute_mordred_from_smiles_list
from .forms import SimulationForm
from .services.csv_utils import robust_csv_reader
from .services.smiles_utils import canonicalize_smiles
from .services.agape_predictor import AGAPEPredictor

from rdkit import Chem
from rdkit.Chem import Draw

import csv
import base64
import io
import logging

HIGH_CONFIDENCE_THRESHOLD = 0.85
MODERATE_CONFIDENCE_THRESHOLD = 0.65
MAX_CSV_ROWS = 151

MODEL_DNN = "DNN"

INPUT_CSV = "csv"
INPUT_SMILES = "smiles"
INPUT_DRAW = "draw"

PRED_ACTIVE = "ACTIVE"
PRED_INACTIVE = "INACTIVE"
PRED_INVALID = "INVALID"

CONF_HIGH = "High"
CONF_MODERATE = "Moderate"
CONF_LOW = "Low"

logger = logging.getLogger(__name__)

predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        predictor = AGAPEPredictor()
    return predictor

class GeneratorView(TemplateView):
    template_name = "generator/generator.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["form"] = GenerationForm()
        context["active_page"] = "generator"
        return context


class ResultsView(TemplateView):
    template_name = "generator/results.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["active_page"] = "results"
        return context


class ContactView(TemplateView):
    template_name = "generator/contact_us.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["active_page"] = "contact_us"
        return context


class FAQView(TemplateView):
    template_name = "generator/faq.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["active_page"] = "FAQ"
        return context

class DisclaimerGuideView(TemplateView):
    template_name = "generator/disclaimer.html"

class UserGuideView(TemplateView):
    template_name = "generator/user_guide.html"

@method_decorator(xframe_options_sameorigin, name="dispatch")
class JSMEView(TemplateView):
    template_name = "generator/jsme_embed.html"




def run_simulation(request):
    return JsonResponse({"status": "temporarily disabled"})


def preview_smiles(request):
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "POST method required."}, status=405)

    smiles = (request.POST.get(INPUT_SMILES) or "").strip()

    if not smiles:
        return JsonResponse({"ok": False, "error": "No SMILES provided."}, status=400)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return JsonResponse({"ok": False, "error": "Invalid SMILES."}, status=400)

    try:
        img = Draw.MolToImage(mol, size=(450, 300))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return JsonResponse({
            "ok": True,
            "image": f"data:image/png;base64,{image_b64}",
            "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
        })

    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=500)




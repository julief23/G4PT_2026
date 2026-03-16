from django.views.generic import TemplateView
from django.utils.decorators import method_decorator
from django.views.decorators.clickjacking import xframe_options_sameorigin
from django.conf import settings
from simulator.services.mordred_engine import compute_mordred_from_smiles_list
from django.contrib import messages
from django.http import FileResponse, HttpResponse
from django.shortcuts import render, redirect
import pandas as pd
import csv
import os
import io
import mimetypes
from .forms import SimulationForm
from .services.csv_utils import robust_csv_reader
from .services.smiles_utils import canonicalize_smiles
from .services.predictor import AGAPEPredictor
from django.http import JsonResponse
from rdkit import Chem
from rdkit.Chem import Draw
import base64
import io


predictor = AGAPEPredictor()

class SimulatorView(TemplateView):
    template_name = "simulator/simulator.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["form"] = SimulationForm()
        context["active_page"] = "simulator"
        return context


class ResultsView(TemplateView):
    template_name = "simulator/results.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["active_page"] = "results"
        return context


class ContactView(TemplateView):
    template_name = "simulator/contact_us.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["active_page"] = "contact_us"
        return context


class FAQView(TemplateView):
    template_name = "simulator/faq.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["active_page"] = "FAQ"
        return context

class DisclaimerGuideView(TemplateView):
    template_name = "simulator/disclaimer.html"

class UserGuideView(TemplateView):
    template_name = "simulator/user_guide.html"

@method_decorator(xframe_options_sameorigin, name="dispatch")
class JSMEView(TemplateView):
    template_name = "simulator/jsme_embed.html"


def preview_smiles(request):
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "POST method required."}, status=405)

    smiles = (request.POST.get("smiles") or "").strip()

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

def run_simulation(request):

    if request.method == "POST":
        form = SimulationForm(request.POST, request.FILES)

        if form.is_valid():

            job_name = form.cleaned_data["job_name"]
            model_type = form.cleaned_data["modelType"]  # DNN or ML
            input_mode = form.cleaned_data["classicalInputType"]

            try:
                smiles_list = []

                # ------------------------------------------
                # 1️⃣ EXTRACT INPUT
                # ------------------------------------------

                if input_mode == "smiles":
                    smi = form.cleaned_data.get("smiles")
                    smiles_list = [smi.strip()]

                elif input_mode == "draw":
                    smi = form.cleaned_data.get("draw")
                    smiles_list = [smi.strip()]

                elif input_mode == "csv":

                    csv_file = form.cleaned_data.get("classicalCsv")

                    df = robust_csv_reader(csv_file)

                    MAX_ROWS = 151

                    if len(df) > MAX_ROWS:
                        messages.error(
                            request,
                            f"CSV too large: {len(df)} rows detected. Maximum allowed is {MAX_ROWS}."
                        )
                        return render(request, "simulator/simulator.html", {"form": form})

                    cols_lower = [c.lower().strip() for c in df.columns]

                    if "smiles" not in cols_lower:
                        raise ValueError("CSV must contain a 'SMILES' column.")

                    smiles_col = df.columns[cols_lower.index("smiles")]

                    smiles_list = df[smiles_col].astype(str).tolist()

                else:
                    raise ValueError("Invalid input mode.")

                if not smiles_list:
                    raise ValueError("No SMILES provided.")

                # ------------------------------------------
                # 2️⃣ CANONICALIZATION
                # ------------------------------------------

                canonical_smiles = []
                validity_flags = []
                invalid_smiles = []

                for smi in smiles_list:

                    can = canonicalize_smiles(smi)

                    if can is None:
                        canonical_smiles.append(None)
                        validity_flags.append(False)
                        invalid_smiles.append(smi)
                    else:
                        canonical_smiles.append(can)
                        validity_flags.append(True)

                # ------------------------------------------
                # HARD STOP FOR SINGLE MOLECULE MODE
                # ------------------------------------------

                if input_mode in ["smiles", "draw"]:

                    if not validity_flags[0]:
                        messages.error(
                            request,
                            f"Invalid SMILES structure: {smiles_list[0]}"
                        )

                        return render(request, "simulator/simulator.html", {
                            "form": form
                        })

                # ------------------------------------------
                # 3️⃣ HANDLE BATCH LOGIC
                # ------------------------------------------

                results = []

                valid_smiles_only = [
                    smi for smi, valid in zip(canonical_smiles, validity_flags)
                    if valid
                ]

                descriptor_df = None
                use_precomputed = False

                if valid_smiles_only:

                    if input_mode == "csv":

                        if model_type.upper() == "DNN":
                            required_features = predictor.dnn_features
                        else:
                            required_features = predictor.xgb_features

                        if all(f in df.columns for f in required_features):
                            use_precomputed = True

                    if use_precomputed:

                        descriptor_df = df.loc[
                            [i for i, v in enumerate(validity_flags) if v]
                        ].reset_index(drop=True)

                    else:

                        descriptor_df = compute_mordred_from_smiles_list(
                            valid_smiles_only
                        )

                    preds, probs, imputation_percent, per_row_impute = predictor.predict(
                        descriptor_df,
                        model_type
                    )

                    if imputation_percent > 0:
                        messages.warning(
                            request,
                            f"{round(imputation_percent,2)}% of descriptor values "
                            "were missing or invalid and replaced during preprocessing."
                        )

                else:

                    preds = []
                    probs = []
                    imputation_percent = 0

                # ------------------------------------------
                # 4️⃣ REBUILD RESULTS IN ORIGINAL ORDER
                # ------------------------------------------

                valid_index = 0

                for original_smi, is_valid in zip(smiles_list, validity_flags):

                    if not is_valid:

                        results.append({
                            "smiles": original_smi,
                            "prediction": "INVALID",
                            "probability": None,
                            "confidence": None,
                            "imputed_features": None,
                        })

                    else:

                        prob_active = float(probs[valid_index])
                        pred = preds[valid_index]

                        model_confidence = max(prob_active, 1 - prob_active)

                        if model_confidence > 0.85:
                            confidence = "High"
                        elif model_confidence > 0.65:
                            confidence = "Moderate"
                        else:
                            confidence = "Low"

                        results.append({
                            "smiles": original_smi,
                            "prediction": "ACTIVE" if pred == 1 else "INACTIVE",
                            "probability_active": round(prob_active, 4),
                            "model_confidence": round(model_confidence, 4),
                            "confidence_level": confidence,
                            "imputed_features": per_row_impute[valid_index],
                        })

                        valid_index += 1

                # ------------------------------------------
                # 5️⃣ WARN ABOUT INVALID SMILES
                # ------------------------------------------

                if input_mode == "csv" and invalid_smiles:

                    messages.warning(
                        request,
                        f"{len(invalid_smiles)} invalid SMILES detected and skipped."
                    )

                # ------------------------------------------
                # 6️⃣ SUCCESS MESSAGE
                # ------------------------------------------

                if len(valid_smiles_only) > 0:

                    messages.success(
                        request,
                        f"Prediction completed successfully for {len(valid_smiles_only)} molecule(s)."
                    )

                else:

                    messages.error(
                        request,
                        "No valid SMILES found. Prediction was not performed."
                    )

                    return render(request, "simulator/simulator.html", {
                        "form": form
                    })

                request.session["prediction_results"] = results
                request.session["job_name"] = job_name

                return render(request, "simulator/results.html", {
                    "results": results,
                    "job_name": job_name,
                    "model_used": model_type,
                    "descriptors_used": (
                        "Precomputed (CSV)"
                        if use_precomputed else
                        "Computed via Mordred"
                    ),
                    "imputation_percent": round(imputation_percent, 2),
                })

            except Exception as e:

                messages.error(request, f"Simulation error: {str(e)}")

                return render(request, "simulator/simulator.html", {
                    "form": form
                })

    else:
        form = SimulationForm()

    return render(request, "simulator/simulator.html", {
        "form": form
    })

def download_prediction_csv(request):

    results = request.session.get("prediction_results")
    job_name = request.session.get("job_name", "agape_results")

    # Create CSV in memory
    output = io.StringIO()
    if not results:
        return HttpResponse("No prediction results available.", status=400)

    fieldnames = results[0].keys()

    writer = csv.DictWriter(output, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(results)

    response = HttpResponse(
        output.getvalue(),
        content_type="text/csv"
    )

    response["Content-Disposition"] = (
        f'attachment; filename="{job_name}_predictions.csv"'
    )

    return response


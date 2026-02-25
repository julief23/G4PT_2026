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
from .services.smiles_utils import canonicalize_smiles
from .services.predictor import AGAPEPredictor

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


@method_decorator(xframe_options_sameorigin, name="dispatch")
class JSMEView(TemplateView):
    template_name = "simulator/jsme_embed.html"

def run_simulation(request):

    if request.method == "POST":
        form = SimulationForm(request.POST, request.FILES)

        if form.is_valid():

            job_name = form.cleaned_data["job_name"]
            model_type = form.cleaned_data["modelType"]
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
                    df = pd.read_csv(csv_file)

                    if "SMILES" not in df.columns:
                        raise ValueError("CSV must contain a 'SMILES' column.")

                    smiles_list = df["SMILES"].astype(str).tolist()

                else:
                    raise ValueError("Invalid input mode.")

                if not smiles_list:
                    raise ValueError("No SMILES provided.")

                # ------------------------------------------
                # 2️⃣ CANONICALIZATION
                # ------------------------------------------

                canonical_smiles = []
                validity_flags = []   # keeps track of validity
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
                # SINGLE MOLECULE MODE: BLOCK IF INVALID
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

                # Only compute descriptors for valid molecules
                valid_smiles_only = [
                    smi for smi, valid in zip(canonical_smiles, validity_flags)
                    if valid
                ]

                descriptor_df = None

                if valid_smiles_only:

                    # Check if CSV already has required 75 features
                    use_precomputed = False

                    if input_mode == "csv":
                        required_features = predictor.feature_names
                        if all(f in df.columns for f in required_features):
                            use_precomputed = True

                    if use_precomputed:
                        descriptor_df = df.loc[
                            [i for i, v in enumerate(validity_flags) if v]
                        ].reset_index(drop=True)
                        descriptor_df["SMILES"] = valid_smiles_only

                    else:
                        descriptor_df = compute_mordred_from_smiles_list(
                            valid_smiles_only
                        )

                    # Align features
                    descriptor_df = descriptor_df[predictor.feature_names]

                    preds, probs = predictor.predict(descriptor_df)

                else:
                    preds = []
                    probs = []

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
                        })
                    else:
                        prob_active = float(probs[valid_index])
                        pred = preds[valid_index]

                        # Compute confidence as probability of predicted class
                        model_confidence = max(prob_active, 1 - prob_active)

                        # Define confidence level based on distance from 0.5
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
                        })

                        valid_index += 1

                # ------------------------------------------
                # 5️⃣ CSV MODE WARNING IF INVALID FOUND
                # ------------------------------------------

                if input_mode == "csv" and invalid_smiles:
                    messages.warning(
                        request,
                        f"{len(invalid_smiles)} invalid SMILES detected and skipped."
                    )

                request.session["prediction_results"] = results
                request.session["job_name"] = job_name

                return render(request, "simulator/results.html", {
                    "results": results,
                    "job_name": job_name
                })

            except Exception as e:
                messages.error(request, str(e))
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

    if not results:
        return HttpResponse("No prediction results available.", status=400)

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


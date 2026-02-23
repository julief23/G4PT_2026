from django.views.generic import TemplateView
from django.utils.decorators import method_decorator
from django.views.decorators.clickjacking import xframe_options_sameorigin

from django.contrib import messages
from django.http import FileResponse, HttpResponse
from django.shortcuts import render, redirect
import pandas as pd
import csv
import os
import mimetypes

from config import settings
from simulator.services.mordred_engine import compute_mordred_from_smiles_list


class SimulatorView(TemplateView):
    template_name = "simulator/simulator.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
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




def robust_csv_reader(uploaded_file):
    """
    Reads CSV with automatic delimiter detection.
    Handles edge cases like single-column comma strings.
    """
    content = uploaded_file.read().decode("utf-8", errors="replace")
    uploaded_file.seek(0)

    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(content[:2048], delimiters=",;\t")
        delimiter = dialect.delimiter
    except Exception:
        delimiter = ","

    df = pd.read_csv(uploaded_file, delimiter=delimiter)

    if df.shape[1] == 1 and "," in str(df.iloc[0, 0]):
        header = df.iloc[0, 0].split(",")
        data = df.iloc[1, 0].split(",")
        df = pd.DataFrame([data], columns=header)

    df.columns = [c.strip() for c in df.columns]
    return df


def run_simulation(request):
    if request.method != "POST":
        return HttpResponse("Invalid request method.", status=405)

    job_name = request.POST.get("job_name")
    classical_input_type = request.POST.get("classicalInputType")
    smiles_text = request.POST.get("smiles")
    classical_file = request.FILES.get("classicalCsv")

    if not (job_name or "").strip():
        messages.error(request, "The Job Name field is required.")
        return redirect('/?section=simulation')

    # ---------------------------
    # HANDLE CLASSICAL INPUT
    # ---------------------------

    if classical_input_type == "csv":

        if not classical_file:
            messages.error(request, "You must upload a CSV file.")
            return redirect('/?section=simulation')

        try:
            df_uploaded = robust_csv_reader(classical_file)
        except Exception as e:
            messages.error(request, f"Error reading CSV: {str(e)}")
            return redirect('/?section=simulation')

        # If CSV contains SMILES column → compute Mordred
        if "SMILES" in df_uploaded.columns:
            try:
                df_final = compute_mordred_from_smiles_list(
                    df_uploaded["SMILES"].tolist()
                )
            except Exception as e:
                messages.error(request, f"Descriptor error: {str(e)}")
                return redirect('/?section=simulation')
        else:
            # Assume descriptors already provided
            df_final = df_uploaded

    elif classical_input_type == "smiles":

        if not (smiles_text or "").strip():
            messages.error(request, "You must provide a SMILES string.")
            return redirect('/?section=simulation')

        try:
            df_final = compute_mordred_from_smiles_list([smiles_text])
        except Exception as e:
            messages.error(request, f"Descriptor error: {str(e)}")
            return redirect('/?section=simulation')

    else:
        messages.error(request, "Invalid input type.")
        return redirect('/?section=simulation')

    if df_final.empty:
        messages.error(request, "No valid molecules found.")
        return redirect('/?section=simulation')

    # ---------------------------
    # SAVE OUTPUT FILE
    # ---------------------------

    output_dir = os.path.join(settings.BASE_DIR, "frontend", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    output_filename = f"{job_name}_descriptors.csv"
    output_path = os.path.join(output_dir, output_filename)

    df_final.to_csv(output_path, index=False)

    # Store results in session
    request.session["last_results"] = {
        "job_name": job_name,
        "n_molecules": len(df_final),
        "n_features": df_final.shape[1],
        "output_file": output_filename
    }

    messages.success(
        request,
        f"Descriptor generation completed successfully. "
        f"{len(df_final)} molecule(s) processed."
    )

    return redirect('/?section=results')


def download_prediction_csv(request, job_name):
    output_path = os.path.join(
        settings.BASE_DIR,
        "frontend",
        "outputs",
        f"{job_name}_descriptors.csv"
    )

    if not os.path.exists(output_path):
        return HttpResponse("File not found.", status=404)

    content_type, _ = mimetypes.guess_type(output_path)
    response = FileResponse(open(output_path, "rb"), content_type=content_type)
    response["Content-Disposition"] = (
        f'attachment; filename="{job_name}_descriptors.csv"'
    )
    return response



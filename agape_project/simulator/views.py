from django.views.generic import TemplateView
from django.utils.decorators import method_decorator
from django.views.decorators.clickjacking import xframe_options_sameorigin
from config import settings
from simulator.services.mordred_engine import compute_mordred_from_smiles_list
from django.contrib import messages
from django.http import FileResponse, HttpResponse
from django.shortcuts import render, redirect
import pandas as pd
import csv
import os
import mimetypes
from .forms import SimulationForm

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
            # Everything validated safely here
            job_name = form.cleaned_data["job_name"]
            model_type = form.cleaned_data["modelType"]
            input_mode = form.cleaned_data["classicalInputType"]

            # Later: run Mordred here

            return render(request, "simulator/simulator.html", {
                "form": form,
                "success": "Validation successful."
            })

    else:
        form = SimulationForm()

    return render(request, "simulator/simulator.html", {
        "form": form
    })

def download_prediction_csv(request, job_name):
    output_path = os.path.join(
        settings.BASE_DIR,
        "simulator",
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



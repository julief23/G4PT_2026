from django.views.generic import TemplateView
from django.shortcuts import redirect

class HomeView(TemplateView):
    template_name = "frontend/home.html"


def home_redirect(request):
    return redirect("home")
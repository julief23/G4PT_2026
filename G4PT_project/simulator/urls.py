from django.urls import path
from .views import (
    SimulatorView,
    ResultsView,
    ContactView,
    FAQView,
    JSMEView,
    UserGuideView,
    DisclaimerGuideView,
    preview_smiles,

)

urlpatterns = [
    path("", SimulatorView.as_view(), name="simulation"),
    path("results/", ResultsView.as_view(), name="results"),
    path("contact_us/", ContactView.as_view(), name="contact"),
    path("FAQ/", FAQView.as_view(), name="faq"),
    path("user-guide/", UserGuideView.as_view(), name="user_guide"),
    path("jsme/", JSMEView.as_view(), name="jsme"),
    path("run_simulation/", run_simulation, name="run_simulation"),
    path("disclaimer/", DisclaimerGuideView.as_view(), name="disclaimer"),    
    path("preview-smiles/", preview_smiles, name="preview_smiles"),
]
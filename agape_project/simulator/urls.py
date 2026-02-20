from django.urls import path
from .views import SimulatorView, ResultsView, ContactView, FAQView

app_name = "simulator"

urlpatterns = [
    path("", SimulatorView.as_view(), name="simulation"),
    path("results/", ResultsView.as_view(), name="results"),
    path("contact_us/", ContactView.as_view(), name="contact"),
    path("FAQ/", FAQView.as_view(), name="faq"),
]
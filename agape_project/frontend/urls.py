from django.urls import path
from .views import HomeView, home_redirect

app_name = "frontend"

urlpatterns = [
    path("", HomeView.as_view(), name="home"),
    path("home/", home_redirect),
]
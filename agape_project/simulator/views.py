from django.shortcuts import render

from django.shortcuts import render

def about(request):
    return render(request, "frontend/about.html")

def contact(request):
    return render(request, "frontend/contact.html")

def simulation(request):
    return render(request, "frontend/simulation.html")

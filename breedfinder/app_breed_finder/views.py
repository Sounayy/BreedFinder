from app_breed_finder.models import Breed
from django.shortcuts import render

# Create your views here.


def home(request):
    breeds = Breed.objects.all()
    return render(
        request,
        "app_breed_finder/home.html",
        {"breeds": breeds},
    )

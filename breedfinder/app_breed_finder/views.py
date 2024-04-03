from app_breed_finder.models import Breed
from django.shortcuts import render
from app_breed_finder.forms import UploadImageForm
import logging
from django.shortcuts import redirect
from django.urls import reverse
from app_breed_finder.utils.helpers import predicit_breed_from_image

logger = logging.getLogger("mylogger")


def home(request):
    return render(
        request,
        "app_breed_finder/home.html",
    )


def breed_identifier(request):

    breeds = Breed.objects.all()
    breeds = breeds[:5]

    breeds_predicted_id = request.session.get("breeds_predicted_id", None)
    if breeds_predicted_id is None:
        breeds_predicted = breeds
    else:
        breeds_predicted = [
            Breed.objects.get(id=breed_id) for breed_id in breeds_predicted_id
        ]

    first_breed = breeds_predicted[0]
    other_breeds = breeds_predicted[1:]

    if request.method == "POST":
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data["image_uploaded"]
            breeds_predicted_id = predicit_breed_from_image(image)
            request.session["breeds_predicted_id"] = breeds_predicted_id
            return redirect(reverse("breed-identifier"))
    else:
        form = UploadImageForm()

    return render(
        request,
        "app_breed_finder/breed-identifier.html",
        {
            "first_breed": first_breed,
            "other_breeds": other_breeds,
            "form": form,
        },
    )


def breedex(request):
    breeds = Breed.objects.all()
    selected_animal = request.GET.get("animal")
    if not selected_animal:
        selected_animal = "DOG"  # Par défaut, les chiens sont séléctionnés
    breeds = breeds.filter(animal=selected_animal)
    return render(
        request,
        "app_breed_finder/breedex.html",
        {
            "breeds": breeds,
            "selected_animal": selected_animal,
        },
    )


def breed_detail(request, breed_id):
    breed = Breed.objects.get(id=breed_id)
    return render(
        request,
        "app_breed_finder/breed-detail.html",
        {
            "breed": breed,
        },
    )

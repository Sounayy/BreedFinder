from app_breed_finder.models import Breed
from django.shortcuts import render
from app_breed_finder.forms import UploadImageForm
from django.shortcuts import redirect
from django.urls import reverse
from app_breed_finder.utils.helpers import predict_breed_from_image


def home(request):
    """Render the home page."""

    return render(
        request,
        "app_breed_finder/home.html",
    )


def breed_identifier(request):
    """Identify the breed of an uploaded image."""

    # Sends the default 4 breeds from the database
    breeds = Breed.objects.all()[:4]

    # Get the predicted breeds if the user made a prediction
    breeds_predicted_id = request.session.get("breeds_predicted_id", None)
    if breeds_predicted_id is None:
        breeds_predicted = breeds
    else:
        breeds_predicted = [
            Breed.objects.get(id=breed_id) for breed_id in breeds_predicted_id
        ]

    # Separate the first predicted breed from the other 3 breeds
    first_breed = breeds_predicted[0]
    other_breeds = breeds_predicted[1:]

    # Get the user's image and pass it as a parameter to the prediction function
    if request.method == "POST":
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data["image_uploaded"]
            breeds_predicted_id = predict_breed_from_image(image)
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
    """Display a list of breeds."""

    breeds = Breed.objects.all()

    selected_animal = request.GET.get("animal")
    if not selected_animal:
        # By default, dogs are selected
        selected_animal = "DOG"

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
    """Render the detail page for a specific breed."""

    breed = Breed.objects.get(id=breed_id)
    return render(
        request,
        "app_breed_finder/breed-detail.html",
        {
            "breed": breed,
        },
    )

from app_breed_finder.models import Breed
from django.shortcuts import render
from app_breed_finder.forms import UploadImageForm
import logging
from django.shortcuts import redirect
from django.urls import reverse
from app_breed_finder.utils.helpers import predicit_breed_from_image


def home(request):
    logger = logging.getLogger("mylogger")
    breeds = Breed.objects.all()
    breeds_predicted_id = request.session.get("breeds_predicted_id", None)
    if breeds_predicted_id is None:
        breeds_predicted = breeds
    else:
        breeds_predicted = [
            Breed.objects.get(id=breed_id) for breed_id in breeds_predicted_id
        ]
    if request.method == "POST":
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data["image_uploaded"]
            breeds_predicted_id = predicit_breed_from_image(image)
            request.session["breeds_predicted_id"] = breeds_predicted_id
            return redirect(reverse("home"))
    else:
        form = UploadImageForm()
    return render(
        request,
        "app_breed_finder/home.html",
        {
            "breeds": breeds,
            "form": form,
            "breeds_predicted": breeds_predicted,
        },
    )

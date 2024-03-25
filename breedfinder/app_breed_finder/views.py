from app_breed_finder.models import Breed
from django.shortcuts import render
from app_breed_finder.forms import UploadImageForm
import logging
from django.shortcuts import redirect
from django.urls import reverse


def home(request):
    logger = logging.getLogger("mylogger")
    breeds = Breed.objects.all()
    image = None
    if request.method == "POST":
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data["image_uploaded"]
            context_data = {
                "breeds": breeds,
                "form": form,
            }
            redirect_url = reverse("home")
            return redirect(redirect_url, context=context_data)
    else:
        form = UploadImageForm()
    context_data = {
        "breeds": breeds,
        "form": form,
    }
    return render(
        request,
        "app_breed_finder/home.html",
        context_data,
    )

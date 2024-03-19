from django.http import HttpResponse
from app_breed_finder.models import Breed

# Create your views here.


def test(request):
    breeds = Breed.objects.all()
    return HttpResponse(
        f"""
        <h1> BreedFinder </h1>
        <h2> Mes races de chats préférées </h2>
        <ul>
            <li>{breeds[0].name}</li>
            <li>{breeds[1].name}</li>
        </ul>
        """
    )

from app_breed_finder.models import Breed


def predicit_breed_from_image(image):

    results = ["British Shorthair", "Bedlington", "British Longhair", "Chartreux"]
    breeds_predicited_id = []
    for result in results:
        breeds_predicited_id.append(Breed.objects.get(name=result).id)
    return breeds_predicited_id

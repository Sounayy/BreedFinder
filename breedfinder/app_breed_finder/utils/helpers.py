from app_breed_finder.models import Breed
import cv2
import numpy as np
from app_breed_finder.utils.prediction import predict_image

MODEL_CAT_DOG = "C:/Users/egeke/Documents/ProjetBreedFinder/breedfinder/app_breed_finder/utils/VGG16_Cat_vs_Dog.h5"
WEIGHTS_DOG = "C:/Users/egeke/Documents/ProjetBreedFinder/breedfinder/app_breed_finder/utils/EFNB7_weights_Dog.h5"
WEIGHTS_CAT = "C:/Users/egeke/Documents/ProjetBreedFinder/breedfinder/app_breed_finder/utils/EFNB7_weights_Cat.h5"


def predict_breed_from_image(image):
    """Predict the breed of an image."""
    image = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    results = predict_image(
        image=image,
        model_cat_vs_dog=MODEL_CAT_DOG,
        weights_dog=WEIGHTS_DOG,
        weights_cat=WEIGHTS_CAT,
    )

    breeds_predicted_id = []
    for result in results:
        breeds_predicted_id.append(Breed.objects.get(name=result).id)
    return breeds_predicted_id

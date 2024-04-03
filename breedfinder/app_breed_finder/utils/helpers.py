from app_breed_finder.models import Breed
import logging
import cv2
import numpy as np
from app_breed_finder.utils.prediction import predict_image

logger = logging.getLogger("mylogger")
model_path = "C:/Users/egeke/Documents/ProjetBreedFinder/breedfinder/app_breed_finder/utils/classfication_model_dataset_Cat_vs_Dog_30.h5"
weights_dog_path = "C:/Users/egeke/Documents/ProjetBreedFinder/breedfinder/app_breed_finder/utils/weihgts_model_dog.h5"
test = 0


def predicit_breed_from_image(image):
    image_data = image.read()
    nparr = np.frombuffer(image_data)

    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = predict_image(
        image=image,
        model_cat_vs_dog=model_path,
        weights_dog=weights_dog_path,
        weights_cat=test,
    )
    # results = ["British Shorthair", "Bedlington", "British Longhair", "Chartreux"]
    breeds_predicited_id = []
    for result in results:
        breeds_predicited_id.append(Breed.objects.get(name=result).id)
        logger.info(result)
    return breeds_predicited_id

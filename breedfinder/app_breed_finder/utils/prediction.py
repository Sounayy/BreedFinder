import numpy as np
import keras
from keras import layers
from keras.applications import EfficientNetB7
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2


def predict_image(image, model_cat_vs_dog, weights_dog, weights_cat):
    """
    Function to predict whether the given image is a cat or a dog, and if so, predict the breed.

    Args:
    image (numpy.ndarray): The input image.
    model_cat_vs_dog (str): Path to the model that predicts cat vs. dog.
    weights_dog (str): Path to the weights of the dog breed classification model.
    weights_cat (str): Path to the weights of the cat breed classification model.

    Returns:
    list: List containing the predicted breed and similar breeds.
    """

    def build_model(num_classes):
        """
        Function to build the EfficientNet model for breed classification.

        Args:
        num_classes (int): Number of output classes.

        Returns:
        keras.Model: Compiled Keras model.
        """
        inputs = layers.Input(shape=(600, 600, 3))
        model = EfficientNetB7(
            include_top=False, input_tensor=inputs, weights="imagenet"
        )

        # Freeze the pretrained weights
        model.trainable = False

        # Rebuild top
        x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
        x = layers.BatchNormalization()(x)

        top_dropout_rate = 0.2
        x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

        # Compile
        model = keras.Model(inputs, outputs, name="EfficientNet")
        optimizer = keras.optimizers.Adam(learning_rate=1e-2)
        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )
        return model

    def pre_process(img, dim: int, normalize=False):
        """
        Function to preprocess the image.

        Args:
        img (numpy.ndarray): The input image.
        dim (int): Dimension to resize the image.
        normalize (bool): Whether to normalize the image.

        Returns:
        numpy.ndarray: Preprocessed image.
        """
        img = cv2.resize(img, (dim, dim))
        img = np.array(img)
        if normalize:
            img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    # List of dog breeds
    dog_breeds = {
        # Dictionary of dog breeds and their similar breeds
        # Key: Breed Name
        # Value: List of similar breeds
        "Chihuahua": ["Papillon", "Toy Terrier", "Pomeranian"],
        "Japanese Spaniel": ["Maltese Dog", "Pekinese", "Shih-Tzu"],
        "Maltese Dog": ["Japanese Spaniel", "Pekinese", "Shih-Tzu"],
        "Pekinese": ["Japanese Spaniel", "Maltese Dog", "Shih-Tzu"],
        "Shih-Tzu": ["Japanese Spaniel", "Maltese Dog", "Pekinese"],
        "Blenheim Spaniel": [
            "English Foxhound",
            "Papillon",
            "English Springer",
        ],
        "Papillon": ["Chihuahua", "Toy Terrier", "Pomeranian"],
        "Toy Terrier": ["Chihuahua", "Papillon", "Pomeranian"],
        "Rhodesian Ridgeback": ["Doberman", "Redbone", "Vizsla"],
        "Afghan Hound": ["Saluki", "Borzoi", "Irish Wolfhound"],
        "Basset": ["Bloodhound", "Bluetick", "Beagle"],
        "Beagle": ["Basset", "Bluetick", "Bloodhound"],
        "Bloodhound": ["Basset", "Beagle", "Bluetick"],
        "Bluetick": ["Basset", "Beagle", "Bloodhound"],
        "Black and tan Coonhound": ["Bluetick", "Redbone", "Bloodhound"],
        "Walker Hound": ["Redbone", "Bluetick", "Black and tan Coonhound"],
        "English Foxhound": ["Redbone", "Bluetick", "Basset"],
        "Redbone": ["Walker Hound", "Bluetick", "Black and tan Coonhound"],
        "Borzoi": ["Afghan Hound", "Saluki", "Irish Wolfhound"],
        "Irish Wolfhound": ["Afghan Hound", "Borzoi", "Saluki"],
        "Italian Greyhound": ["Whippet", "Ibizan Hound", "Scottish Deerhound"],
        "Whippet": ["Italian Greyhound", "Ibizan Hound", "Scottish Deerhound"],
        "Ibizan Hound": ["Italian Greyhound", "Whippet", "Scottish Deerhound"],
        "Norwegian Elkhound": ["Otterhound", "Saluki", "Scottish Deerhound"],
        "Otterhound": ["Norwegian Elkhound", "Saluki", "Scottish Deerhound"],
        "Saluki": ["Norwegian Elkhound", "Otterhound", "Scottish Deerhound"],
        "Scottish Deerhound": ["Italian Greyhound", "Whippet", "Ibizan Hound"],
        "Weimaraner": ["Vizsla", "German Short haired Pointer", "Rhodesian Ridgeback"],
        "Staffordshire Bullterrier": [
            "American Staffordshire Terrier",
            "Bull Mastiff",
            "Boxer",
        ],
        "American Staffordshire Terrier": [
            "Staffordshire Bullterrier",
            "Bull Mastiff",
            "Boxer",
        ],
        "Bedlington Terrier": ["Kerry Blue Terrier", "Airedale", "Border Terrier"],
        "Border Terrier": ["Kerry Blue Terrier", "Airedale", "Bedlington Terrier"],
        "Kerry Blue Terrier": ["Bedlington Terrier", "Airedale", "Border Terrier"],
        "Irish Terrier": ["Kerry Blue Terrier", "Airedale", "Bedlington Terrier"],
        "Norfolk Terrier": ["Norwich Terrier", "Cairn", "Border Terrier"],
        "Norwich Terrier": ["Norfolk Terrier", "Cairn", "Border Terrier"],
        "Yorkshire Terrier": ["West Highland White Terrier", "Cairn", "Silky Terrier"],
        "Wire haired Fox Terrier": ["Lakeland Terrier", "Norwich Terrier", "Airedale"],
        "Lakeland Terrier": ["Wire haired Fox Terrier", "Norwich Terrier", "Airedale"],
        "Sealyham Terrier": ["Border Terrier", "Norfolk Terrier", "Yorkshire Terrier"],
        "Airedale": ["Bedlington Terrier", "Kerry Blue Terrier", "Border Terrier"],
        "Cairn": ["Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier"],
        "Australian Terrier": ["Silky Terrier", "Cairn", "Yorkshire Terrier"],
        "Dandie Dinmont": ["Skye Terrier", "Cairn", "Norwich Terrier"],
        "Boston Bull": [
            "Bull Mastiff",
            "Staffordshire Bullterrier",
            "American Staffordshire Terrier",
        ],
        "Miniature Schnauzer": [
            "Standard Schnauzer",
            "Giant Schnauzer",
            "Scotch Terrier",
        ],
        "Giant Schnauzer": [
            "Miniature Schnauzer",
            "Standard Schnauzer",
            "Scotch Terrier",
        ],
        "Standard Schnauzer": [
            "Miniature Schnauzer",
            "Giant Schnauzer",
            "Scotch Terrier",
        ],
        "Scotch Terrier": ["Yorkshire Terrier", "West Highland White Terrier", "Cairn"],
        "Tibetan Terrier": ["Lhasa", "Shih-Tzu", "Lakeland Terrier"],
        "Silky Terrier": ["Australian Terrier", "Yorkshire Terrier", "Cairn"],
        "Soft-coated Wheaten Terrier": [
            "Irish Terrier",
            "Airedale",
            "Bedlington Terrier",
        ],
        "West Highland White Terrier": ["Yorkshire Terrier", "Cairn", "Scotch Terrier"],
        "Lhasa": ["Tibetan Terrier", "Shih-Tzu", "Lakeland Terrier"],
        "Flat coated Retriever": [
            "Golden Retriever",
            "Chesapeake Bay Retriever",
            "Labrador Retriever",
        ],
        "Curly-coated Retriever": [
            "Chesapeake Bay Retriever",
            "Golden Retriever",
            "Labrador Retriever",
        ],
        "Golden Retriever": [
            "Flat coated Retriever",
            "Chesapeake Bay Retriever",
            "Labrador Retriever",
        ],
        "Labrador Retriever": [
            "Flat coated Retriever",
            "Chesapeake Bay Retriever",
            "Golden Retriever",
        ],
        "Chesapeake Bay Retriever": [
            "Flat coated Retriever",
            "Curly-coated Retriever",
            "Labrador Retriever",
        ],
        "German Short haired Pointer": ["Weimaraner", "Vizsla", "Rhodesian Ridgeback"],
        "Vizsla": ["Weimaraner", "German Short haired Pointer", "Rhodesian Ridgeback"],
        "English Setter": ["Irish Setter", "Gordon Setter", "Brittany Spaniel"],
        "Irish Setter": ["English Setter", "Gordon Setter", "Brittany Spaniel"],
        "Gordon Setter": ["English Setter", "Irish Setter", "Brittany Spaniel"],
        "Brittany Spaniel": ["English Setter", "Irish Setter", "Gordon Setter"],
        "Clumber": ["Sussex Spaniel", "Brittany Spaniel", "English Springer"],
        "English Springer": [
            "Welsh Springer Spaniel",
            "Cocker Spaniel",
            "Sussex Spaniel",
        ],
        "Welsh Springer Spaniel": [
            "English Springer",
            "Sussex Spaniel",
            "Cocker Spaniel",
        ],
        "Cocker Spaniel": [
            "English Springer",
            "Welsh Springer Spaniel",
            "Sussex Spaniel",
        ],
        "Sussex Spaniel": ["Clumber", "English Springer", "Welsh Springer Spaniel"],
        "Irish Water Spaniel": [
            "Sussex Spaniel",
            "English Springer",
            "Welsh Springer Spaniel",
        ],
        "Kuvasz": ["Great Pyrenees", "Samoyed", "Kelpie"],
        "Schipperke": ["Groenendael", "Malinois", "Briard"],
        "Groenendael": ["Schipperke", "Malinois", "Briard"],
        "Malinois": ["Schipperke", "Groenendael", "Briard"],
        "Briard": ["Schipperke", "Groenendael", "Malinois"],
        "Kelpie": ["Great Pyrenees", "Samoyed", "Kuvasz"],
        "Komondor": ["Kuvasz", "Great Pyrenees", "Samoyed"],
        "Old English Sheepdog": ["Irish Wolfhound", "Komondor", "Kuvasz"],
        "Shetland Sheepdog": ["Collie", "Border Collie", "Groenendael"],
        "Collie": ["Border Collie", "Shetland Sheepdog", "Groenendael"],
        "Border Collie": ["Collie", "Shetland Sheepdog", "Groenendael"],
        "Bouvier Des Flandres": ["Boxer", "Bull Mastiff", "Rottweiler"],
        "Rottweiler": ["Bouvier Des Flandres", "Boxer", "Bull Mastiff"],
        "German Shepherd": [
            "Belgian Sheepdog",
            "Dutch Shepherd",
            "Bouvier Des Flandres",
        ],
        "Doberman": ["Rottweiler", "Bouvier Des Flandres", "Boxer"],
        "Miniature Pinscher": ["Manchester Terrier", "Doberman", "Boston Bull"],
        "Greater Swiss Mountain Dog": [
            "Bernese Mountain Dog",
            "Appenzeller",
            "Entlebucher",
        ],
        "Bernese Mountain Dog": [
            "Greater Swiss Mountain Dog",
            "Appenzeller",
            "Entlebucher",
        ],
        "Appenzeller": [
            "Greater Swiss Mountain Dog",
            "Bernese Mountain Dog",
            "Entlebucher",
        ],
        "Entlebucher": [
            "Greater Swiss Mountain Dog",
            "Bernese Mountain Dog",
            "Appenzeller",
        ],
        "Boxer": ["Bouvier Des Flandres", "Rottweiler", "Doberman"],
        "Bull Mastiff": ["Bouvier Des Flandres", "Rottweiler", "Boxer"],
        "Tibetan Mastiff": [
            "Bernese Mountain Dog",
            "Greater Swiss Mountain Dog",
            "Boxer",
        ],
        "French Bulldog": ["Boston Bull", "Bulldog", "Staffordshire Bullterrier"],
        "Great Dane": ["Bull Mastiff", "Bouvier Des Flandres", "Rottweiler"],
        "Saint Bernard": [
            "Greater Swiss Mountain Dog",
            "Bernese Mountain Dog",
            "Newfoundland",
        ],
        "Eskimo Dog": ["Samoyed", "Great Pyrenees", "Malamute"],
        "Malamute": ["Samoyed", "Great Pyrenees", "Eskimo Dog"],
        "Siberian Husky": ["Samoyed", "Great Pyrenees", "Malamute"],
        "Affenpinscher": ["Brussels Griffon", "Pug", "Pekingese"],
        "Basenji": ["Whippet", "Pharaoh Hound", "Ibizan Hound"],
        "Pug": ["Affenpinscher", "Brussels Griffon", "Pekingese"],
        "Leonberg": ["Saint Bernard", "Newfoundland", "Great Pyrenees"],
        "Newfoundland": ["Saint Bernard", "Leonberg", "Great Pyrenees"],
        "Great Pyrenees": ["Saint Bernard", "Leonberg", "Newfoundland"],
        "Samoyed": ["Siberian Husky", "Malamute", "Eskimo Dog"],
        "Pomeranian": ["Chihuahua", "Papillon", "Toy Terrier"],
        "Chow": ["Keeshond", "Norwegian Elkhound", "Eskimo Dog"],
        "Keeshond": ["Chow", "Norwegian Elkhound", "Eskimo Dog"],
        "Brabancon Griffon": ["Brussels Griffon", "Affenpinscher", "Pug"],
        "Pembroke": ["Cardigan", "Shetland Sheepdog", "Welsh Springer Spaniel"],
        "Cardigan": ["Pembroke", "Shetland Sheepdog", "Welsh Springer Spaniel"],
        "Toy Poodle": ["Miniature Poodle", "Standard Poodle", "Miniature Pinscher"],
        "Miniature Poodle": ["Toy Poodle", "Standard Poodle", "Miniature Pinscher"],
        "Standard Poodle": ["Toy Poodle", "Miniature Poodle", "Miniature Pinscher"],
        "Mexican Hairless": ["Dingo", "Dhole", "African Hunting Dog"],
        "Dingo": ["Mexican Hairless", "Dhole", "African Hunting Dog"],
        "Dhole": ["Mexican Hairless", "Dingo", "African Hunting Dog"],
        "African Hunting Dog": ["Mexican Hairless", "Dingo", "Dhole"],
    }

    # List of cat breeds
    cat_breeds = {
        # Dictionary of cat breeds and their similar breeds
        # Key: Breed Name
        # Value: List of similar breeds
        "Abyssinian": ["Egyptian Mau", "Ocicat", "Bengal"],
        "American Bobtail": ["Manx", "Siberian", "Maine Coon"],
        "American Shorthair": [
            "British Shorthair",
            "Russian Blue",
            "European Shorthair",
        ],
        "Bengal": ["Egyptian Mau", "Ocicat", "Savannah"],
        "Birman": ["Ragdoll", "Himalayan", "Maine Coon"],
        "Bombay": ["Burmese", "Tuxedo", "Tonkinese"],
        "British Shorthair": ["Scottish Fold", "American Shorthair", "Russian Blue"],
        "Egyptian Mau": ["Savannah", "Bengal", "Ocicat"],
        "Maine Coon": ["Norwegian Forest Cat", "Siberian", "Ragdoll"],
        "Persian": ["Himalayan", "Exotic Shorthair", "British Shorthair"],
        "Ragdoll": ["Maine Coon", "Norwegian Forest Cat", "Siberian"],
        "Russian Blue": ["British Shorthair", "Scottish Fold", "Chartreux"],
        "Siamese": ["Balinese", "Oriental", "Tonkinese"],
        "Sphynx": ["Devon Rex", "Cornish Rex", "Oriental"],
        "Tuxedo": ["Thai", "Burmese", "Bombay"],
    }

    # Step 1: Predict whether it's a cat or a dog
    # Preprocess for the first model
    image_cat_dog = pre_process(image, 224, normalize=True)

    # Load the first model
    modelCatDog = load_model(model_cat_vs_dog)

    # Prediction
    pred = np.argmax(modelCatDog.predict(image_cat_dog, verbose=0))

    if pred == 1:
        # Load the dog breed classification model
        modelDog = build_model(len(dog_breeds))
        modelDog.load_weights(weights_dog)

        # Preprocess for the dog breed model
        image_dog = pre_process(image, 600)

        # Prediction
        pred = np.argmax(modelDog.predict(image_dog, verbose=0))
        pred = list(dog_breeds.keys())[pred]
        similar_breeds = dog_breeds[pred]

        # Retrieve similar dog breeds
        list_pred = [pred] + similar_breeds

    else:
        # Load the cat breed classification model
        modelCat = build_model(len(cat_breeds))
        modelCat.load_weights(weights_cat)

        # Preprocess for the cat breed model
        image_cat = pre_process(image, 600)

        # Prediction
        pred = np.argmax(modelCat.predict(image_cat, verbose=0))
        pred = list(cat_breeds.keys())[pred]
        similar_breeds = cat_breeds[pred]

        # Retrieve similar cat breeds
        list_pred = [pred] + similar_breeds

    return list_pred

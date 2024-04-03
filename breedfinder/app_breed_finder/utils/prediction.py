import numpy as np
import keras
from keras import layers
from keras.applications import EfficientNetB7
from tensorflow.keras.models import load_model
import cv2


def predict_image(image, model_cat_vs_dog, weights_dog, weights_cat):

    def build_model(num_classes):
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
        img = cv2.resize(img, (dim, dim))
        img = np.array(img)
        if normalize:
            img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def similar_breeds(pred, list_similar):
        for list in list_similar:
            if pred in list:
                list.remove(pred)
                list.insert(0, pred)
                return list

    # List des races
    # chien
    dog_breeds = [
        "Chihuahua",
        "Japanese Spaniel",
        "Maltese Dog",
        "Pekinese",
        "Shih-Tzu",
        "Blenheim Spaniel",
        "Papillon",
        "Toy Terrier",
        "Rhodesian Ridgeback",
        "Afghan Hound",
        "Basset",
        "Beagle",
        "Bloodhound",
        "Bluetick",
        "Black-and-tan Coonhound",
        "Walker Hound",
        "English Foxhound",
        "Redbone",
        "Borzoi",
        "Irish Wolfhound",
        "Italian Greyhound",
        "Whippet",
        "Ibizan Hound",
        "Norwegian Elkhound",
        "Otterhound",
        "Saluki",
        "Scottish Deerhound",
        "Weimaraner",
        "Staffordshire Bullterrier",
        "American Staffordshire Terrier",
        "Bedlington Terrier",
        "Border Terrier",
        "Kerry Blue Terrier",
        "Irish Terrier",
        "Norfolk Terrier",
        "Norwich Terrier",
        "Yorkshire Terrier",
        "Wire-haired Fox Terrier",
        "Lakeland Terrier",
        "Sealyham Terrier",
        "Airedale",
        "Cairn",
        "Australian Terrier",
        "Dandie Dinmont",
        "Boston Bull",
        "Miniature Schnauzer",
        "Giant Schnauzer",
        "Standard Schnauzer",
        "Scotch Terrier",
        "Tibetan Terrier",
        "Silky Terrier",
        "Soft-coated Wheaten Terrier",
        "West Highland White Terrier",
        "Lhasa",
        "Flat-coated Retriever",
        "Curly-coated Retriever",
        "Golden Retriever",
        "Labrador Retriever",
        "Chesapeake Bay Retriever",
        "German Short-haired Pointer",
        "Vizsla",
        "English Setter",
        "Irish Setter",
        "Gordon Setter",
        "Brittany Spaniel",
        "Clumber",
        "English Springer",
        "Welsh Springer Spaniel",
        "Cocker Spaniel",
        "Sussex Spaniel",
        "Irish Water Spaniel",
        "Kuvasz",
        "Schipperke",
        "Groenendael",
        "Malinois",
        "Briard",
        "Kelpie",
        "Komondor",
        "Old English Sheepdog",
        "Shetland Sheepdog",
        "Collie",
        "Border Collie",
        "Bouvier Des Flandres",
        "Rottweiler",
        "German Shepherd",
        "Doberman",
        "Miniature Pinscher",
        "Greater Swiss Mountain Dog",
        "Bernese Mountain Dog",
        "Appenzeller",
        "Entlebucher",
        "Boxer",
        "Bull Mastiff",
        "Tibetan Mastiff",
        "French Bulldog",
        "Great Dane",
        "Saint Bernard",
        "Eskimo Dog",
        "Malamute",
        "Siberian Husky",
        "Affenpinscher",
        "Basenji",
        "Pug",
        "Leonberg",
        "Newfoundland",
        "Great Pyrenees",
        "Samoyed",
        "Pomeranian",
        "Chow",
        "Keeshond",
        "Brabancon Griffon",
        "Pembroke",
        "Cardigan",
        "Toy Poodle",
        "Miniature Poodle",
        "Standard Poodle",
        "Mexican Hairless",
        "Dingo",
        "Dhole",
        "African Hunting Dog",
    ]

    # chiens similaires
    similar_dog_breeds = [
        ["Chihuahua", "Japanese Spaniel", "Maltese Dog", "Pekinese"],
        ["Shih-Tzu", "Blenheim Spaniel", "Papillon", "Toy Terrier"],
        ["Rhodesian Ridgeback", "Afghan Hound", "Basset", "Beagle"],
        ["Bloodhound", "Bluetick", "Black-and-tan Coonhound", "Walker Hound"],
        ["English Foxhound", "Redbone", "Borzoi", "Irish Wolfhound"],
        ["Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound"],
        ["Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner"],
        [
            "Staffordshire Bullterrier",
            "American Staffordshire Terrier",
            "Bedlington Terrier",
            "Border Terrier",
        ],
        ["Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier", "Norwich Terrier"],
        [
            "Yorkshire Terrier",
            "Wire-haired Fox Terrier",
            "Lakeland Terrier",
            "Sealyham Terrier",
        ],
        ["Airedale", "Cairn", "Australian Terrier", "Dandie Dinmont"],
        [
            "Miniature Schnauzer",
            "Giant Schnauzer",
            "Standard Schnauzer",
            "Scotch Terrier",
        ],
        [
            "Tibetan Terrier",
            "Silky Terrier",
            "Soft-coated Wheaten Terrier",
            "West Highland White Terrier",
        ],
        [
            "Lhasa",
            "Flat-coated Retriever",
            "Curly-coated Retriever",
            "Golden Retriever",
        ],
        [
            "Labrador Retriever",
            "Chesapeake Bay Retriever",
            "German Short-haired Pointer",
            "Vizsla",
        ],
        ["English Setter", "Irish Setter", "Gordon Setter", "Brittany Spaniel"],
        ["Clumber", "English Springer", "Welsh Springer Spaniel", "Cocker Spaniel"],
        ["Sussex Spaniel", "Irish Water Spaniel", "Kuvasz", "Schipperke"],
        ["Groenendael", "Malinois", "Briard", "Kelpie"],
        ["Komondor", "Old English Sheepdog", "Shetland Sheepdog", "Collie"],
        ["Border Collie", "Bouvier Des Flandres", "Rottweiler", "German Shepherd"],
        [
            "Doberman",
            "Miniature Pinscher",
            "Greater Swiss Mountain Dog",
            "Bernese Mountain Dog",
        ],
        ["Appenzeller", "Entlebucher", "Boxer", "Bull Mastiff"],
        ["Tibetan Mastiff", "French Bulldog", "Great Dane", "Saint Bernard"],
        ["Eskimo Dog", "Malamute", "Siberian Husky", "Affenpinscher"],
        ["Basenji", "Pug", "Leonberg", "Newfoundland"],
        ["Great Pyrenees", "Samoyed", "Pomeranian", "Chow"],
        ["Keeshond", "Brabancon Griffon", "Pembroke", "Cardigan"],
        ["Toy Poodle", "Miniature Poodle", "Standard Poodle", "Mexican Hairless"],
        ["Dingo", "Dhole", "African Hunting Dog"],
    ]

    # chat
    cat_breeds = []

    # chats similaires
    similar_cat_breeds = []

    # première étape : prédire si c'est un chien ou un chat
    # pre process pour le premier model
    image_cat_dog = pre_process(image, 224, normalize=True)

    # chargement du premier model
    modelCatDog = load_model(model_cat_vs_dog)

    # prediction
    pred = np.argmax(modelCatDog.predict(image_cat_dog))

    if pred == 1:
        # chargement du model Dog
        modelDog = build_model(len(dog_breeds))
        modelDog.load_weights(weights_dog)

        # pre process pour le model dog
        image_dog = pre_process(image, 600)

        # preidciton
        pred = np.argmax(modelDog.predict(image_dog))
        pred = dog_breeds[pred]

        # récupération des races de chiens similaires
        list_pred = similar_breeds(pred, similar_dog_breeds)

    else:
        # chargement du model Dog
        modelCat = build_model(len(cat_breeds))
        modelCat.load_weights(weights_cat)

        # pre process pour le model dog
        image_cat = pre_process(image, 600)

        # preidciton
        pred = np.argmax(modelCat.predict(image_cat))
        pred = cat_breeds[pred]

        # récupération des races de chats similaires
        list_pred = similar_breeds(pred, similar_cat_breeds)

    return list_pred

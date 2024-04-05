import csv
from app_breed_finder.models import Breed

file_path = "app_breed_finder/static/app_breed_finder/data/breeds.csv"

with open(file_path, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:

        # Create a Breed instance from the CSV data
        instance = Breed(
            name=row["name"],
            animal=row["animal"],
            description=row["description"],
            personality=row["personality"],
            origin=row["origin"],
            eating_habits=row["eating_habits"],
            life_esperance_bottom=int(row["life_esperance_bottom"]),
            life_esperance_top=int(row["life_esperance_top"]),
        )

        # Save the object into the database
        instance.save()

print("CSV data has been loaded into the Django database.")

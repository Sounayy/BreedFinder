from django.db import models


class Breed(models.Model):

    class Animal(models.TextChoices):
        CAT = "CAT"
        DOG = "DOG"

    name = models.fields.CharField(max_length=100)
    animal = models.fields.CharField(choices=Animal.choices, max_length=50)
    description = models.fields.TextField()
    personality = models.fields.TextField()
    origin = models.fields.TextField()
    eating_habits = models.fields.TextField()

    def __str__(self) -> str:
        return f"{self.name}"

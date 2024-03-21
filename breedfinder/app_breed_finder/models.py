from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator


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
    life_esperance_bottom = models.fields.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(30)]
    )
    life_esperance_top = models.fields.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(30)]
    )
    profile_picture = models.ImageField(upload_to="images/", null=True)

    def __str__(self) -> str:
        return f"{self.name}"

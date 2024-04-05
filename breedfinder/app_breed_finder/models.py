from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator


class Breed(models.Model):
    """Model representing a breed."""

    class Animal(models.TextChoices):
        """Choices for the animal type."""

        CAT = "CAT"
        DOG = "DOG"

    name = models.CharField(max_length=100)
    animal = models.CharField(choices=Animal.choices, max_length=50)
    description = models.TextField()
    personality = models.TextField()
    origin = models.TextField()
    eating_habits = models.TextField()
    life_esperance_bottom = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(30)]
    )
    life_esperance_top = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(30)]
    )
    profile_picture = models.ImageField(upload_to="images/", null=True)

    def __str__(self) -> str:
        """String representation of the breed."""
        return f"{self.name}"

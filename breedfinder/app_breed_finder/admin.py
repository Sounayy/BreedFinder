from django.contrib import admin
from app_breed_finder.models import Breed


class BreedAdmin(admin.ModelAdmin):
    """
    Admin interface customization for the Breed model.
    """

    list_display = ("name", "animal")


admin.site.register(Breed, BreedAdmin)

from django.contrib import admin
from app_breed_finder.models import Breed


class BreedAdmin(admin.ModelAdmin):
    list_display = ("name", "animal")


admin.site.register(Breed, BreedAdmin)

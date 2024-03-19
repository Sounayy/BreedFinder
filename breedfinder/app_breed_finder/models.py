from django.db import models


class Breed(models.Model):

    name = models.fields.CharField(max_length=100)

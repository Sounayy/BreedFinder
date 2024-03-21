# Generated by Django 5.0.3 on 2024-03-21 13:55

import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("app_breed_finder", "0003_rename_decription_breed_description"),
    ]

    operations = [
        migrations.AddField(
            model_name="breed",
            name="life_esperance_bottom",
            field=models.IntegerField(
                default=10,
                validators=[
                    django.core.validators.MinValueValidator(1),
                    django.core.validators.MaxValueValidator(30),
                ],
            ),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="breed",
            name="life_esperance_top",
            field=models.IntegerField(
                default=15,
                validators=[
                    django.core.validators.MinValueValidator(1),
                    django.core.validators.MaxValueValidator(30),
                ],
            ),
            preserve_default=False,
        ),
    ]
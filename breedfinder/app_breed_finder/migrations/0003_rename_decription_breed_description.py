# Generated by Django 5.0.3 on 2024-03-20 08:57

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        (
            "app_breed_finder",
            "0002_breed_animal_breed_decription_breed_eating_habits_and_more",
        ),
    ]

    operations = [
        migrations.RenameField(
            model_name="breed",
            old_name="decription",
            new_name="description",
        ),
    ]

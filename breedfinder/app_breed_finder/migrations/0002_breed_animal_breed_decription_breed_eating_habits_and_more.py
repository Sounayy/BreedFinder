# Generated by Django 5.0.3 on 2024-03-20 08:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("app_breed_finder", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="breed",
            name="animal",
            field=models.CharField(
                choices=[("CAT", "Cat"), ("DOG", "Dog")], default="CAT", max_length=50
            ),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="breed",
            name="decription",
            field=models.TextField(default=""),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="breed",
            name="eating_habits",
            field=models.TextField(default=""),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="breed",
            name="origin",
            field=models.TextField(default=""),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="breed",
            name="personality",
            field=models.TextField(default=""),
            preserve_default=False,
        ),
    ]

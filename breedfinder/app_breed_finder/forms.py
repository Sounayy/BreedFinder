from django import forms


class UploadImageForm(forms.Form):
    image_uploaded = forms.ImageField(
        error_messages={"required": "Veuillez upload une image."}, label=""
    )

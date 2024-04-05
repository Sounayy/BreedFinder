from django import forms


class UploadImageForm(forms.Form):
    """
    A form for uploading an image.
    """

    image_uploaded = forms.ImageField(
        error_messages={"required": "Please upload an image."},
        label="",
    )

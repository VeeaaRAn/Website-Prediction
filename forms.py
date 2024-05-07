from django import forms
from .models import Post

class PostForm(forms.ModelForm):
    class Meta:
        model = Post
        fields = ['text', 'image']

class ImageForm(forms.Form):
    image = forms.ImageField(label='Image')

    




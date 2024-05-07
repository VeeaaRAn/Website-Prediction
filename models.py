from django.db import models
from sorl.thumbnail import ImageField

class Post(models.Model):
    text = models.TextField()
    image = models.ImageField()  # Path to store uploaded images
    prediction = models.CharField(max_length=255, blank=True)  # Field for storing prediction
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.text[:20]}..."  # Truncate text for display

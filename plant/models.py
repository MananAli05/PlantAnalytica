from django.db import models
class Plant(models.Model):
    image = models.ImageField(upload_to='plants/')
    prediction_name = models.CharField(max_length=100, blank=True, null=True)
    disease_name = models.CharField(max_length=100, blank=True, null=True)
    prescription = models.TextField(blank=True, null=True)

    def __str__(self):
        return self.prediction_name if self.prediction_name else "Unpredicted Plant"

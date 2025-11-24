from rest_framework import serializers
from .models import Plant
class PlantSerializer(serializers.ModelSerializer):
    class Meta:
        model = Plant
        fields = 'image', 'prediction_name', 'disease_name', 'prescription'
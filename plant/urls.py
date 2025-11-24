from .views import PlantDetectionView,home
from django.urls import path

urlpatterns = [
    path('',home, name='home'),
    path('api/detect-plant/', PlantDetectionView.as_view(), name='detect-plant'),
]
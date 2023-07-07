from django.urls import path
from secondapp import views


urlpatterns = [
    path('secondhome/',views.home),
    # Add more URL patterns as needed
]

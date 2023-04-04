from django.urls import path
from authapp.views import *
from authapp import views


urlpatterns = [
    path('register/', UserRegistrationView.as_view(),name='register'),
    path('login/', UserLoginView.as_view(),name='loginin'),
    path('logout/', LogoutUser.as_view(),name='logout'),
    path('generate/', views.ContenViews.as_view()),
]
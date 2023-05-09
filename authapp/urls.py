from django.urls import path
from authapp.views import *
from authapp import views


urlpatterns = [
    path('register/', UserRegistrationView.as_view(),name='register'),
    path('login/', UserLoginView.as_view(),name='loginin'),
    path('logout/', LogoutUser.as_view(),name='logout'),
    path('generate/', views.ContenViews.as_view()),
    path('cricketscraping/',CricketScrapingView.as_view(),name='scraps'),
    path('get_mobile/',WebScrapDataView.as_view(),name='scrap'),
    # path('get_tech/',TechnologyView.as_view(),name='scrap'),

]
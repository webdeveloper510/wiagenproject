from django.urls import path
from authapp.views import *
from authapp import views


urlpatterns = [
    path('register/', UserRegistrationView.as_view(),name='register'),
    path('login/', UserLoginView.as_view(),name='loginin'),
    path('logout/', LogoutUser.as_view(),name='logout'),
    path('generate/', views.ContenViews.as_view()),
    path('cricketscraping/',CricketScrapingView.as_view(),name='scraps'),
    path('mobiletechnologyscraping/',WebScrapDataView.as_view()),
    path('mobiletechnologyscraping2/',MobileAppDevelopementView.as_view()),
    path('footballscraping/',FootballScrapingView.as_view(),name='scrap'),
    path('technologyscraping/',EmergingTechnologyView.as_view(),name='scrap'),
    path('prediction/',TechnologiesView.as_view()),
    path('listquestionandanswer/',QuestionandAnswerListView.as_view())
]

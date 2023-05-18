from django.urls import path
from authapp.views import *
from authapp import views


urlpatterns = [
    path('register/', UserRegistrationView.as_view(),name='register'),
    path('login/', UserLoginView.as_view(),name='loginin'),
    path('userprofile/', ProfileView.as_view(),name='profile'),
    path('logout/', LogoutUser.as_view(),name='logout'),
    path('adminscrapping/', views.AdminScraping.as_view()),
    path('generate/', views.ContenViews.as_view()),
    path('cricketscraping/',CricketScrapingView.as_view(),name='scraps'),
    path('mobiletechnologyscraping/',WebScrapDataView.as_view()),
    path('mobiletechnologyscraping2/',MobileAppDevelopementView.as_view()),
    path('footballscraping/',FootballScrapingView.as_view()),
    path('technologyscraping/',EmergingTechnologyView.as_view()),
    path('prediction/',TechnologiesView.as_view()),
    path('label/<int:user_id>',GetLabelByUser_id.as_view()),
    path('pdfreader/',PDFView.as_view()),
]

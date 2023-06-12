from django.urls import path
from authapp.views import *
from authapp import views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('register/', UserRegistrationView.as_view(),name='register'),
    path('login/', UserLoginView.as_view(),name='loginin'),
    path('userprofile/', ProfileView.as_view(),name='profile'),
    path('logout/', LogoutUser.as_view(),name='logout'),
    path('adminscrapping/', views.AdminScraping.as_view()),
    path('prediction/',TechnologiesView.as_view()),
    # path('label/<int:user_id>/',GetLabelByUser_id.as_view()),
    path('label/',GetLabelByUser_id.as_view()),
    path('pdfresult/',PDFReaderView.as_view()),
    path('pdfdata/',GetAllPdf.as_view()),
    path('urldata/',GetALLUrls.as_view()),
    path('SaveData/',SaveQuestionAnswer.as_view()),
    path('ShowData/<int:user_id>/',ShowAllData.as_view()),
    path('local/',local_save.as_view()),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)


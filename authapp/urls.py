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
    path('prediction1/',prediction1.as_view()),
    path('prediction2/',prediction2.as_view()),
    path('finalprediction/',finalPrediction.as_view()),
    path('label/',GetLabelByUser_id.as_view()),
    path('pdfresult/',PDFReaderView.as_view()),
    path('pdfdata/',GetAllPdf.as_view()),
    path('urldata/',GetALLUrls.as_view()),
    path('SaveData/',SaveQuestionAnswer.as_view()),
    path('ShowData/',ShowAllData.as_view()),
    path('trainmodel/',Train_model.as_view()),
    path('twodatabase/',TrainSecondDatabase.as_view()),
    path('finaltrainmodel/',finalTrainModel.as_view()),   
    path('deletelabel/',label_delete.as_view()),
    path('deletequestion/',question_delete.as_view()),
    # USER API's
    path('createdatabase/',createuserdatabase.as_view()),
    path('userdatabase/',TrainUserDatabase.as_view()),
    path('alluserdata/',GetUserDatabase.as_view()),
    path('userprediction/',UserPrediction.as_view()),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)


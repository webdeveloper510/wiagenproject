from .models import *
from .models import LANGUAGE_CHOICES
from .serializers import *
from rest_framework_simplejwt.tokens import RefreshToken
from authapp.renderer import UserRenderer
from rest_framework.permissions import IsAuthenticated
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import action
from rest_framework.views import APIView
from distutils import errors
from rest_framework.response import Response
from django.contrib.auth import authenticate
from rest_framework import status
import openai
import googletrans
from googletrans import Translator
from django.conf import settings

translator = Translator()
openai.api_key=settings.API_KEY

#Creating tokens manually
def get_tokens_for_user(user):
    refresh = RefreshToken.for_user(user)
    return {
        'refresh': str(refresh),
        'access': str(refresh.access_token),
    }

class UserRegistrationView(APIView):
 renderer_classes=[UserRenderer]
 def post(self,request,format=None):
    serializer=UserRegistrationSerializer(data=request.data)
    if serializer.is_valid(raise_exception=True):
        user=serializer.save()
        return Response({'message':'Registation successful',"status":"status.HTTP_200_OK"})
    return Response({errors:serializer.errors},status=status.HTTP_400_BAD_REQUEST)
 
class UserLoginView(APIView):
    renderer_classes=[UserRenderer]
    def post(self,request,format=None):
        email=request.data.get('email')
        password=request.data.get('password')
        user=authenticate(email=email,password=password)
        
        if user is not None:
              token= get_tokens_for_user(user)
              return Response({'message':'Login successful','status':'status.HTTP_200_OK',"token":token})
        else:
              return Response({'message':'Please Enter Valid email or password',"status":"status.HTTP_404_NOT_FOUND"})

class LogoutUser(APIView):
  renderer_classes = [UserRenderer]
  permission_classes=[IsAuthenticated]
  def post(self, request, format=None):
    return Response({'message':'Logout Successfully','status':'status.HTTP_200_OK'})


class ContenViews(APIView):
    renderer_classes=[UserRenderer]  
    def post(self,request):
        input=request.data.get('input')
        language=request.data.get('language')
        user_id=request.data.get('user_id')
        if not input:
            return Response({"message":"please provide input text"})
        # if not Language.objects.filter(language=language).exists():
        #      return Response({"message":"please provide valid language"})
        if not language:
            return Response({"message":"please select langusge"})
       
        if not user_id:
            return Response({"message":"user is required"})
        if not User.objects.filter(id=user_id).exists():
             return Response({"message":" user does not exist"})
        user=User.objects.get(id=user_id)
        user.user=user
        content_data=Content.objects.create(input=input,language=language,user_id=user)
        serializers=ContentSerializer(data=content_data)
        content_data.save()
        input_text=content_data.input                   ## get input data from the database.
        language_detect=content_data.language           ## get language from the database
        content_id=content_data.id
        response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Auto Response Generator \n\nUser: {input_text} \n\nAI:\n",
        temperature=0.7,
        max_tokens=600,
        top_p=1,
        frequency_penalty=1,    
        presence_penalty=1,
        )
        output= response.choices[0].text
        translated_output = translator.translate(output, dest=language_detect).text
        update_content=Content.objects.filter(id=content_id).update(output=translated_output)
        return Response({'msg':'Data Added Succesfully','status':'status.HTTP_201_CREATED','output':translated_output})
         
# @csrf_exempt
# def index1(request):    
#         input_text=request.POST.get('alpha')
#         if input_text:
#             language=detect(input_text)
#             text_writer=content_generate(input_text,language)  ### call to function 
#             return render(request,'generate.html',{"output":text_writer})
#         else:
#             return HttpResponse(" Please Enter Some Text")
#     else:
#         return render(request,'generate.html')
 

                  

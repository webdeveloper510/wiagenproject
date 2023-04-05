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
        variant=request.data.get('variant')
        user_id=request.data.get('user_id')
        if not input:
            return Response({"message":"please provide input text"})
        if not language:
            return Response({"message":"please select langusge"})
        if not variant:
            return Response({"message":"select variant for creativity"})
        if not user_id:
            return Response({"message":"user is required"})
        if not User.objects.filter(id=user_id).exists():
             return Response({"message":" user does not exist"})
         
        user=User.objects.get(id=user_id)
        user.user=user
        content_data=Content.objects.create(input=input,language=language,user_id=user,variant=variant)
        serializers=ContentSerializer(data=content_data)
        content_data.save()
        
#### get data from the database
       
        input_text=content_data.input                   ## get input data from the database.
        language_detect=content_data.language           ## get language from the database
        variant_data=content_data.variant               ## get variant option detect from the database
        content_id=content_data.id
        
        if variant=="1 variant":
            loops=1
        elif variant=="2 variant":
            loops=2
        elif variant=="3 variant":
            loops=3
        else:
            return Response ({"message":"Invalid varinat value"})
        
        outputs=[]
        for i in range(loops):
            response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"Auto Response Generator \n\nUser: {input_text} \n\nAI:\n",
            temperature=1,
            max_tokens=300,
            top_p=1,
            frequency_penalty=1,    
            presence_penalty=1,
            )
            output= response.choices[0].text
            translated_output = translator.translate(output, dest=language_detect).text                
            outputs.append(translated_output)
            print('VARIANT ---------------->',outputs)
        update_content=Content.objects.filter(id=content_id).update(output=outputs)
        return Response({'msg':'Data Added Succesfully','status':'status.HTTP_201_CREATED','output':outputs})


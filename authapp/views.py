from .models import *
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

api_key="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"   
openai.api_key=api_key
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
        Languageid=Language.objects.get(id=language)
        Languageid.Languageid=Languageid
        content_data=Content.objects.create(input=input,language=Languageid,user_id=user)
        serializers=ContentSerializer(data=content_data)
        content_data.save()
        cdata=content_data.input
        content_id=content_data.id
        response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Auto Response Generator \n\nUser: {cdata} \n\nAI:\n",
        temperature=0.7,
        max_tokens=300,
        top_p= 1,
        frequency_penalty=1,    
        presence_penalty=1,
        )
        output= response.choices[0].text
        update_content=Content.objects.filter(id=content_id).update(output=output)
        return Response({'msg':'Data Added Succesfully','status':'status.HTTP_201_CREATED','output':output})
        
        # content_object = Content.objects.all()
        # serializer=ContentSerializer(data=request.data)
        # if serializer.is_valid(raise_exception=True):
        #     user=serializer.save()
        # return Response({'msg':'Data Added Succesfully'},status=status.HTTP_201_CREATED)
        # return Response({errors:serializer.errors},status=status.HTTP_400_BAD_REQUEST)
    
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
 

                  

from rest_framework import serializers
from .models import *

class UserRegistrationSerializer(serializers.ModelSerializer):
    class Meta:
        model=User
        fields=['id','email','password','firstname','lastname']

        extra_kwargs={
            'email': {'error_messages': {'required': "email is required",'blank':'please provide a email'}},
            'password': {'error_messages': {'required': "password is required",'blank':'please Enter a password'}},
            'firstname': {'error_messages': {'required': "firstname is required",'blank':'firstname could not blank'}},
            'lastname': {'error_messages': {'required': "lastname is required",'blank':'lastname could not blank'}},
          }
        
    def create(self, validated_data,):
       user=User.objects.create(
       email=validated_data['email'],
       firstname=validated_data['firstname'],
       lastname=validated_data['lastname'],)
       user.set_password(validated_data['password']) 
       user.save()
       return user
    
class ContentSerializer(serializers.ModelSerializer):
    class Meta:
        model= Content
        fields="__all__"
        
class LanguageSerializer(serializers.ModelSerializer):
    class Meta:
        model= Language
        fields="__all__"
        

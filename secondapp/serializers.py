from rest_framework import serializers
from secondapp.models import *


class Topic2Serializer(serializers.ModelSerializer):
    class Meta:
        model= Topic2
        fields = '__all__'
        
        
class database2QuestionAndAnswrSerializer(serializers.ModelSerializer):
    class Meta:
        model= database2QuestionAndAnswr
        fields = '__all__'
        
class User_PDF2Serializer(serializers.ModelSerializer):
    class Meta:
        model= User_PDF2
        fields = '__all__'
           
    def create(self, validate_data):
        return User_PDF2.objects.create(**validate_data)
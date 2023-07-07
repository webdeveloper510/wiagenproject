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
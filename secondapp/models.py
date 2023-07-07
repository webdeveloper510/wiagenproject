from django.db import models
from django.contrib.auth.models import *


class Topic2(models.Model):
    topic_name=models.CharField(max_length=200,null=True,blank=True)
    
class database2QuestionAndAnswr(models.Model):
    topic=models.ForeignKey(Topic2,on_delete=models.CASCADE)
    question=models.TextField(max_length=1000)
    answer=models.TextField(max_length=1000)
    
    

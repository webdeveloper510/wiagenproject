from django.db import models
from django.contrib.auth.models import *
from authapp.models import *


class Topic2(models.Model):
    topic_name=models.CharField(max_length=200,null=True,blank=True)
    
class database2QuestionAndAnswr(models.Model):
    topic=models.ForeignKey(Topic2,on_delete=models.CASCADE)
    question=models.TextField(max_length=1000)
    answer=models.TextField(max_length=1000)
    
    
class User_PDF2(models.Model):
    pdf=models.FileField(upload_to="user_pdf/",blank=True,null=True)
    pdf_filename=models.CharField(max_length=200,blank=True,null=True)
    
class UrlTable2(models.Model):
    url=models.CharField(max_length=200,blank=True,null=True)
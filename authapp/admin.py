from django.contrib import admin
from .models import *

# Register your models here.
@admin.register(Content)
class ServiceAgreementAdmin(admin.ModelAdmin):
  list_display = ('id','user_id','input','output','variant','language')
  
@admin.register(Language)
class ServiceAgreementAdmin(admin.ModelAdmin):
  list_display = ('id','language')


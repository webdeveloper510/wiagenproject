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
from django.db import DatabaseError
import googletrans
from googletrans import Translator
from django.conf import settings
from bs4 import BeautifulSoup
import requests
import json
import re
import pandas as pd
import numpy as np
translator = Translator()
from keras.utils import pad_sequences
from nltk.corpus import stopwords
from numpy import loadtxt
from rest_framework.authentication import TokenAuthentication
from keras.models import load_model
from keras.models import Sequential,model_from_json 
from keras.layers import LSTM, Dense,Dropout ,Embedding,SpatialDropout1D,GlobalAveragePooling1D 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
import itertools
from django.core.handlers.wsgi import WSGIRequest
import os
import PyPDF2 as pdf
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
import re 
import pdfplumber
from django.contrib.auth import login
import spacy
import nltk
from keybert import KeyBERT
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import pickle
import sys
import mysql.connector
import sentencepiece
openai.api_key=settings.API_KEY
nlp = spacy.load('en_core_web_sm')
from urllib.parse import urljoin
from secondapp.models import *
from secondapp.serializers import *
from django.contrib.auth import get_user_model
User = get_user_model()
# url="http://127.0.0.1:8000/static/media/"
url="http://13.53.234.84:8000/static/media/"


#Creating tokens manually
def get_tokens_for_user(user):
    refresh = RefreshToken.for_user(user)
    return {
        'refresh': str(refresh),
        'access': str(refresh.access_token),
    }
    
def connect_to_database(user_database):
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database=user_database)
    return mydb
class UserRegistrationView(APIView):
    renderer_classes=[UserRenderer]
    def post(self,request,format=None):
        serializer=UserRegistrationSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            user=serializer.save()
            database_name= f"{serializer.data['firstname']}{serializer.data['id']}db"
            User_Id=User.objects.get(id=serializer.data['id'])
            datadb=UserdatabaseName.objects.create(database_name=database_name,user=User_Id)
            datadb.save()
            return Response({'message':'Registation successful',"status":"status.HTTP_200_OK",'user_id': user.id})
        return Response({errors:serializer.errors},status=status.HTTP_400_BAD_REQUEST)
 
class UserLoginView(APIView):
    renderer_classes=[UserRenderer]
    def post(self,request,format=None):
        email = request.data.get('email')
        password = request.data.get('password')
        user = authenticate(email=email, password=password)
        if not email:
             return Response({"message":"email is required"},status=status.HTTP_400_BAD_REQUEST)
        if not password:
             return Response({"message":"password is required"},status=status.HTTP_400_BAD_REQUEST)
        
        if not User.objects.filter(email=email).exists(): 
            return Response({"message":"invalid email address"},status=status.HTTP_400_BAD_REQUEST)
        
        if user:
            login(request, user)
            token=get_tokens_for_user(user)
            user_data=User.objects.filter(email=email).values("firstname")
            username=user_data[0]['firstname']
            return Response({'message':'Login successful',"username":username,"token":token},status=status.HTTP_200_OK)
        else:
              return Response({'message':'Please Enter Valid email or password'},status=status.HTTP_400_BAD_REQUEST)

class ProfileView(APIView):
    renderer_classes=[UserRenderer]
    permission_classes=[IsAuthenticated]
    def post(self,request,format=None):
        serializer = UserProfileSerializer(request.user)
        return Response(serializer.data,status=status.HTTP_200_OK)


class LogoutUser(APIView):
    renderer_classes = [UserRenderer]
    permission_classes=[IsAuthenticated]
    def post(self, request, format=None):
        return Response({'message':'Logout Successfully','status':'status.HTTP_200_OK'})

## Admin Prediction Question Answer From Database 1.
class prediction1(APIView):
    model_path=os.getcwd()+"/authapp/saved_file/saved_model/classification_model.json"
    model_weight_path=os.getcwd()+"/authapp/saved_file/saved_model/classification_model_weights.h5"
    cluster_label_path=os.getcwd()+'/authapp/saved_file/saved_model/cluster_labels.pkl'
    def clean_text(self,text):
        REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')     
        BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
        STOPWORDS=set(stopwords.words("english"))
        if not text:
            return ""
        text=text.lower()
        text=REPLACE_BY_SPACE_RE.sub(' ',text)
        text=BAD_SYMBOLS_RE.sub(' ',text)
        text=text.replace('x','')
        text=' '.join(word for word in text.split() if word not in STOPWORDS)
        return text
    
    def chatgpt(self,input):
        response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Auto Response Generator \n\nUser {input} \n\nAI:",
        temperature=1,
        max_tokens=1000,
        top_p=0,
        frequency_penalty=1,
        presence_penalty=1
        )
        output= response.choices[0].text
        return output
    def automaticgetlabel(self, text):
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(text)
        keyword= kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english',top_n=1)
        return keyword
    
    def post(self, request,user_input):
        service = QuestionAndAnswer.objects.all().order_by('id')
        serializer = QuestionAndAnswerSerializer(service, many=True)
        array=[]
        for x in serializer.data:
            topic_id=x["topic"]
            question=self.clean_text(x["question"])
            answer=x["answer"]
            topicvalue=Topic.objects.filter(id=topic_id).values('topic_name')
            TopicName=(topicvalue[0]['topic_name'])
            data_dict={"Topic":TopicName,"question":question,"answer":answer}
            array.append(data_dict)
            
        data=[dict['question'] for dict in array]
        # Tokenizer data.
        MAX_NB_WORDS = 10000
        MAX_SEQUENCE_LENGTH =200
        EMBEDDING_DIM = 100
        oov_token = "<OOV>"
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', oov_token = "<OOV>", lower=True)
        tokenizer.fit_on_texts(data)
        word_index = tokenizer.word_index
        
        # get the json labels
        with open(self.cluster_label_path, "rb") as file:
            cluster_labels = pickle.load(file)
        # Load Saved Model.
        json_file = open(self.model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(self.model_weight_path)
        
        
        # Take user input and preprocess as input
        clean_user_input=self.clean_text(user_input)
        new_input = tokenizer.texts_to_sequences([clean_user_input])
        new_input = pad_sequences(new_input, maxlen=MAX_SEQUENCE_LENGTH) 

        # Make Prediction
        pred =loaded_model.predict(new_input)
        databasew_match=pred, cluster_labels[np.argmax(pred)]
        result=databasew_match[1]
        # Get the answer based on the question.
        filter_data = [dict for dict in array if dict["Topic"].strip()== result.strip()]
        get_all_questions=[dict['question'] for dict in filter_data] 
        vectorizer = TfidfVectorizer()
        vectorizer.fit(get_all_questions)
        question_vectors = vectorizer.transform(get_all_questions)                                  # 2. all questions
        input_vector = vectorizer.transform([clean_user_input])
        # check the similarity of the model
        # similarity_scores = question_vectors.dot(input_vector.T).toarray().squeeze()
        similarity_scores = question_vectors.dot(input_vector.T).toarray().flatten()  # Ensure 1-dimensional array
        max_sim_index = np.argmax(similarity_scores)
        similarity_percentage = similarity_scores[max_sim_index] * 100
        if (similarity_percentage)>=65:
            answer = filter_data[max_sim_index]['answer']               
            response_data = {
                "Question":user_input,
                "Label": result,
                "Answer": answer,
                "AnswerSource":"This Response is Coming From Database 1"}
            return response_data

        else:
            input = user_input
            response = self.chatgpt(input)
            get_label = self.automaticgetlabel(response)
            title_label=get_label[0][0]
            label=title_label.title()
            response_data = {
                "Question":user_input,
                "Label": label,
                "Answer": response,
                "AnswerSource":"This Response is Coming From Chatgpt 1"}
            return response_data
        
## Admin Prediction Question Answer From Database 2.
class prediction2(APIView):
    model2_path=os.getcwd()+"/secondapp/saved_model/2classification_model.json"
    model2_weight_path=os.getcwd()+"/secondapp/saved_model/2classification_model_weights.h5"
    cluster_path=os.getcwd()+"/secondapp/saved_model/2cluster_labels.pkl"
    technology=prediction1()
    def post(self,request,user_input):
        print('DATABASE2')
        database2=database2QuestionAndAnswr.objects.using('second_db').all().order_by('id')
        database2_serializer =database2QuestionAndAnswrSerializer(database2, many=True)
        array=[]
        for data in database2_serializer.data:
            topic_id=data["topic"]
            question=self.technology.clean_text((data["question"]))
            answer=data["answer"]
            topicvalue = Topic2.objects.using('second_db').filter(id=topic_id).values('topic_name')
            TopicName=(topicvalue[0]['topic_name'])
            data_dict={"Topic":TopicName,"question":question,"answer":answer}
            array.append(data_dict)
        Questions=[dict['question'] for dict in array]
        # define the parameter for model.
        MAX_NB_WORDS = 10000
        MAX_SEQUENCE_LENGTH =200
        EMBEDDING_DIM = 100
        oov_token = "<OOV>"
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',oov_token = "<OOV>", lower=True)
        tokenizer.fit_on_texts(Questions)
        word_index = tokenizer.word_index
        sequence= tokenizer.texts_to_sequences(Questions)
        ## Create input for model
        input_data=pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)             # input 
        
        # convert label into dummies using LabelEncoder.
        Y_data = [dict['Topic'].strip() for dict in array]
        lbl_encoder = LabelEncoder()
        lbl_encoder.fit(Y_data)
        output_Y = lbl_encoder.transform(Y_data) 
        with open(self.cluster_path, "rb") as file:
            cluster_labels = pickle.load(file)
        json_file = open(self.model2_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(self.model2_weight_path)
        
        user_input=request.POST.get('input').title()
        clean_user_input=self.technology.clean_text(user_input)
        new_input = tokenizer.texts_to_sequences([clean_user_input])
        new_input = pad_sequences(new_input, maxlen=MAX_SEQUENCE_LENGTH) 

        # Make Prediction
        pred =loaded_model.predict(new_input)
        databasew_match=pred, cluster_labels[np.argmax(pred)]
        result=databasew_match[1]
        # Get the answer based on the question.
        filter_data = [dict for dict in array if dict["Topic"].strip()== result.strip()]
        get_all_questions=[dict['question'] for dict in filter_data] 
        vectorizer = TfidfVectorizer()
        vectorizer.fit(get_all_questions)
        question_vectors = vectorizer.transform(get_all_questions)                                  # 2. all questions
        input_vector = vectorizer.transform([clean_user_input])
        similarity_scores = question_vectors.dot(input_vector.T).toarray().flatten()  # Ensure 1-dimensional array
        max_sim_index = np.argmax(similarity_scores)
        similarity_percentage = similarity_scores[max_sim_index] * 100
        if (similarity_percentage)>=65:
            answer = filter_data[max_sim_index]['answer']               
            response_data = {
                "Question":user_input,
                "Label": result,
                "Answer": answer,
                "AnswerSource":"This Response is Coming From Database 2"}
            return response_data
        
        else:
            input = user_input
            response = self.technology.chatgpt(input)
            get_label = self.technology.automaticgetlabel(input)
            title_label=get_label[0][0]
            label=title_label.title()
            response_data = {
                "Question":user_input,
                "Label": label,
                "Answer": response,
                "AnswerSource":"This Response is Coming From Chatgpt 2"}
            return response_data

# Admin final Prediction API
class finalPrediction(APIView):
    def post(self,request):
        selected_database=request.data.get("database_id")
        user_input=request.data.get('input')
        database=databaseName.objects.filter(id=selected_database).values('database_name')
        database=database[0]['database_name']
        if database=="default":
            database1 = prediction1()
            response = database1.post(request, user_input)
            return Response(response)
        else:
            database2=prediction2()
            response=database2.post(request,user_input)
            return Response(response)

# Admin API for Make Question And Answer From the URL Text
class AdminScraping(APIView):
    def run_model(self,input_strings,tokenizer,model,**generator_args):
        input_ids = tokenizer.batch_encode_plus(input_strings, return_tensors="pt", padding=True, truncation=True)["input_ids"]
        res = model.generate(input_ids, **generator_args)
        output = tokenizer.batch_decode(res, skip_special_tokens=True)
        return output

    def clean_text(self,text):
        cleaned_text = re.sub(r'\s+', ' ', text)
        cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
        return cleaned_text.strip()

    def split_text_into_qa_pairs(self,text):
        if not isinstance(text, str):
            raise ValueError("Input 'text' must be a string.")
        qa_pairs = []
        rules = re.split(r'\d+\s*', text)
        for rule in rules[1:]:
            rule = rule.strip()
            
            if not rule:
                continue
            rule_lines = rule.split('\n')
            question = rule_lines[0].strip().lstrip('Ans:')
            answer = ' '.join(rule_lines).strip().title()
            if question and answer:
                qa_pairs.append((question, answer))
        return qa_pairs
    
    def post(self, request, format=None):
        database_name=request.data.get("database_id")
        database = databaseName.objects.filter(id=database_name).values('database_name')
        database = database[0]['database_name']
        print("DATABASE------------------------>>>>>",database)
        url = request.data.get("url")
        if not url:
            return Response({"message":"url is required"},status=status.HTTP_400_BAD_REQUEST)
        if database == "default":
            print("Database 1 is here -------------------------------->>.")
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0;  Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
            url_data=UrlTable.objects.create(url=url)
            response = requests.get(url,headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")
            all_text = soup.get_text()
        else:
            print("Database 2 is here -------------------------------->>.")
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
            url_data=UrlTable2.objects.using('second_db').create(url=url)
            response = requests.get(url,headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")
            all_text = soup.get_text()
        
        if all_text=="":
            return Response({"message":"No Data Found"},status=status.HTTP_200_OK)
        else:
            all_text = re.sub(r'\s+', ' ', all_text).strip()
            model_name = "allenai/t5-small-squad2-question-generation"
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name)
            extracted_data =self.clean_text(all_text)
            qa_pairs = self.split_text_into_qa_pairs(extracted_data)
            questions =self.run_model([pair[1] for pair in qa_pairs],tokenizer,model, max_new_tokens=256)
            generated_qa_pairs = list(zip(questions, [pair[1] for pair in qa_pairs]))
            response_data = {"QA_Pairs": []}
            
            for i, (question, answer) in enumerate(generated_qa_pairs):
                technologiesview=prediction1()
                get_label=technologiesview.automaticgetlabel(answer)
                if len(get_label) > 0:
                    title_label = get_label[0][0]
                    print("Title------------------>>>>", title_label)
                    label = title_label.title()
                else:
                    label="Label Not Found"                
                response_data["QA_Pairs"].append({
                    "Question": question.strip(),
                    "Answer": answer.strip(),
                    "Label": label.strip()})
            return Response(response_data)
    
# Admin API for Get Label
class GetLabelByUser_id(APIView):
    def post(self, request ,format=None):
        database_name=request.data.get("database_id")
        database = databaseName.objects.filter(id=database_name).values('database_name')
        database = database[0]['database_name']
        print('dTAAAAAAA------------->>>',database)
        if database == "default":
            data= Topic.objects.all().values().order_by("-id")
            return Response({"data":data})
        else:
            data= Topic2.objects.using('second_db').all().values().order_by("-id")
            return Response({"data":data})

# Admin API for Make Question And Answer From the PDF Text
class PDFReaderView(APIView):
    def run_model(self,input_strings, tokenizer ,model,**generator_args):
        input_ids = tokenizer.batch_encode_plus(input_strings, return_tensors="pt", padding=True, truncation=True)["input_ids"]
        res = model.generate(input_ids, **generator_args)
        output = tokenizer.batch_decode(res, skip_special_tokens=True)
        return output
    def pdfreader_func(self,path):
        file = open(path, 'rb')
        doc = pdf.PdfReader(file)
        if doc.is_encrypted:
            return  "PDF is Encrypted"
        page_number = len(doc.pages)
        extracted_text = ''
        for i in range(page_number):
            current_page = doc.pages[i]
            text = current_page.extract_text()
            extracted_text += text 
        return extracted_text
    
    def split_text_into_qa_pairs(self, text):
        if not isinstance(text, str):
            raise ValueError("Input 'text' must be a string.")
        qa_pairs = []
        rules = re.split(r'\n(?=\d+)', text)
        for rule in rules:
            rule = rule.strip()
            if not rule:
                continue
            rule_lines = rule.split('\n')
            question = rule_lines[0].strip().lstrip('Ans:')
            answer = ' '.join(rule_lines[1:]).strip().title()
            if question and answer:
                qa_pairs.append((question, answer))
        return qa_pairs
    def post(self,request, format=None):
        pdffile=  request.FILES.get("pdf")
        database_name=request.data.get("database_id")
        database = databaseName.objects.filter(id=database_name).values('database_name')
        database = database[0]['database_name']
        if database == "default":
            print('HERE Database1---------------->>>')
            database1_pdffile=User_PDF.objects.create(pdf=pdffile,pdf_filename=pdffile.name)
            serializer=User_PDFSerializer(data=database1_pdffile)
            database1_pdffile.save()
            full_url = urljoin(url, database1_pdffile.pdf.name)
            pdf_path=database1_pdffile.pdf.path
            database1_pdffile.pdf = full_url
            database1_pdffile.save()
        else:
            print('HERE DATABASE2-------------------->>>')
            database2_pdffile=User_PDF2.objects.using('second_db').create(pdf=pdffile,pdf_filename=pdffile.name)
            serializer=User_PDF2Serializer(data=database2_pdffile)
            database2_pdffile.save()
            full_url = urljoin(url, database2_pdffile.pdf.name)
            pdf_path=database2_pdffile.pdf.path
            database2_pdffile.pdf = full_url
            database2_pdffile.save()
            
        model_name = "allenai/t5-small-squad2-question-generation"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        extracted_text = self.pdfreader_func(pdf_path)
        qa_pairs = self.split_text_into_qa_pairs(extracted_text)
        questions =self.run_model([pair[1] for pair in qa_pairs], tokenizer,model,max_new_tokens=256)
        generated_qa_pairs = list(zip(questions, [pair[1] for pair in qa_pairs]))
        aligned_qa_pairs = [(f"Q.{i+1} {question.strip()}\nAns: {answer.strip()}") for i, (question, answer) in enumerate(generated_qa_pairs)]
        response_data = {"QA_Pairs": []}
        for i, (question, answer) in enumerate(generated_qa_pairs):
            technologiesview=prediction1()
            print('Answer------------>>>>',answer)
            get_label=technologiesview.automaticgetlabel(answer)
            title_label=get_label[0][0]
            label=title_label.title()
            response_data["QA_Pairs"].append({
                "Question": question.strip(),
                "Answer": answer.strip(),
                "Label": label.strip()})
        return Response(response_data)

# Admin API For Get PDF From The Pdf Table in Database.
class GetAllPdf(APIView):
    def post(self, request, format=None):
            selected_database=request.data.get("database_id")
            database=databaseName.objects.filter(id=selected_database).values('database_name')
            database=database[0]['database_name']
            if database=="default":
                database1_pdffiles= User_PDF.objects.all().values().order_by('-id')
                return Response({'data':database1_pdffiles})
            else:
                database2_pdffiles= User_PDF2.objects.using('second_db').all().values().order_by('-id')
                return Response({'data':database2_pdffiles})
                

class GetALLUrls(APIView):
    def post(self, request, format=None):
        selected_database=request.data.get("database_id")
        database=databaseName.objects.filter(id=selected_database).values('database_name')
        database=database[0]['database_name']
        if database=="default":
            data= UrlTable.objects.all().values().order_by('-id')
            return Response({'data':data})
        else:
            print("second")
            data= UrlTable2.objects.using('second_db').all().values().order_by('-id')
            return Response({'data':data})

# Admin API for Save Question Answer in Database.
class SaveQuestionAnswer(APIView):
    def post(self, request, format=None):
        print("here")
        response = request.data.get("Response")
        print("response------------------>>>>",response)
        select_database = request.data.get("database_id")
        database = databaseName.objects.filter(id=select_database).values('database_name')
        database = database[0]['database_name']
        # get_label=""
        if database == "default":
            for data in response:
                question = data['question']
                answer = data['answer']
                label = data['label']
                if not question and not answer and not label:
                    return Response({"message": "Data Not Found"})
                # Get all topic names.
                all_labels = Topic.objects.all().values('topic_name')
                labels = [data['topic_name'] for data in all_labels]
                best_match = None
                best_similarity = 0
                for item in labels:
                    similarity = fuzz.ratio(label, item)
                    if 50 <= similarity <= 90:
                        best_match = item
                        break  # Exit the loop after finding the first matching label
                print('best match---------->',best_match)
                if best_match:
                    get_label = best_match
                    update=Topic.objects.filter(topic_name=label).update(topic_name=get_label.title())
                else:
                    get_label = label
                if not Topic.objects.filter(topic_name=get_label.title()).exists():
                    topic_save = Topic.objects.create(topic_name=get_label.title())
                    topic_save.save()
                filter_topic_id = Topic.objects.filter(topic_name=get_label.title()).values("id")
                topic_id = filter_topic_id[0]['id']
                saveQuesAns = QuestionAndAnswer.objects.create(question=question, answer=answer, topic_id=topic_id)
                saveQuesAns.save()
        else:
            for data in response:
                question=data['question']
                answer=data['answer']
                label=data['label']
                if not question and not answer and not label:
                    return Response({"message":"Data Not Found"})
                
                # get all topic name.
                All_labels=Topic2.objects.using('second_db').all().values('topic_name')
                access_labels=[data['topic_name'].strip() for data in All_labels ]
                best_match = None
                best_similarity = 0
                for item in access_labels:
                    similarity = fuzz.ratio(label, item)
                    if 50 <= similarity <= 90:
                        best_match = item
                        break  # Exit the loop after finding the first matching label
                if best_match:
                    get_label = best_match
                    update=Topic2.objects.using('second_db').filter(topic_name=label).update(topic_name=get_label.title())
                else:
                    get_label = label
                if not Topic2.objects.using('second_db').filter(topic_name = get_label.title()).exists():
                    topic_save=Topic2.objects.using('second_db').create(topic_name=get_label.title())
                    topic_save.save()
                filter_topic_id= Topic2.objects.using('second_db').filter(topic_name=get_label).values("id")
                topic_id = filter_topic_id[0]['id']
                saveQuesAns=database2QuestionAndAnswr.objects.using('second_db').create(question=question,answer=answer,topic_id=topic_id)
                saveQuesAns.save()
        return Response({"message":"Data Save Sucessfully"})

# Admin API For Showing Data based on the User ID
class ShowAllData(APIView):
    def post(self, request, format=None):
        selected_database=request.data.get("database_id")
        database=databaseName.objects.filter(id=selected_database).values('database_name')
        database=database[0]['database_name']
        label_id = request.data.get("id")   ## GET TOPIC ID
        if database=="default":
            print("First DATABASE ")
            if not Topic.objects.filter(id=label_id).exists():
                return Response({"message": "Data Not Found"})
            questions = QuestionAndAnswer.objects.filter(topic_id=label_id).values("question")
            answers = QuestionAndAnswer.objects.filter(topic_id=label_id).values("answer")
            id= QuestionAndAnswer.objects.filter(topic_id=label_id).values("id")
            response_data = []
            for question_data, answer_data ,id_data in zip(questions, answers,id):
                response_data.append({
                    "Question": question_data["question"],
                    "Answer": answer_data["answer"],
                    "id":id_data["id"]
                })
            if response_data:
                return Response(response_data)
            else:
                return Response({"message":"Data Not Found"})
        else:
            if not Topic2.objects.using("second_db").filter(id=label_id).exists():
                return Response({"message": "Data Not Found"})
            print("second DATABASE ")
            questions = database2QuestionAndAnswr.objects.using('second_db').filter(topic_id=label_id).values("question")
            answers = database2QuestionAndAnswr.objects.using('second_db').filter(topic_id=label_id).values("answer")
            id = database2QuestionAndAnswr.objects.using('second_db').filter(topic_id=label_id).values("id")

            response_data = []
            for question_data, answer_data ,id_data in zip(questions, answers,id):
                response_data.append({
                    "Question": question_data["question"],
                    "Answer": answer_data["answer"],
                    "id": id_data["id"]
                })
            if response_data:
                return Response(response_data)
            else:
                return Response({"message":"Data Not Found"})
            # return Response(response_data)
    
# Admin API For Train Database 1 Data.
class Train_model(APIView):
    def post(self,request,format=None):
        technologiesview=prediction1() 
        get_data = QuestionAndAnswer.objects.all().order_by('id')
        serializer = QuestionAndAnswerSerializer(get_data, many=True)
        # create array with question ,answer and topic.
        array=[]
        for x in serializer.data:
            topic_id=x["topic"]
            question=technologiesview.clean_text((x["question"]))
            answer=x["answer"]
            topicvalue=Topic.objects.filter(id=topic_id).values('topic_name')
            TopicName=(topicvalue[0]['topic_name'])
            data_dict={"Topic":TopicName,"question":question,"answer":answer}
            array.append(data_dict)
        
        Questions=[dict['question'] for dict in array]

        # define the parameter for model.
        MAX_NB_WORDS = 10000
        MAX_SEQUENCE_LENGTH =200
        EMBEDDING_DIM = 100
        oov_token = "<OOV>"
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',oov_token = "<OOV>", lower=True)
        tokenizer.fit_on_texts(Questions)
        word_index = tokenizer.word_index
        sequence= tokenizer.texts_to_sequences(Questions)
        ## Create input for model
        input_data=pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)             # input 
        
        # convert label into dummies using LabelEncoder.
        Y_data=[dict['Topic'] for dict in array]
        lbl_encoder = LabelEncoder()
        lbl_encoder.fit(Y_data)
        output_Y = lbl_encoder.transform(Y_data) 
        
        
        cluster_label = lbl_encoder.classes_.tolist()
        num_class=len(cluster_label)
          
        # define the layers for sequential model.
        model = Sequential()
        model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(num_class, activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
        epochs = 800
        batch_size=128
        model.fit(input_data, np.array(output_Y), epochs=epochs, batch_size=batch_size)

        # # Save Model
        model_json=model.to_json()
        with open(os.getcwd()+"/authapp/saved_file/saved_model/classification_model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(os.getcwd()+"/authapp/saved_file/saved_model/classification_model_weights.h5")
        # Save the cluster label list
        with open(os.getcwd()+"/authapp/saved_file/saved_model/cluster_labels.pkl", "wb") as file:
            pickle.dump(cluster_label, file)
        return Response({"message":"Model Trained and Saved with successfully."})

# Admin API For Train Database 2 Data.
class TrainSecondDatabase(APIView):
    technologies=prediction1()
    clean_preprocess=technologies.clean_text
    chatgpt_model=technologies.chatgpt
    
    def post(self, request):
        database2=database2QuestionAndAnswr.objects.using('second_db').all().order_by('id')
        database2_serializer =database2QuestionAndAnswrSerializer(database2, many=True)
        array=[]
        for data in database2_serializer.data:
            topic_id=data["topic"]
            question=self.technologies.clean_text((data["question"]))
            answer=data["answer"]
            topicvalue = Topic2.objects.using('second_db').filter(id=topic_id).values('topic_name')
            TopicName=(topicvalue[0]['topic_name'])
            data_dict={"Topic":TopicName,"question":question,"answer":answer}
            array.append(data_dict)
        
        Questions=[dict['question'] for dict in array]
        # define the parameter for model.
        MAX_NB_WORDS = 10000
        MAX_SEQUENCE_LENGTH =200
        EMBEDDING_DIM = 100
        oov_token = "<OOV>"
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',oov_token = "<OOV>", lower=True)
        tokenizer.fit_on_texts(Questions)
        word_index = tokenizer.word_index
        sequence= tokenizer.texts_to_sequences(Questions)
        ## Create input for model
        input_data=pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)             # input 
        
        # convert label into dummies using LabelEncoder.
        Y_data = [dict['Topic'].strip() for dict in array]
        lbl_encoder = LabelEncoder()
        lbl_encoder.fit(Y_data)
        output_Y = lbl_encoder.transform(Y_data) 
        cluster_label = lbl_encoder.classes_.tolist()
        num_class=len(cluster_label)
        # define the layers for sequential model.
        model = Sequential()
        model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(num_class, activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
        epochs =800
        batch_size=128
        model.fit(input_data, np.array(output_Y), epochs=epochs, batch_size=batch_size)
        # # Save Model
        model_json=model.to_json()
        with open(os.getcwd()+"/secondapp/saved_model/2classification_model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(os.getcwd()+"/secondapp/saved_model/2classification_model_weights.h5")
        # Save the cluster label list
        with open(os.getcwd()+"/secondapp/saved_model/2cluster_labels.pkl", "wb") as file:
            pickle.dump(cluster_label, file)
        return Response({'message':"Mode Train Accurate"})
    
    
class finalTrainModel(APIView):
    def post(self,request):
        selected_database=request.data.get("database_id")
        database=databaseName.objects.filter(id=selected_database).values('database_name')
        database=database[0]['database_name']
        if database=="default":
            classCall = Train_model()
            classCall.post(request)
            return Response({"message":"Model Train Successfully"})
        else:
            second=TrainSecondDatabase()
            second.post(request)
            return Response({"message":"Model Train Successfully"})

# Admin API For Delete Label And related Question Answer       
class label_delete(APIView):
    def post(self,request, format=None):
        label_id=request.data.get("label_id")
        database_name=request.data.get('database_id')
        database = databaseName.objects.filter(id=database_name).values('database_name')
        database = database[0]['database_name']
        if database == "default":
            delete_all_data= QuestionAndAnswer.objects.filter(topic_id=label_id).delete()
            deleted_label = Topic.objects.filter(id=label_id).delete()
            return Response({"message":"Data is deleted"})
        else:
            delete_all_data2= database2QuestionAndAnswr.objects.using("second_db").filter(topic_id=label_id)
            deleted_label2= Topic2.objects.using("second_db").filter(id=label_id).delete()
            return Response({"message":"Data is deleted"})
            
# Admin API For Delete Question And related Question Answer       
class question_delete(APIView):
    def post(self,request, format=None):
        question_id=request.data.get("question_id")
        database_name=request.data.get('database_id')
        database = databaseName.objects.filter(id=database_name).values('database_name')
        database = database[0]['database_name']
        if database == "default":
            print('Database------1')
            if not QuestionAndAnswer.objects.filter(id=question_id).exists():
                return Response({"message":"No Data Found"})
            else:
                delete_all_data= QuestionAndAnswer.objects.filter(id=question_id).delete()
                return Response({"message":"Data is deleted"})
        else:
            print('Database------2')
            if not database2QuestionAndAnswr.objects.using('second_db').filter(id=question_id).exists():
                return Response({"message":"No Data Found"})
            else:
                delete_all_data= database2QuestionAndAnswr.objects.using("second_db").filter(id=question_id).delete()
                return Response({"message":"Data is deleted"})
            
            
## User Related API's     
class createuserdatabase(APIView):
    def post(self, request):
        user_id = request.data.get('user_id')
        try:
            if not user_id:
                return Response({"status": status.HTTP_400_BAD_REQUEST, "message": "user_id is required"})
            try:
                user = User.objects.get(id=user_id)
            except User.DoesNotExist:
                return Response({"status": status.HTTP_400_BAD_REQUEST, "message": "user does not exist"})
            userdb = UserdatabaseName.objects.filter(user=user).values('database_name')
            if not userdb:
                return Response({"status": status.HTTP_400_BAD_REQUEST, "message": "user database name not found"})
            userdata_base_name = userdb[0]['database_name']
            print("user_id---------->>>",userdata_base_name)
            mydb = mysql.connector.connect(
                host="localhost",
                user="root",
                password=""
            )
            mycursor = mydb.cursor()
            mycursor.execute(f"CREATE DATABASE IF NOT EXISTS {userdata_base_name}")
            mydb = connect_to_database(userdata_base_name)
            mycursor = mydb.cursor()
            # Create the Topic table if not exists
            mycursor.execute("CREATE TABLE IF NOT EXISTS Topic (id INT AUTO_INCREMENT PRIMARY KEY, topic_name VARCHAR(255))")
            
            # Create the QuestionAndAnswer table if not exists
            mycursor.execute("CREATE TABLE IF NOT EXISTS QuestionAndAnswer (id INT AUTO_INCREMENT PRIMARY KEY, user_id VARCHAR(255), topic_id INT, question TEXT, answer TEXT, FOREIGN KEY (topic_id) REFERENCES Topic(id))")
            mydb.commit()
            return Response({"status": status.HTTP_200_OK, "message": "Database and tables created successfully"})
        except Exception as e:
            return Response({"status": status.HTTP_500_INTERNAL_SERVER_ERROR, "message": str(e)})

# User Database Train Model API's
class TrainUserDatabase(APIView):
    def post(self,request):
        user_id=request.data.get("user_id")
        userdb=UserdatabaseName.objects.filter(user=user_id).values("database_name")
        userdata_base_name=userdb[0]['database_name']
        mydb =connect_to_database()
        mycursor = mydb.cursor(userdata_base_name)
        mycursor.execute("Select topic_id, question,answer From QuestionAndAnswer")
        data=mycursor.fetchall()
        array=[]
        for record in data:
            topic_id=record[0]
            question=record[1]
            answer=record[2]
            mycursor.execute("SELECT topic_name FROM Topic WHERE id = %s", (topic_id,))
            existing_topic = mycursor.fetchall()
            
            if existing_topic:
                topic_name = existing_topic[0][0]  # Extract the topic_name from the fetched result
                data_dict={"Topic":topic_name,"question":question,"answer":answer}
                array.append(data_dict)
        Questions=[dict['question'] for dict in array]
        
        # define the parameter for model.
        MAX_NB_WORDS = 10000
        MAX_SEQUENCE_LENGTH =200
        EMBEDDING_DIM = 100
        oov_token = "<OOV>"
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',oov_token = "<OOV>", lower=True)
        tokenizer.fit_on_texts(Questions)
        word_index = tokenizer.word_index
        sequence= tokenizer.texts_to_sequences(Questions)
        ## Create input for model
        input_data=pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)             # input 
        
        # convert label into dummies using LabelEncoder.
        Y_data=[dict['Topic'] for dict in array]
        lbl_encoder = LabelEncoder()
        lbl_encoder.fit(Y_data)
        output_Y = lbl_encoder.transform(Y_data) 
        
        
        cluster_label = lbl_encoder.classes_.tolist()
        num_class=len(cluster_label)
          
        # define the layers for sequential model.
        model = Sequential()
        model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(num_class, activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
        epochs = 1000
        batch_size=128
        model.fit(input_data, np.array(output_Y), epochs=epochs, batch_size=batch_size)
        
        # make directory with userdatabase name.
        model_dir = os.path.join(os.getcwd(), "authapp", "saved_file", userdata_base_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        # # Save Model
        model_json=model.to_json()
        with open(f"{model_dir}/classification_model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(f"{model_dir}/classification_model_weights.h5")
        # Save the cluster label list
        with open(f"{model_dir}/cluster_labels.pkl", "wb") as file:
            pickle.dump(cluster_label, file)
        return Response({"message":"Model Trained and Saved with successfully."})
    
# User Prediction Question And Answer
class UserPrediction(APIView):
    def post(self,request):
        prediction=prediction1()
        user_id=request.data.get("user_id")
        user_input=request.data.get("user_input")
        userdb=UserdatabaseName.objects.filter(user=user_id).values("database_name")
        userdata_base_name=userdb[0]['database_name']
        mydb = connect_to_database(userdata_base_name)
        mycursor = mydb.cursor()
        mycursor.execute("Select topic_id, question,answer From QuestionAndAnswer")
        data=mycursor.fetchall()
        array=[]
        for record in data:
            topic_id=record[0]
            question=record[1]
            answer=record[2]
            mycursor.execute("SELECT topic_name FROM Topic WHERE id = %s", (topic_id,))
            existing_topic = mycursor.fetchall()
            
            if existing_topic:
                topic_name = existing_topic[0][0]  # Extract the topic_name from the fetched result
                data_dict={"Topic":topic_name,"question":question,"answer":answer}
                array.append(data_dict)
        Questions=[dict['question'] for dict in array]
        
        # define the parameter for model.
        MAX_NB_WORDS = 10000
        MAX_SEQUENCE_LENGTH =200
        EMBEDDING_DIM = 100
        oov_token = "<OOV>"
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',oov_token = "<OOV>", lower=True)
        tokenizer.fit_on_texts(Questions)
        word_index = tokenizer.word_index
        
        # directory path
        model_path=os.getcwd()+f"/authapp/saved_file/{userdata_base_name}/classification_model.json"
        model_weight_path=os.getcwd()+f"/authapp/saved_file/{userdata_base_name}/classification_model_weights.h5"
        cluster_label_path=os.getcwd()+f"/authapp/saved_file/{userdata_base_name}/cluster_labels.pkl"
        
        with open(cluster_label_path, "rb") as file:
            cluster_labels = pickle.load(file)
        # Load Saved Model.
        json_file = open(model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(model_weight_path)
        
        # Take user input and preprocess as input
        clean_user_input=prediction.clean_text(user_input)
        new_input = tokenizer.texts_to_sequences([clean_user_input])
        new_input = pad_sequences(new_input, maxlen=MAX_SEQUENCE_LENGTH) 

        # Make Prediction
        pred =loaded_model.predict(new_input)
        databasew_match=pred, cluster_labels[np.argmax(pred)]
        result=databasew_match[1]
        
        # Get the answer based on the question.
        filter_data = [dict for dict in array if dict["Topic"].strip()== result.strip()]
        get_all_questions=[dict['question'] for dict in filter_data] 
        vectorizer = TfidfVectorizer()
        vectorizer.fit(get_all_questions)
        question_vectors = vectorizer.transform(get_all_questions)                                  # 2. all questions
        input_vector = vectorizer.transform([clean_user_input])
        # check the similarity of the model
        similarity_scores = question_vectors.dot(input_vector.T).toarray().flatten()  # Ensure 1-dimensional array
        max_sim_index = np.argmax(similarity_scores)
        similarity_percentage = similarity_scores[max_sim_index] * 100
        if (similarity_percentage)>=65:
            answer = filter_data[max_sim_index]['answer']     
            response_data = {
                "Question":user_input,
                "Label": result,
                "Answer": answer,
                "AnswerSource":f"This Response is Coming From {userdata_base_name}."}
        else:
            input = user_input
            response = prediction.chatgpt(input)
            get_label = prediction.automaticgetlabel(response)
            title_label=get_label[0][0]
            label=title_label.title()
            response_data = {
                "Question":user_input,
                "Label": label,
                "Answer": response,
                "AnswerSource":"This Response is Coming From Chatgpt"}
            return response_data
        return Response({"message":response_data})
    
## Get All Created User Databases.
class GetUserDatabase(APIView):
    def get(self,request,format=None):
        user_database= UserdatabaseName.objects.all().values()
        return Response({"user_database":user_database})
    
# API For Getting Label of USER.
class UserLAbelShow(APIView):
    def post(self,request):
        user_id=request.data.get("user_id")
        userdb=UserdatabaseName.objects.filter(user=user_id).values("database_name")
        userdata_base_name=userdb[0]['database_name']
        mydb=connect_to_database(userdata_base_name)
        mycursor=mydb.cursor()
        mycursor.execute("Select id ,topic_name From Topic")
        data=mycursor.fetchall()
        if data:
            labels = []
            for result in data:
                label = {
                    "id": result[0],
                    "topic_name": result[1]
                }
                labels.append(label)
            mydb.close()
        else:
            return Response({"data":"Data Not Found"})
        return Response({"data":labels})
    
class ShowUserLabelDAta(APIView):
    def post(self,request,format=None):
        user_id=request.data.get("user_id")
        label_id=request.data.get("label_id")
        
        userdb=UserdatabaseName.objects.filter(user=user_id).values("database_name")
        userdata_base_name=userdb[0]['database_name']
        mydb = connect_to_database(userdata_base_name)
        mycursor = mydb.cursor()
        mycursor.execute("SELECT question, answer FROM QuestionAndAnswer WHERE topic_id= %s", (label_id,))
        data = mycursor.fetchall()
        mydb.close()
        if data:
            print("data 0------>>",data[0])
            question, answer = data[0]  
            response_data = {
                "Question": question,
                "Answer": answer,
                "id": label_id
                }
        else:
            return Response({"data":"Data Not Found"})
        return Response({"data": response_data})
    
class ShowAdminLabel(APIView):
    def post(self,request):
        database_id=request.data.get("database_id")
        userdb=UserdatabaseName.objects.filter(id=database_id).values("database_name")
        userdata_base_name=userdb[0]['database_name']
        mydb=connect_to_database(userdata_base_name)
        mycursor=mydb.cursor()
        mycursor.execute("Select id ,topic_name From Topic")
        data=mycursor.fetchall()
        if data:
            labels = []
            for result in data:
                label = {
                    "id": result[0],
                    "topic_name": result[1]
                }
                labels.append(label)
            mydb.close()
        else:
            return Response({"data":"Data Not Found"})
        return Response({"data":labels})

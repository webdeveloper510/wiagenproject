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
from bs4 import BeautifulSoup
import requests
import re
import csv
import pandas as pd
import numpy as np
translator = Translator()
from keras.utils import pad_sequences
from nltk.corpus import stopwords
from numpy import loadtxt
from keras.models import load_model
from keras.models import Sequential,model_from_json 
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense,Dropout ,Embedding,SpatialDropout1D,GlobalAveragePooling1D 
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
import itertools
import os
import PyPDF2 as pdf
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
import re 
import pdfplumber
from django.contrib.auth import login
import spacy
import nltk
stemmer=PorterStemmer()
import sentencepiece
openai.api_key=settings.API_KEY
nlp = spacy.load('en_core_web_sm')
from urllib.parse import urljoin

url="http://127.0.0.1:8000/static/media/"


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
    
class TechnologiesView(APIView):
    model_path="/home/codenomad/Desktop/wiagenproject/authapp/saved_file/saved_model/classification_model.json"
    model_weight_path="/home/codenomad/Desktop/wiagenproject/authapp/saved_file/saved_model/classification_model_weights.h5"
    
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
    def automaticgetlabel(self,input):
        doc = nlp(input )
        merged_text = []
        label_found = False  
        for word in doc.ents:
            if word.label_ in ["GPE", "ORG", "LOC", "PERSON", "MONEY", "ORDINAL", "PRODUCT", "NORP", "FAC", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "PERCENT", "QUANTITY", "CARDINALS"]:
                merged_text.append(word.text.title())
                label_found=True
        label= " ".join(merged_text)
        if not label_found or label.strip() == "":
            alternative_label=[]
            for token in doc:
                if token.pos_ == "PROPN" or token.ent_type_ == "PERSON":
                    alternative_label.append(token.text.title())
                if token.pos_ == "NNP":
                    alternative_label.append(token.text.title())
                if token.pos_ == "VERB":
                    alternative_label.append(token.text.title())
            
            if alternative_label:
                label = " ".join(alternative_label)
            else:
                label="No Label Found"
        return label
    
    def post(self, request):
        service = QuestionAndAnswr.objects.all().order_by('id')
        serializer = QuestionAndAnswrSerializer(service, many=True)
        array=[]
        for x in serializer.data:
            topic_id=x["topic"]
            question=self.clean_text(x["question"])
            answer=x["answer"]
            topicvalue=Topic.objects.filter(id=topic_id).values('Topic')
            TopicName=(topicvalue[0]['Topic'])
            data_dict={"Topic":TopicName,"question":question,"answer":answer}
            array.append(data_dict)
        data=[dict['question'] for dict in array]
        # Tokenizer data.
        MAX_NB_WORDS = 1000
        MAX_SEQUENCE_LENGTH =200
        EMBEDDING_DIM = 100
        oov_token = "<OOV>"
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS, oov_token = "<OOV>", lower=True)
        tokenizer.fit_on_texts(data)
        word_index = tokenizer.word_index
        # FIND THE TOTAL NUMBER OF CLASS
        cluster=list(set(dict['Topic'] for dict in array))
        num_class=len(list(set(cluster)))
        #Load Saved Model.
        json_file = open(self.model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(self.model_weight_path)
        # Take user input
        user_input=request.POST.get('input')
        clean_user_input=self.clean_text(user_input)
        new_input_tokenizer = tokenizer.texts_to_sequences([clean_user_input])
        new_input = pad_sequences(new_input_tokenizer, maxlen=MAX_SEQUENCE_LENGTH) 
        # Make Prediction
        pred =loaded_model.predict(new_input)
        label=['Cricket', 'Mobile', 'Technology']
        databasew_match=pred, label[np.argmax(pred)]
        result=databasew_match[1]
        # # get the Answer
        filter_data = [dict for dict in array if dict["Topic"].strip()== result.strip()]
        get_all_questions=[dict['question'] for dict in filter_data] 
        vectorizer = TfidfVectorizer()
        vectorizer.fit(get_all_questions)
        question_vectors = vectorizer.transform(get_all_questions)                                  # 2. all questions
        input_vector = vectorizer.transform([clean_user_input])
        similarity_scores = question_vectors.dot(input_vector.T).toarray().squeeze()
        max_sim_index = np.argmax(similarity_scores)
        similarity_percentage = similarity_scores[max_sim_index] * 100
        if (similarity_percentage)>=70:
            answer = filter_data[max_sim_index]['answer']
            if not Topic.objects.filter(Topic=result).exists():
                userLabel_data=Topic.objects.create(Topic=result)
            response_data = {
                "Label": result,
                "Answer": answer,
                "AnswerSource":"This Response is Coming From Database"}
            return Response(response_data)
        else:
            input=user_input.title()
            label=self.automaticgetlabel(input)
            if not Topic.objects.filter(Topic=label).exists():
                userLabel_data=Topic.objects.create(Topic=label.strip())
            response=self.chatgpt(input)
            response_data = {
                "Label": label,
                "Answer": response,
                "AnswerSource":"This Response is Coming From Chatgpt"}
            return Response(response_data)

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
        url = request.data.get("url")
        if not url:
            return Response({"message":"url is required"},status=status.HTTP_400_BAD_REQUEST)
        else :
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
            url_data=UrlTable.objects.create(url=url)
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
                    technologiesview=TechnologiesView()
                    label=technologiesview.automaticgetlabel(question.title())
                    response_data["QA_Pairs"].append({
                        "Question": question.strip(),
                        "Answer": answer.strip(),
                        "Label": label.strip()})
                return Response(response_data)
    
class GetLabelByUser_id(APIView):
    def get(self, request ,format=None):
            userlabel= Topic.objects.all().values_list('id','Topic').order_by("-id")
            unique_id=[]
            unique_label=[]
            for label in userlabel:
                if label[1] not in unique_label:
                    if label[0] not in unique_id:
                        unique_label.append(label[1])
                        unique_id.append(label[0])
            return Response({"unique_id":unique_id,"unique_label":unique_label})


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
        pdffile_data=User_PDF.objects.create(pdf=pdffile,pdf_filename=pdffile.name)
        serializer=User_PDFSerializer(data=pdffile_data)
        pdffile_data.save()
        full_url = urljoin(url, pdffile_data.pdf.name)
        pdf_path=pdffile_data.pdf.path
        pdffile_data.pdf = full_url
        pdffile_data.save()
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
            technologiesview=TechnologiesView()
            label=technologiesview.automaticgetlabel(question.title())
            response_data["QA_Pairs"].append({
                "Question": question.strip(),
                "Answer": answer.strip(),
                "Label": label.strip()})
        return Response(response_data)
    
class GetAllPdf(APIView):
    def get(self, request, format=None):
            pdffiles= User_PDF.objects.all().values('pdf_filename','pdf').order_by('-id')
            pdffilename=[]
            pdfdownload=[]
            for pdf in pdffiles:
                if pdf['pdf_filename'] not in pdffilename:
                    pdffilename.append(pdf['pdf_filename'])
                    pdfdownload.append(pdf['pdf'])
            if pdffilename:
                return Response({'pdffilename': pdffilename, 'pdfdownload': pdfdownload})
            else:
                return Response({'data':"Pdf Does Not Exist"})

class GetALLUrls(APIView):
    def get(self, request, format=None):
            Allurl= UrlTable.objects.all().values('url').order_by('-id')
            url_list=[]
            for url in Allurl:
                if url not in url_list :
                    url_list.append(url)
            if url_list:
                return Response({'labels': list(url_list)})
            else:
                return Response({'data':"Url Does Not Found"})

class SaveQuestionAnswer(APIView):
    def post(self,request, format=None):
        response=request.data.get("Response")
        user_id=request.data.get('user_id')
        if not user_id:
            return Response({"message":"user is required"})
        if not User.objects.filter(id=user_id).exists():
            return Response({"message":"user does not  is exist"})
        for data in response:
            question=data['question']
            answer=data['answer']
            label=data['label']
            if not question and not answer and not label:
                return Response({"message":"Data Not Found"})
            if not Topic.objects.filter(Topic = label).exists():
                topic_save=Topic.objects.create(Topic=label)
                topic_save.save()
            topic_id= Topic.objects.filter(Topic=label).values("id")
            topic_id = topic_id[0]['id']
            technology_table=QuestionAndAnswr.objects.create(question=question,answer=answer,topic_id=topic_id,user_id=user_id)
        return Response({"message":"Data Save Sucessfully"})
        
class ShowAllData(APIView):
    def post(self, request,user_id, format=None):
        user_id=request.data.get('user_id')
        if not user_id:
            return Response({'message':'user_id is required'},status=status.HTTP_400_BAD_REQUEST)
        label_id = request.data.get("id")
        if not Topic.objects.filter(id=label_id).exists():
            return Response({"message": "Data Not Found"})
        questions = QuestionAndAnswr.objects.filter(topic_id=label_id).values("question")
        answers = QuestionAndAnswr.objects.filter(topic_id=label_id).values("answer")
        response_data = []
        for question_data, answer_data in zip(questions, answers):
            response_data.append({
                "Question": question_data["question"],
                "Answer": answer_data["answer"]
            })
        return Response(response_data)
    

class localtolive(APIView):
    def get(self, request, format=None):
        data = pd.read_csv('/home/codenomad/Desktop/wiagenproject/local_question_answer.csv')

        for index, row in data.iterrows():
            question = row['question']
            answer = row['answer']
            topic_id = row['topic']

            QuestionAndAnswr.objects.create(question=question, answer=answer, topic_id=topic_id)
            QuestionAndAnswr.save()
        return Response({"message": "Data saved successfully"})

        
 
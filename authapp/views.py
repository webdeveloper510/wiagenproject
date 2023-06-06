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
            max_tokens=1000,
            top_p=1,
            frequency_penalty=1,    
            presence_penalty=1,
            )   
            output= response.choices[0].text
            translated_output = translator.translate(output, dest=language_detect).text
            outputs.append(translated_output)
        update_content=Content.objects.filter(id=content_id).update(output=outputs)
        return Response({'msg':'Data Added Succesfully','status':'status.HTTP_201_CREATED','output':outputs})   
    


class ClusterView(APIView):
    def get(self,request):
        data=Cricket_Question_and_Answer.objects.all()
        serializer=CricketSerializer(data=data,many=True)
        if serializer.is_valid():
            df1=pd.DataFrame(data=serializer.data)
            print(df1)
        else:
            return Response({"message":"Sorry"})
        return Response({"message":df1})
    
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
    def post(self, request):
        user_id=request.data.get('user_id')
        if not user_id:
            return Response({'message':'user_id is required'},status=status.HTTP_400_BAD_REQUEST)
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
            userLabel_data=User_Label.objects.create(user_id=user_id,Label=result)
            response_data = {
                "Label": result,
                "Answer": answer,
                "AnswerSource":"[Database Response]"}
        # return Response({"Label":result,"Answer":answer})
            return Response(response_data)

        else:
            input=user_input.title()
            print('input-------------------------------------->>>>',input)
            doc = nlp(input)
            print('doc----------------------->>>>>',doc)
            # Merge consecutive NOUN tokens
            merged_text = []
            for word in doc.ents:
                print(word)
                print('-------------------->>',word.text)
                print('Label =================================>>>>>',word.text,word.label_)
                if word.label_ == "GPE" or "ORG" or "LOC" or "PERSON" or 'MONEY' or 'ORDINAL' or 'PRODUCT' or 'NORP' or 'FAC' or 'EVENT' or 'WORK_OF_ART'or'LAW' or 'LANGUAGE' or "PERCENT" or 'QUANTITY' or 'CARDINALS':
                    merged_text.append(word.text.title())
            sentence = " ".join(merged_text)
            label=sentence
            userLabel_data=User_Label.objects.create(user_id=user_id,Label=label)
            response=self.chatgpt(input)
            response_data = {
                "Label": label,
                "Answer": response,
                "AnswerSource":"[Chatgpt Response]"}
            return Response(response_data)

            # return Response({"Label":label,"Answer":response})


class CricketScrapingView(APIView):
    def post(self, request, format=None):
        topic_id=request.data.get('topic_id')
        if not topic_id:
            return Response({"message":"topic_id is required"})
        urls = ["https://www.prep4ias.com/top-300-cricket-general-knowledge-questions-and-answers/","https://www.edubabaji.com/top-50-cricket-gk-questions-answers-in-english/"]

        array=[] 
        for index, url in enumerate(urls):
            if index ==0:
                    print(url)

                    response = requests.get(url)
                    html_content = response.content
                    soup = BeautifulSoup(html_content, "html.parser")
                    pattern = r'^\d{1,2}\b'
                    h3_tags = soup.find_all("h3")
                    count = 0
                    for h3 in h3_tags:
                        if count == 60:
                            break
                        question=h3.text
                        question=re.sub(r'\.', '', question)
                        question=re.sub(r'\d+', '', question)
                        # print(question)
                        answer=h3.find_next("p").text
                        answer=re.sub(r'\.', '', answer)
                        answer= re.sub(r'\bAns\b', '', answer)
                        data_dict={"question":question,"answer":answer}
                        array.append(data_dict)
                        # print(array)
                        count += 1
            if index ==1:
                    print(url)
                    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}

                    response = requests.get(url,headers=headers)

                    soup = BeautifulSoup(response.content, "html.parser")

                    video_wrapper = soup.find("div", {"class": "jetpack-video-wrapper"})

                    ol_tag = soup.find('ol')

                    for li_tag in ol_tag.find_all('li'):
                        li_text = li_tag.text.strip()
                        strong_tag = li_tag.find('strong')
                        if strong_tag:
                            strong_text = strong_tag.text.strip()
                        else:
                            strong_text = None
                        
                        question=li_text
                        question=question.split('?')[0] + '?'
                        # print(question)

                        answer=strong_text
                        answer=answer[1:]
                        # print(answer)
                        data_dict2={"question":question,"answer":answer}
                        array.append(data_dict2)
        print(array)
        for x in array:
             question=(x['question'])
             answer=(x['answer'])
             topic = Topic.objects.get(id= topic_id)
             topic.topic = topic
             cricketdata=QuestionAndAnswr.objects.create(topic=topic,question=question,answer=answer)
             serializer=QuestionAndAnswrSerializer(data=cricketdata)
             cricketdata.save()
        return Response({"message":"scrap data successfully","status":"200","data":len(array)})
    
#mobile waves
class WebScrapDataView(APIView):
     def post(self, request, format=None):
        topic_id=request.data.get('topic_id')
        if not topic_id:
            return Response({"message":"topic_id is required"})

        page = requests.get("https://buildfire.com/mobile-technology-waves/")
        soup = BeautifulSoup(page.content, "html.parser")
        p_tags = soup.select("h2 + p")
        array=[]
        count=0
        pattern = r'\d+'
        for p_tag in p_tags:
            h2_tag = p_tag.previous_sibling.previous_sibling
            if h2_tag is not None and h2_tag.name == "h2":
                print(h2_tag.text)
            print(p_tag.text)
            question = re.sub(pattern, '', h2_tag.text)
            answer=re.sub(pattern, '', p_tag.text)
            topic = Topic.objects.get(id= topic_id)
            topic.topic = topic
            scrappy=QuestionAndAnswr.objects.create(topic=topic,question=question,answer=answer)
            serializer=QuestionAndAnswrSerializer(data=scrappy)
            scrappy.save()
            count+=1
            if count==23:
                break
            dict_data={"question":question,"answer":answer}
            array.append(dict_data)
            print(array)
            
        return Response({"message":"scrap data successfully","status":"200","Data":array})
#mobile technology secand part
class MobileAppDevelopementView(APIView):
      def post(self, request, format=None):
            topic_id=request.data.get('topic_id')
            if not topic_id:
                return Response({"message":"topic_id is required"})

            url= "https://splitmetrics.com/blog/mobile-trends-for-2022/"
            response = requests.get(url)
            html_content = response.content
            soup = BeautifulSoup(html_content, "html.parser")
            h3_tags = soup.find_all("h3")
            # count = 0
            # data_count=0
            for h3 in h3_tags:
                      
                 question=h3.text
                 dot_index = question.find('.')
                 if dot_index != -1:
                    question = question[dot_index+1:].strip()
                 print(question)
                 answer=h3.find_next("p").text
                 print(answer)
                 topic = Topic.objects.get(id= topic_id)
                 topic.topic = topic
                 mobiledata=QuestionAndAnswr.objects.create(topic=topic,question=question,answer=answer)
                 serializer=QuestionAndAnswrSerializer(data=mobiledata)
                 mobiledata.save()
            return Response({"message":"scrap data successfully","status":"200"})


#technology
class EmergingTechnologyView(APIView):
    def post(self, request, format=None):
        topic_id=request.data.get('topic_id')
        if not topic_id:
            return Response({"message":"topic_id is required"})
        page = requests.get("https://www.ishir.com/blog/55810/top-15-emerging-technology-trends-to-watch-in-2023-and-beyond.htm")
        soup = BeautifulSoup(page.content, "html.parser")
        h3_tags = soup.find_all('h3')
        array=[]
        pattern = r'^\d{1,2}\b'
        for h3 in h3_tags:
            h3_text = h3.text.strip()
            question=h3_text
            question= question.replace(".", "")
            question = re.sub(pattern, '', question)

            # find the first paragraph tag after the h3 tag and extract its text
            paragraph = h3.find_next('p')
            if paragraph is not None:
                paragraph_text = paragraph.text.strip()
                answer=paragraph_text
            topic = Topic.objects.get(id= topic_id)
            topic.topic = topic
            scrappy=QuestionAndAnswr.objects.create(topic=topic,question=question,answer=answer)
            serializers=QuestionAndAnswrSerializer(data=scrappy)
            dict_data={"question":question,"answer":answer}
            array.append(dict_data)
        return Response({"message":"scrap data successfully","status":"200","data":array})


#football question and answer api
class FootballScrapingView(APIView):
      def post(self, request, format=None):
            topic_id=request.data.get('topic_id')
            if not topic_id:
             return Response({"message":"topic_id is required"})
            url = "https://questionsgems.com/football-quiz-questions/"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
            response = requests.get(url,headers=headers)

            soup = BeautifulSoup(response.text, "html.parser")

            question_elements = soup.find_all('p')

            for count, x in enumerate(question_elements[1:150], start=2):
    
                question = x.text.strip()
                answer_index = question.find("Answer:")
                if answer_index != -1:
                    answer = question[answer_index + len("Answer:"):].strip()
                    question = question[:answer_index].strip()
                    print("q",question)
                    print("a",answer)
                    count+=1
                    topic = Topic.objects.get(id= topic_id)
                    topic.topic = topic
                    football_data=QuestionAndAnswr.objects.create(topic=topic,question=question,answer=answer)
                    serializer=QuestionAndAnswrSerializer(data=football_data)
                    football_data.save()
            return Response({"message":"scrap data successfully","status":"200"})

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
                    doc = nlp(question.title())
                    merged_text = []
                    label_found = False  
                    for word in doc.ents:
                        if word.label_ in ["GPE", "ORG", "LOC", "PERSON", "MONEY", "ORDINAL", "PRODUCT", "NORP", "FAC", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "PERCENT", "QUANTITY", "CARDINALS"]:
                            merged_text.append(word.text.title())
                            label_found=True
                    label = " ".join(merged_text)
                    if not label_found or label.strip() == "":
                        alternative_label=[]
                        for token in doc:
                            if token.pos_ == "PROPN":
                                alternative_label.append(token.text.title())
                            if token.pos_ == "NNP":
                                alternative_label.append(token.text.title())
                            if token.pos_ == "VERB":
                                alternative_label.append(token.text.title())
                        if alternative_label:
                            label = " ".join(alternative_label)
                    else:
                        label="No Label Found"
                    response_data["QA_Pairs"].append({
                        "Question": question.strip(),
                        "Answer": answer.strip(),
                        "Label": label.strip()})
                return Response(response_data)
    
class GetLabelByUser_id(APIView):
    def get(self, request, user_id):
            labels = User_Label.objects.filter(user_id=user_id).values_list('Label', flat=True).order_by("-id")
            print('------------------------->>>>',labels)
            if labels:
                return Response({'labels': list(labels)})
            else:
              return Response({'error': 'User Label does not exist'})

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
        pdffile_name = pdffile.name
        pdffile_data=User_PDF.objects.create(pdf=pdffile,pdf_filename=pdffile_name)
        serializer=User_PDFSerializer(data=pdffile_data)
        pdffile_data.save()
        print("---->",pdffile_name)
        full_url = urljoin(url, pdffile_data.pdf.name)
        pdf_path=pdffile_data.pdf.path
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
            doc = nlp(question.title())
            merged_text = []
            label_found = False  
            for word in doc.ents:
                if word.label_ in ["GPE", "ORG", "LOC", "PERSON", "MONEY", "ORDINAL", "PRODUCT", "NORP", "FAC", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "PERCENT", "QUANTITY", "CARDINALS"]:
                    merged_text.append(word.text.title())
                    label_found=True
            label = " ".join(merged_text)
            if not label_found or label.strip() == "":
                alternative_label=[]
                for token in doc:
                    if token.pos_ == "PROPN":
                        alternative_label.append(token.text.title())
                    if token.pos_ == "NNP":
                        alternative_label.append(token.text.title())
                    if token.pos_ == "VERB":
                        alternative_label.append(token.text.title())
                    
                if alternative_label:
                    label = " ".join(alternative_label)
            else:
                label="No Label Found"
            response_data["QA_Pairs"].append({
                "Question": question.strip(),
                "Answer": answer.strip(),
                "Label": label.strip()})
        return Response(response_data)
    
class GetAllPdf(APIView):
    def get(self, request, format=None):
            pdf_list = User_PDF.objects.all().values('pdf_filename').order_by('-id')
            if pdf_list:
                return Response({'labels': list(pdf_list)})
            else:
                return Response({'data':"Pdf Does Not Exist"})

class GetALLUrls(APIView):
    def get(self, request, format=None):
            url_list = UrlTable.objects.all().values('url').order_by('-id')
            if url_list:
                return Response({'labels': list(url_list)})
            else:
                return Response({'data':"Url Does Not Found"})
            
            
class SaveQuestionAnswer(APIView):
    def post(self,request, format=None):
        response=request.data.get("Response")
        # response=[{'question': 'What is the actual Team Roster?', 'answer': "Smc League. Failure Or Refusal To Sig N Smc Liability Waiver Form Shall Result In The  Player Not Being Allowed To Participate In League. Any Player Found To Be Playing Without  Signing The Liability Waiver Shall Be Immediately Suspended From That Match And May  Only Return To Play Upon Signing T He Liability Waiver Following That Match. There Are No  Exceptions For Failure To Agree To Waive Liability. Player Must Also Be Sure To Sign The  Appropriate Team'S Waiver Or Could Be Ruled Ineligible. Waiver Forms Are Available At The  Field Or Information T Able.   Note: Your Team'S Waiver Of Liability Form Is The Actual Team Roster. Submitted  Registration Rosters Are Not Considered Official Until Each Player Has Signed The Waiver Of  Liability And Participated In League Play.   All Players Must Be 19 Years Of A Ge Or Older - Picture Id'S Must Be Produced Upon Request  Of Referee Or League Official. Failure To Produce Accurate Picture Id Upon Request Shall  Result In Removal Of Player From Match Play Until Such Time As Proof Of Age/Identity Can  Be Verified.   To Be E Ligible For Playoffs, All Players Must Have Participated In A Minimum Of Two  Week'S Matches.", 'label': 'Team Roster'}, {'question': 'What is the minimum number of Forfeit Points to be Awarded For Every Of The Following Time Limits?', 'answer': 'Following Week. A Game Forfeit Will Automatically Score The Offending Team In The  Standings As -3 Standing Points, 0 -1 Game, And 0 -50 Points. Although Smc Does Not Have  Any Monetary Forfeit Penalties, Any Team That Forfeits Three Regular Season Matches For Any  Reason Shall Automatically Be Removed From Playoff Contention.   Forfeited Points Will Be Start To Be Declared If There Are Less Than The Required Number Of  Rostered/Registered Players Available To Start The Match. Seven Forfe It Points Will Be  Awarded For Every Of The Following Time Limits:', 'label': 'Forfeit Awarded Time Limits'}, {'question': 'What is an Extra Point Kicked From The Three -Yard Line Will Add One Point?', 'answer': 'Points. Field Goals (Where Available) Will Count As Three (3) Points. Intercepted Or   Recovered Fumbles Of Extra Point Attempts Returned For Score Will Count As Two (2) Points   For The Defense. For Extra Point Conversions:     Mens: An Extra Point Kicked From The Three -Yard Line Will Add One Point. An  Extra Point “Play” From The Three -Yard Line Will Add One Point. An Extra Point  “Play” From The 10 -Yard Line Will Add Two Points.  Coed : An Extra Point Kicked From The Three -Yard Line Will Add One Point. An  Extra Point “Play” From The Three -Yard Line Will Add One Point. An Extra Point  Executed From The Same Spot With A Female Participant (Qb, Receiver, Rusher) Will  Add Two Points. An Extr A Point Executed From The Same Spot With A Female  Participant (Qb, Receiver, Rusher) Will Add Three Points.', 'label': 'No Label Found'}, {'question': 'What is the name of the phrase that marks the Penalty at the point of infraction?', 'answer': 'Result In An Automatic Penalty With The Ball Marked At The Point Of Infraction Unless The  Pass Is Less Than The Penalty: Meaning --If A Pass Ex Ceeds Ten Yards And There Is Pass  Interference, The Penalty Is Marked At The Spot Of Foul With Automatic First Down -- If Pass  Interference Is Called Less Than Ten Yards From Line Of Scrimmage, The Penalty Is Marked 10  Yards From Line Of Scrimmage With Auto Matic First Down. Pass Interference In The End  Zone Will Result In A New First Down On The One -Yard Line. Offensive Pass Interference Will  Result In A 10 -Yard Penalty From The Line Of Scrimmage.', 'label': 'Phrase Marks Penalty Infraction'}]
        for data in response:
            question=data.get('question')
            answer=data.get('answer')
            label=data.get('label')
            if not question or not answer or not label:
                return Response({"message":"Data is Not Found"})
            if not Topic.objects.filter(Topic = label).exists():
                topic_save=Topic.objects.create(Topic=label)
                topic_save.save()
            topic_id= Topic.objects.filter(Topic=label).values("id")
            topic_id = topic_id[0]['id']
            technology_table=QuestionAndAnswr.objects.create(question=question,answer=answer,topic_id=topic_id)
        return Response({"message":"Data Save Sucessfully"})
        
            
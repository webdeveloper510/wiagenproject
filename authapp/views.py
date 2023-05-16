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
from django.contrib.auth import login
import spacy
import nltk
nlp = spacy.load("en_core_web_sm")
nltk.download('stopwords')
# path=os.path.abspath("src/examplefile.txt")


stemmer=PorterStemmer()
import nltk
openai.api_key=settings.API_KEY

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
    model_path="/home/deepika/Desktop/Deepika/wiagenproject/authapp/saved_file/saved_model/classification_model.json"
    model_weight_path="/home/deepika/Desktop/Deepika/wiagenproject/authapp/saved_file/saved_model/classification_model_weights.h5"
    
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
        prompt=f"Auto Response Generator \n\nUser: {input} \n\nAI:\n",
        temperature=1,
        max_tokens=300,
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
        print('Result_-------------------------------------->>>>',result)
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
        print("Similarity Score",similarity_percentage)
        if (similarity_percentage)>=70:
            answer = filter_data[max_sim_index]['answer']
            userLabel_data=User_Label.objects.create(user_id=user_id,Label=result)
            return Response({"Label":result,"Answer":answer})
        else:
            input=user_input
            doc = nlp(input)
            # Merge consecutive NOUN tokens
            merged_text = []
            for token in doc:
                if token.pos_ == "NOUN":
                    merged_text.append(token.text.title())
            sentence = " ".join(merged_text)
            label=sentence
            userLabel_data=User_Label.objects.create(user_id=user_id,Label=label)
            response=self.chatgpt(input)
            return Response({"Label":label,"Result ":response})


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
    def post(self, request, format=None):
        url = request.data.get("url")
        if not url:
            return Response({"message":"url is required"},status=status.HTTP_400_BAD_REQUEST)
        
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
        
        response = requests.get(url,headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        first_paragraph = soup.find('p')
        
        if first_paragraph:
            return Response({"data": first_paragraph.text.strip()}, status=status.HTTP_200_OK)
        
        first_image = soup.find('img', alt=True)
        
        if first_image:
            return Response({"data": first_image['alt']}, status=status.HTTP_200_OK)
        
        h1_tag = soup.find('h1')
        
        if h1_tag:
            return Response({"data": h1_tag.text.strip()}, status=status.HTTP_200_OK)
        else:
         return Response({"message": "No data found."}, status=status.HTTP_404_NOT_FOUND)
       

# class GetLabelByUser_id(APIView):
#     def post(self, request, format=None):
#         user_id=request.data.get('user_id')
#         if not user_id:
#             return Response({'message':'user_id Required'},status=status.HTTP_400_BAD_REQUEST)
#         if not User_Label.objects.filter(user_id=user_id).exists():
#             return Response({'message':'user_id does not exist'},status=status.HTTP_400_BAD_REQUEST)
#         else:
#             user_label= User_Label.objects.all().order_by('id')
#             serializer = User_LabelSerializer(user_label, many=True)
#             array=[]
#             for x in serializer.data:
#                 user_id=(x['user_id'])
#                 Label= (x['Label'])
#                 if user_id==user_id:
#                     dict_data={"label":Label}
#                     array.append(dict_data)
#                     print(array)
#         return Response({'message':'success','data':array},status=status.HTTP_200_OK)
    
class GetLabelByUser_id(APIView):
    def get(self, request, user_id):
    
            labels = User_Label.objects.filter(user_id=user_id).values_list('Label', flat=True)

            if labels:
                return Response({'labels': list(labels)})
            
            else:
              return Response({'error': 'User Label does not exist'})

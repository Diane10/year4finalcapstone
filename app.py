import os
import base64
import streamlit as st
 
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')
import seaborn as sns 
import streamlit as st 
from sklearn.ensemble import  AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,plot_confusion_matrix,plot_roc_curve,precision_score,recall_score,precision_recall_curve,roc_auc_score,auc
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
import base64
from textblob import TextBlob 
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from textblob import Word
from textblob import TextBlob
import nltk
import nltk
nltk.download('punkt')
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import   WordNetLemmatizer
nltk.download("wordnet")
nltk.download("brown")
nltk.download('averaged_perceptron_tagger') 
from nltk.corpus import wordnet 
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import streamlit as st
import pandas as pd

model = tf.keras.models.load_model("model.sav")
#Nlp
 
wordnet_lemmatizer=WordNetLemmatizer() 
def sumy_summarize(docx):
    parser = PlaintextParser.from_string(docx,Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document,3)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result
    
def pos_tagger(nltk_tag): 
    if nltk_tag.startswith('J'): 
        return wordnet.ADJ 
    elif nltk_tag.startswith('V'): 
        return wordnet.VERB 
    elif nltk_tag.startswith('N'): 
        return wordnet.NOUN 
    elif nltk_tag.startswith('R'): 
        return wordnet.ADV 
    else:           
        return None    
 
def predict_object(image_file):
    image = Image.open(image_file) 
    image = image.resize((32,32),Image.ANTIALIAS)
    img_array = np.asarray(image, dtype='int32')
    img_array = img_array.reshape(1, 32, 32, 3)
    prediction = model.predict(img_array)
    obj = np.argmax(prediction, axis=None, out=None)
    return obj
try:
    from enum import Enum
    from io import BytesIO, StringIO
    from typing import Union
 
    import pandas as pd
    import streamlit as st
except Exception as e:
    print(e)
if st.sidebar.checkbox('NLP'):
  st.subheader("Natural Language Processing")
  message =st.text_area("Enter text")
  blob = TextBlob(message)
  if st.checkbox('Noun phrases'):
      if st.button("Analyse",key="1"):
          blob = TextBlob(message)
          st.write(blob.noun_phrases)
  if st.checkbox("show sentiment analysis"):  
      if st.button("Analyse",key="2"):
          blob = TextBlob(message)
          result_sentiment= blob.sentiment
          st.success(result_sentiment)
          polarity = blob.polarity
          subjectivity = blob.subjectivity
          st.write(polarity, subjectivity)
  if st.checkbox("show words"): 
      if st.button("Analyse",key="3"):
          blob = TextBlob(message)
          st.write (blob.words)
  if st.checkbox("show sentence"):
      if st.button("Analyse",key='30'):
          blob = TextBlob(message)
          st.write(blob.sentences)
  if st.checkbox("Tokenize sentence"): 
      if st.button("Analyse",key='27'):
          list2 = nltk.word_tokenize(message) 
          st.write(list2) 
  if st.checkbox("POS tag "): 
      if st.button("Analyse",key='20'):
          pos_tagged = nltk.pos_tag(nltk.word_tokenize(message))   
          st.write(pos_tagged) 
    
        
  if st.checkbox("Text preprocessing"):
      selection = st.selectbox("Select type:", ("Lemmatizer", "PorterStemmer"))
      if st.button("Analyse",key="4"):
          if selection == "Lemmatizer": 
              tokenization=nltk.word_tokenize(message)
              for w in tokenization:
              
                  st.write("Lemma for {} is {}".format(w,wordnet_lemmatizer.lemmatize(w))) 
                            
    
          elif selection == "PorterStemmer":
              porter_stemmer=PorterStemmer()
              tokenization=nltk.word_tokenize(message)
              for w in tokenization:
                  st.write("Stemming for {} is {}".format(w,porter_stemmer.stem(w)))   
              
            
  if st.checkbox("show text summarization"):
      if st.button("Analyse",key="5"):
          st.subheader("summarize your text")
          summary_result= sumy_summarize(message)
          st.success(summary_result)
      
  if st.checkbox("splelling checker"):
      if st.button("Analyse",key="6"):
          blob = TextBlob(message)
          st.write(blob.correct())
  if st.checkbox("language detector"):
      if st.button("Analyse",key="15"):
          blob = TextBlob(message)
          st.write(blob.detect_language())

  if st.checkbox("Translate sentences"):
      selection = st.selectbox("Select language:", ("French", "Spanish","Chinese"))
  
      if st.button("Analyse",key='23'):
          if selection == "French":
              blob = TextBlob(message)
              translated=blob.translate(to="fr")
              st.write(translated)
              
          if selection == "Spanish":
              blob = TextBlob(message)
              translated=blob.translate(to='es')
              st.write(translated)
#                 
          if selection == "Chinese":
              blob = TextBlob(message)
              translated=blob.translate(to="zh")
              st.write(translated)
if st.sidebar.checkbox('computer vision'):
  st.subheader("Welcome to the object detector program") 
    
  st.markdown("Please enter the image file for recognition such as aeroplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck")
    
  uploaded_file = st.file_uploader("Choose an image...", type="jpg")
  result = ""
  r = ""
  if st.button("Predict"):
      result = predict_object(uploaded_file)
      if result == 0:
          r = 'aeroplane'

      elif result == 1:
          r = 'automobile'
            
      elif result == 2:
          r = 'bird'
            
      elif result == 3:
          r = 'cat'
            
      elif result == 4:
          r = 'deer'
            
      elif result == 5:
          r = 'dog'
            
      elif result == 6:
          r = 'frog'
            
      elif result == 7:
          r = 'horse'
            
      elif result == 8:
          r = 'ship'
          
      elif result ==  9:
          r = 'truck'
  
  st.success('The object detected is: {}'.format(r))

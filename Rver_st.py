import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib.colors as mcolors
from scipy.stats import ttest_ind
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
import re
import seaborn as sns


# Download necessary NLTK data
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Load Data
nyt_data = pd.read_csv("NT.csv")  
wapo_data = pd.read_csv("WT.csv")   
wsj_data = pd.read_csv("WS.csv")   

data = pd.concat([
    nyt_data.assign(source="nyt"),
    wapo_data.assign(source="wapo"),
    wsj_data.assign(source="wsj")
])

# Define stopwords
stop_words = set(stopwords.words('english')).union(ENGLISH_STOP_WORDS)

# Preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'\W+', ' ', text.lower())  # Remove non-word characters and lowercase
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return ' '.join(tokens)

# Apply preprocessing to create processed_text
data['processed_text'] = data['text'].apply(preprocess_text)

##################################################################
#### Title #######################################################
##################################################################
st.title("Sentiment Analysis of Major US Newspapers on COVID-19 During Election Period")
st.caption("This page was created by Seizu HSU Yi-Ju")
#container = st.container(border=False)
#container.caption("This page was created by Seizu HSU Yi-Ju")


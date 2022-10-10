
import streamlit as st
from sklearn.metrics import *
from google.oauth2 import service_account

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import pandas as pd
import sqlite3
import pandas as pd
import sqlite3
import pandas as pd
import nltk
import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import collections
import plotly.graph_objs as go
from plotly import tools,subplots
import plotly.offline as py


from PIL import Image
pos_im = Image.open('newplot.png')
neg_im = Image.open('newplot (4).png')
all_im = Image.open('newplot (3).png')
sentiment = Image.open('sentiment.png')




data1 = pd.read_csv('disneyland_clean1.csv')


header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()





def vectorize(text, maxx_features):
    
    vectorizer = TfidfVectorizer(max_features=maxx_features)
    X = vectorizer.fit_transform(text)
    return X

stop_words=set(nltk.corpus.stopwords.words('english'))
vect =TfidfVectorizer(stop_words=stop_words,max_features=4000) #changed 1000 to 2000


from sklearn.decomposition import TruncatedSVD

lsa_model = TruncatedSVD(n_components=20, algorithm='randomized', n_iter=10, random_state=42)
lsa_model_neg = TruncatedSVD(n_components=20, algorithm='randomized', n_iter=10, random_state=42)
lsa_model_pos = TruncatedSVD(n_components=20, algorithm='randomized', n_iter=10, random_state=42)

lsa_topic_neg=lsa_model_neg.fit_transform(vect.fit_transform(data1['Review_Text_Clean'][data1['Analysis']== 'Negative']))
lsa_topic_pos=lsa_model_pos.fit_transform(vect.fit_transform(data1['Review_Text_Clean'][data1['Analysis']== 'Positive']))


vocab = vect.get_feature_names_out()


def draw_word_cloud_pos(index):
    imp_words_topic=""
    comp=lsa_model_pos.components_[index]
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:30]
    for word in sorted_words:
        imp_words_topic=imp_words_topic+" "+word[0]

    wordcloud = WordCloud(width=400, height=400).generate(imp_words_topic)
    return wordcloud

def draw_word_cloud_neg(index):
    imp_words_topic=""
    comp=lsa_model_neg.components_[index]
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:30]
    for word in sorted_words:
        imp_words_topic=imp_words_topic+" "+word[0]

    wordcloud = WordCloud(width=400, height=400).generate(imp_words_topic)
    return wordcloud


condition= 0
origin_list = [condition] + sorted(data1['y'].unique())
default_value_route = ""
default_value_dest_city = ""


origin_choice = st.sidebar.selectbox('Please choose a topic:  ', origin_list, index=0)



with header:
    st.title('Wordcloud for Positive and Negative Reviews')

    st.subheader('WordClouds can be used to understand he various words associated with a given topic. For this model I used TfidfVectorizer in order to reduce the dimentionality of the corpus, as well as SVD. Previous analysis indicated that 20 topics would best describe this corpus. Please select a topic on the left.')

#with dataset:

    st.subheader('you picked topic ' + str(origin_choice) + ', the associated POSITIVE words for this topic can be found below:')

    wordcloud = draw_word_cloud_pos(origin_choice)
    fig, ax = plt.subplots(figsize = (4, 4))
    ax.imshow(wordcloud)
    plt.axis("off")
    st.pyplot(fig)

# with header:
#     st.title('Wordcloud for Negative Reviews')

#     st.subheader('See below to observe the wordcloud for the specified topic.')

#with dataset:

    st.subheader('you picked topic ' + str(origin_choice) + ', the associated NEGATIVE words for this topic can be found below:')

    wordcloud = draw_word_cloud_neg(origin_choice)
    fig, ax = plt.subplots(figsize = (4, 4))
    ax.imshow(wordcloud)
    plt.axis("off")
    st.pyplot(fig)

with header:
    st.title('Disneyland Review Sentiment Analysis.')
    st.subheader('The Majority of reviews for Disneyland reflected a positive sentiment (found using sentiment analysis TextBlob package).')
    st.image(sentiment, caption='Sentiment Distribution')

with header:
    st.title('Disneyland Review NGrams')

    st.subheader('All Reviews NGram.')
    st.image(all_im, caption='All Reviews')

    st.subheader('Positive Reviews NGram.')
    st.image(pos_im, caption='Positive Reviews')

    st.subheader('Negative Reviews NGram.')
    st.image(neg_im, caption='Negative Reviews')

with header:
    st.title('Key Analysis Takeaways.')
    st.subheader('After obseving the sentiment  analysis it appears that although the overwhelming majority of reviews are possitive, theme goers do have complaints about the volume of people at the park, the lines and wait times. One thing Disneyland can consider is expanding some parks and improving upon their fast pass system.')










# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 22:00:16 2020

@author: ritup
"""


################### Q1. Importing Data #######################

import pandas as pd

data = pd.read_csv("OneDrive\\Desktop\\Simplilearn\\NLP\\Projects\\Review Analysis\\K8 Reviews v0.2.csv")
data.head(10)
data.shape
data.columns
# Out[3]: Index(['sentiment', 'review'], dtype='object')

data.sentiment.value_counts()
# Out[4]: 
# 0    7712
# 1    6963
# Name: sentiment, dtype: int64
# Well balanced dataset

################## Q2-Q7. Normalize, Remove Punctuation, POS tags, Lemmatize #######################

import numpy as np
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

review_text = np.array(data['review'])
text = review_text[550]
type(review_text)

def text_preprocessing(text):
    rem_punct = [chars for chars in text if chars not in string.punctuation]
    clean_text = "".join(rem_punct)
    
    clean_text = clean_text.lower()
    
    word_token = word_tokenize(clean_text) 
    no_stopwords = [words for words in word_token if words not in stopwords.words('english')]
    
    lmtz = WordNetLemmatizer()
    lemmatize_words = []
    for each_word in no_stopwords:
        lemmatize_words.append(lmtz.lemmatize(each_word))
       
    pos_tags = nltk.pos_tag(lemmatize_words)
    
    nouns = ['NN', 'NNS', 'NNP', 'NNPS']
    processed_text = [each_tuple[0] for each_tuple in pos_tags if each_tuple[1] in nouns]
    
    return(processed_text)

processed_text = text_preprocessing(text)
processed_text
   
###################### Q8. LDA Model ###############################

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as lda


cv = CountVectorizer(analyzer = text_preprocessing)
bag_of_words = cv.fit_transform(review_text)    

dict = cv.vocabulary_
print(cv.vocabulary_)   
 
lda_init = lda(n_components = 12)
lda_model = lda_init.fit(bag_of_words)

lda_model.components_
top_terms = 20

def print_topics(model, count_vectorizer, n_top_words):
    lst = []
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        str1 = (" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words-1:-1]]))
        print(str1)
        lst.append(str1)
    return(lst)
        
top_topics = print_topics(lda_model, cv, top_terms)

#Topic #0:
#problem note k8 heating charger lenovo month excellent phone turbo service
#display warranty star k4 cable time screen use k5
#
#Topic #1:
#network camera smartphone sim jio work speed ram sensor issue function feature 
#bill quality finger backup print gb memory battery
#
#Topic #2:
#device call option screen feature note issue cast lenovo quality video k8 photo 
#music volta gallery camera button worth doesnt
#
#Topic #3:
#battery product price waste amazon life day camera money item quality drain time 
#thing service purchase return dont work customer
#
#Topic #4:
#phone dont hai h day ho problem plz hi pls bhi call glass purchase buy ka use lenovo 
#bahut ki
#
#Topic #5:
#ok expectation call earphone buy day return card phone recorder issue camera number 
#use policy service slot time help sim
#
#Topic #6:
#battery phone camera backup issue charge hour heat time use drain day mode quality 
#get problem game processor performance charger
#
#Topic #7:
#product super quality gud return camera cash screen hai v phn photo nd display look 
#build suggestion rate get ke
#
#Topic #8:
#camera quality money speaker sound value everything dolby headphone clarity box mode 
#picture depth image average front mp piece atmos
#
#Topic #9:
#phone price time issue lenovo service amazon range buy month feature budget dont 
#day money work problem return experience note
#
#Topic #10:
#mobile product issue lenovo superb glass experience awesome network gorilla please 
#dont delivery buy screen support problem day customer service
#
#Topic #11:
#camera performance feature battery handset good mark look love rear need nice condition 
#cost wise cool work front model improvement

from gensim.test.utils import common_corpus, common_dictionary
from gensim.models import CoherenceModel

# Compute Coherence Score
coherence_model_lda = CoherenceModel(topics = top_topics, text = review_text, 
                                     dictionary = dict, coherence='c_v')

coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


















































import pandas as pd
import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.probability import FreqDist,ConditionalFreqDist
from nltk.util import bigrams
import math


#---1.0---

df_corpus = pd.read_csv("./Projet 1/train.csv")
#df_corpus = pd.read_csv("/course/common/student/P1/train.csv")

def remove_html_tags(text):

  import re
  clean = re.compile('<.*?>')
  return re.sub(clean, '', text)

df_corpus['Text'] = df_corpus['Text'].apply(lambda x: remove_html_tags(x))
df_corpus['Tokens'] = df_corpus['Text'].apply(lambda x: WordPunctTokenizer().tokenize(x))

# Update df_corpus['Tokens'] to complete the preprocessing as described above
to_skip = ['(', ')', '[', ']', '{', '}', ':', ';', '=', '-', '/', '\\', '"', "'"]

def preprocess_tokens(tokens):
    result = []
    for token in tokens:
        for char in to_skip:
            token = token.replace(char, '') #on nettoie le token des éléments indésirables
        
        if token == '': #si un token devient vide, on skip
            continue
        
        if token == '&': #on remplace le & par and
            token = 'and'
        
        result.append(token)
    
    return result

df_corpus['Tokens'] = df_corpus['Tokens'].apply(preprocess_tokens)
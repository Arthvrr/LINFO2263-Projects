#https://www.nltk.org
#https://www.nltk.org/api/nltk.tokenize.regexp.html

"""
The code below can be used for both train.csv and test.csv. In addition, please note that once executed, the df_corpus Dataframe will have 3 columns:

Column "Text" contains the raw text of the question
Column "Tokens" contains the tokenized and preprocessed text of the question
Column "Y" contains the labels later used for classification
"""

import pandas as pd
import nltk
from nltk.tokenize import WordPunctTokenizer

df_corpus = pd.read_csv("/path/to/corpus.csv")

def remove_html_tags(text):

  import re
  clean = re.compile('<.*?>')
  return re.sub(clean, '', text)

df_corpus['Text'] = df_corpus['Text'].apply(lambda x: remove_html_tags(x))
df_corpus['Tokens'] = df_corpus['Text'].apply(lambda x: WordPunctTokenizer().tokenize(x))

# Update df_corpus['Tokens'] to complete the preprocessing as described above
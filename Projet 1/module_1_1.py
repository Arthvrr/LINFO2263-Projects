import pandas as pd
import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.probability import FreqDist,ConditionalFreqDist
from nltk.util import bigrams
import math
#from module_1_0 import df_corpus

#IMPORT MODULE_1_0
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



#---1.1---

all_tokens = []
for tokens in df_corpus['Tokens']: #on met tout dans une liste applatie
    for token in tokens:
        all_tokens.append(token)

f_dist = FreqDist(all_tokens) #compte les occurences de chaque mot avec FreqDist

ten_occ = []
for word,occurence in f_dist.items(): #https://www.w3schools.com/python/python_dictionaries_loop.asp
    if occurence >= 10:
        ten_occ.append((word,occurence)) #si occurence >= 10, on ajoute à la liste ten_occ sous forme de tuple
    else:
        continue #sinon on skip

sorted_ten_occ = ten_occ.copy() #copie de la liste
sorted_ten_occ.sort(key=lambda x: (-x[1],x[0])) #trie la liste de tuples à partir du nombre d'occurences (2ème élément du tuple) --> https://www.geeksforgeeks.org/python/python-program-to-sort-a-list-of-tuples-by-second-item/

top_10 = sorted_ten_occ[:10] #on ne garde que les 10 premiers éléments

print(top_10)

vocab_set = set() #ensemble des mots
n_unk = 0

for occ in ten_occ:
    vocab_set.add(occ[0]) #on ajoute tout les mots dont l'occurence >= 10 au set de vocabulaire

for token in all_tokens:
    if token in vocab_set:
        continue
    else: #si mot pas dans le set, +1 <UNK>
        n_unk += 1

oov_rate = round((n_unk/len(all_tokens))*100,3) #formule du OOV_Rate arrondie à 3 décimales #https://www.askpython.com/python/string/print-format-3f
print(oov_rate)
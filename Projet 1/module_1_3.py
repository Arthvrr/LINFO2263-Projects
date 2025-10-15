import pandas as pd
import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.probability import FreqDist,ConditionalFreqDist
from nltk.util import bigrams
import math
#from module_1_0 import df_corpus
#from module_1_1 import vocab_set
#from module_1_2 import df_test,V

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

#IMPORT MODULE_1_1
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

vocab_set = set() #ensemble des mots

for occ in ten_occ:
    vocab_set.add(occ[0]) #on ajoute tout les mots dont l'occurence >= 10 au set de vocabulaire

#IMPORT MODULE 1_2
V = len(set(all_tokens)) #nombre total de mots uniques

df_test = pd.read_csv("./Projet 1/test.csv")
#df_test = pd.read_csv("/course/common/student/P1/test.csv")

df_test['Text'] = df_test['Text'].apply(lambda x: remove_html_tags(x))
df_test['Tokens'] = df_test['Text'].apply(lambda x: WordPunctTokenizer().tokenize(x))
df_test['Tokens'] = df_test['Tokens'].apply(preprocess_tokens)

vocab_set.add('<UNK>')



#---1.3---

df_HQ = df_corpus[df_corpus['Y'] == 'HQ']
df_LQ = df_corpus[df_corpus['Y'] == 'LQ']

p_HQ = len(df_HQ) / len(df_corpus) #probabilités d'être une question HQ ou LQ
p_LQ = len(df_LQ) / len(df_corpus)

tokens_HQ = {}
tokens_LQ = {}

for tokens in df_HQ['Tokens']: #comptage des tokens dans HQ
    for token in tokens:
        if token in vocab_set:  #on garde seulement les mots du vocabulaire filtré
            if token in tokens_HQ:
                tokens_HQ[token] += 1 #si déjà dans le dico, on incrémente sa valeur, sinon on ajoute au dico avec valeur initiale de 1
            else:
                tokens_HQ[token] = 1

for tokens in df_LQ['Tokens']: #comptage des tokens dans LQ
    for token in tokens:
        if token in vocab_set:
            if token in tokens_LQ:
                tokens_LQ[token] += 1 #si déjà dans le dico, on incrémente sa valeur, sinon on ajoute au dico avec valeur initiale de 1
            else:
                tokens_LQ[token] = 1

n_words_HQ = 0
n_words_LQ = 0

for word,occurence in tokens_HQ.items(): #on compte le nombre de mots HQ dans le dico tokens_HQ
    n_words_HQ += occurence

for word,occurence in tokens_LQ.items(): #on compte le nombre de mots LQ dans le dico tokens_LQ
    n_words_LQ += occurence

proba_w_HQ = {}
proba_w_LQ = {}

for word in vocab_set: #pour chaque mot du vocabulaire, on calcule P(w|HQ) et P(w|LQ)
    if word not in tokens_HQ.keys():
        proba_word_HQ = (1) / (n_words_HQ + V) #si mot pas dans les clés du dico, valeur = 0
    else:
        proba_word_HQ = (tokens_HQ[word] + 1) / (n_words_HQ + V)
        
    if word not in tokens_LQ.keys(): #si mot pas dans les clés du dico, valeur = 0
        proba_word_LQ = (1) / (n_words_LQ + V)
    else:
        proba_word_LQ = (tokens_LQ[word] + 1) / (n_words_LQ + V)
    
    proba_w_HQ[word] = proba_word_HQ
    proba_w_LQ[word] = proba_word_LQ

#On ne prend pas en compte <s> et </s> dans Naive Bayes car on considère la question entière comme un “bag of words” (≠ séquence de bigrams)
tokens_test_clean = []
for tokens in df_test['Tokens']: #nettoyage du test set pour Naive Bayes
    temp_list = []
    for token in tokens: #même chose qu'en haut mais on n'ajoute pas les balises <s> et </s>
        if token not in vocab_set:
            temp_list.append('<UNK>')
        else:
            temp_list.append(token)
    tokens_test_clean.append(temp_list)

predictions = []

for question in tokens_test_clean:
    log_proba_HQ = 0
    log_proba_LQ = 0

    for word in question:
        log_proba_HQ += math.log(proba_w_HQ[word]) #log des probabilités
        log_proba_LQ += math.log(proba_w_LQ[word])
    
    log_proba_HQ += math.log(p_HQ) #ajout du log de probabilité à priori
    log_proba_LQ += math.log(p_LQ)

    if log_proba_HQ > log_proba_LQ: #question prédite comme HQ
        predictions.append('HQ')
    else:
        predictions.append('LQ') #question prédite comme LQ

good_prediction = 0 #on compare les prédictions du modèle et les vraies labels
for i in range(len(tokens_test_clean)):
    if predictions[i] == df_test['Y'][i]: #si bonne prédicittion du modèle
        good_prediction += 1

accuracy = round((good_prediction / len(predictions)) * 100,3) #formule de l'accuracy arrondie à 3 décimales
print(accuracy)
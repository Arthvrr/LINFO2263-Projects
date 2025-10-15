import pandas as pd
import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.probability import FreqDist,ConditionalFreqDist
from nltk.util import bigrams
import math

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



#---1.4--- Smoothed Bigrams Model

df_HQ = df_corpus[df_corpus['Y'] == 'HQ']
df_LQ = df_corpus[df_corpus['Y'] == 'LQ']

p_HQ = len(df_HQ) / len(df_corpus) #probabilités d'être une question HQ ou LQ
p_LQ = len(df_LQ) / len(df_corpus)

bigrams_HQ = [] #générer les bigrams pour HQ avec <UNK> + balises <s> </s>
for tokens in df_HQ['Tokens']:

    tokens_mod = []               
    tokens_mod.append('<s>') 
    for t in tokens:          
        if t in vocab_set: #si token est dans le vocabulaire, on l'ajoute sinon <UNK>
            tokens_mod.append(t)  
        else:                     
            tokens_mod.append('<UNK>')
    tokens_mod.append('</s>')

    bigrams_HQ.extend(list(bigrams(tokens_mod)))

bigrams_LQ = [] #générer les bigrams pour LQ avec <UNK> + balises <s> </s>
for tokens in df_LQ['Tokens']:
    
    tokens_mod = []               
    tokens_mod.append('<s>') 
    for t in tokens:          
        if t in vocab_set: #si token est dans le vocabulaire, on l'ajoute sinon <UNK>
            tokens_mod.append(t)  
        else:                     
            tokens_mod.append('<UNK>')
    tokens_mod.append('</s>')

    bigrams_LQ.extend(list(bigrams(tokens_mod)))

cf_HQ = ConditionalFreqDist(bigrams_HQ) #compte les bigrams avec ConditionalFreqDist
cf_LQ = ConditionalFreqDist(bigrams_LQ)


proba_HQ = {}
proba_LQ = {}

for w1 in cf_HQ: #calcul des probabilités des bigrams pour HQ
    total_w1 = sum(cf_HQ[w1].values())
    proba_HQ[w1] = {}
    for w2 in cf_HQ[w1]:
        proba_HQ[w1][w2] = (cf_HQ[w1][w2] + 1) / (total_w1 + V) #Laplace smothing

for w1 in cf_LQ: #calcul des probabilités des bigrams pour LQ
    total_w1 = sum(cf_LQ[w1].values())
    proba_LQ[w1] = {}
    for w2 in cf_LQ[w1]:
        proba_LQ[w1][w2] = (cf_LQ[w1][w2] + 1) / (total_w1 + V) #Laplace smothing


questions_test = []
for tokens in df_test['Tokens']:
    temp_list = []
    temp_list.append('<s>')
    for token in tokens:
        if token in vocab_set:  #le token est connu
            temp_list.append(token)
        else:               #le token est inconnu donc <UNK>
            temp_list.append('<UNK>')
    temp_list.append('</s>')
    questions_test.append(temp_list)

predictions = []

for tokens in questions_test:
    log_HQ = math.log(p_HQ)
    log_LQ = math.log(p_LQ) #log de probabilité à priori

    for w1, w2 in bigrams(tokens):

        if w1 in proba_HQ and w2 in proba_HQ[w1]: #si bigram w1,w2 déjà vu dans les questions HQ, sinon Laplace Smoothing pour pas donner proba 0
            log_HQ += math.log(proba_HQ[w1][w2])
        else:
            total_w1 = sum(cf_HQ[w1].values()) if w1 in cf_HQ else 0
            log_HQ += math.log(1 / (total_w1 + V))

        if w1 in proba_LQ and w2 in proba_LQ[w1]: #si bigram w1,w2 déjà vu dans les questions LQ, sinon Laplace Smoothing pour pas donner proba 0
            log_LQ += math.log(proba_LQ[w1][w2])
        else:
            total_w1 = sum(cf_LQ[w1].values()) if w1 in cf_LQ else 0
            log_LQ += math.log(1 / (total_w1 + V))

    if log_HQ > log_LQ: #question prédite comme HQ
        predictions.append('HQ')
    else:
        predictions.append('LQ') #question prédite comme LQ

good_prediction = 0 #on compare les prédictions du modèle et les vraies labels
for i in range(len(predictions)):
    if predictions[i] == df_test['Y'][i]: #si bonne prédicittion du modèle
        good_prediction += 1

accuracy = round((good_prediction / len(predictions)) * 100,3) #formule de l'accuracy arrondie à 3 décimales
print(accuracy)
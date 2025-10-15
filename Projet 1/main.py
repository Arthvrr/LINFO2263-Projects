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
from nltk.probability import FreqDist,ConditionalFreqDist
from nltk.util import bigrams
import math


#---1.0---

#df_corpus = pd.read_csv("./Projet 1/train.csv")
df_corpus = pd.read_csv("/course/common/student/P1/train.csv")

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


#---1.2---

questions = []
for tokens in df_corpus['Tokens']:
    temp_list = []
    temp_list.append('<s>') #ajout des balises <s> ... </s> entre chaque question de la colonne Tokens du df_corpus
    for token in tokens:
        temp_list.append(token)
    temp_list.append('</s>')
    questions.append(temp_list)

question_1106 = questions[1106] #question à l'index 1106
bigrams_question_1106 = list(bigrams(question_1106)) #listing des bigrams de la question à l'index 1006 à l'aide de bigrams() de nltk.util
print(bigrams_question_1106)

bigrams_questions = []
for question in questions:
    bigrams_one_question = list(bigrams(question)) #on stocke les bigrams de chaque question dans bigrams_questions
    bigrams_questions.extend(bigrams_one_question) #https://realpython.com/python-flatten-list/

cf_dist = ConditionalFreqDist(bigrams_questions) #compte les mots qui viennent le plus souvent après un mot avec ConditionalFreqDist

proba_bigrams = {}
for w1 in cf_dist:
    total_w1 = sum(cf_dist[w1].values())
    proba_bigrams[w1] = {}

    for w2 in cf_dist[w1]:
        proba_bigrams[w1][w2] = cf_dist[w1][w2] / total_w1 #P(w2|w1) 

sorted_proba_bigrams = sorted(proba_bigrams['<s>'].items(), key=lambda x: -x[1]) #on trie dans l'ordre décroissant proba_digrams pour ne garder que les 5 premières
top_5 = dict(sorted_proba_bigrams[:5]) #conversion en dictionnaire

for word,proba in top_5.items():
    top_5[word] = round(proba,3) #on arrondit à 3 décimales
print(top_5)

#---LAPLACE---
proba_bigrams_laplace = {}
V = len(set(all_tokens)) #nombre total de mots uniques
for w1 in cf_dist:
    total_w1 = sum(cf_dist[w1].values())
    proba_bigrams_laplace[w1] = {}

    for w2 in cf_dist[w1]:
        proba_bigrams_laplace[w1][w2] = (cf_dist[w1][w2] + 1) / (total_w1 + V) #P_laplace(w2|w1) 

sorted_proba_bigrams_laplace = sorted(proba_bigrams_laplace['<s>'].items(), key=lambda x: -x[1]) #on trie dans l'ordre décroissant proba_digrams_laplace pour ne garder que les 5 premières
top_5_laplace = dict(sorted_proba_bigrams_laplace[:5]) #conversion en dictionnaire

for word,proba in top_5_laplace.items():
    top_5_laplace[word] = round(proba,3) #on arrondit à 3 décimales
print(top_5_laplace)


#df_test = pd.read_csv("./Projet 1/test.csv")
df_test = pd.read_csv("/course/common/student/P1/test.csv")


df_test['Text'] = df_test['Text'].apply(lambda x: remove_html_tags(x))
df_test['Tokens'] = df_test['Text'].apply(lambda x: WordPunctTokenizer().tokenize(x))
df_test['Tokens'] = df_test['Tokens'].apply(preprocess_tokens)

vocab_set.add('<UNK>')  #ajouter <UNK> au vocab
questions_test = []
for tokens in df_test['Tokens']:
    temp_list = []
    temp_list.append('<s>') #ajout des balises <s> ... </s> entre chaque question de la colonne Tokens du def_test
    for token in tokens:
        if token not in vocab_set: #remplacer les mots oov par <UNK>
            temp_list.append('<UNK>')
        else:
            temp_list.append(token)
    temp_list.append('</s>')
    questions_test.append(temp_list)


#Calcul de la perplexité
log_prob_sum = 0
N = 0

for question_test in questions_test:
    bigrams_q = list(bigrams(question_test))
    for w1, w2 in bigrams_q:
        if w1 == '<s>':  #<s> n'est pas compté
            continue

        if w2 in cf_dist[w1]:
            count_w1_w2 = cf_dist[w1][w2]
        else:
            count_w1_w2 = 0
        
        if w1 in cf_dist:
            total_w1 = sum(cf_dist[w1].values())
        else:
            total_w1 = 0

        prob = (count_w1_w2 + 1) / (total_w1 + V)  #Laplace
        log_prob_sum += math.log(prob)
        N += 1

perplexity = round(math.exp(-log_prob_sum / N),3) #on arrondit à 3 décimales comme d'hab
print(perplexity)


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
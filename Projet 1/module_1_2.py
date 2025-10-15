import pandas as pd
import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.probability import FreqDist,ConditionalFreqDist
from nltk.util import bigrams
import math
#from module_1_0 import df_corpus,remove_html_tags,preprocess_tokens
#from module_1_1 import all_tokens,vocab_set

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


df_test = pd.read_csv("./Projet 1/test.csv")
#df_test = pd.read_csv("/course/common/student/P1/test.csv")


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


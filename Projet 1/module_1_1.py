import pandas as pd
import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.probability import FreqDist,ConditionalFreqDist
from nltk.util import bigrams
import math
from module_1_0 import df_corpus

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
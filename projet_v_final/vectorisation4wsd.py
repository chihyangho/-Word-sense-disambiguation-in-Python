#enconding: utf8
import numpy as np
from conll_text_manip import *

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import make_pipeline

from time import time

def get_w2v():
    print("Chargement des vecteurs français proposés par fasText https://fasttext.cc/docs/en/crawl-vectors.html")

    with open("embeds/fr_embeds.vec", "r", encoding="utf8") as doc:
        data = [line.split() for line in doc.readlines()][1:]

    w2v = {}
    for row in data:
        w2v[str(row[0])] = [float(x) for x in row[1:]]

    return w2v

def get_sum_Embs(corpus):
    """
    Retourne la matrice d'embeddings servant à la vectorisation des exemples
    Ne prend que les mots présents dans le corpus de classification
    Le seul modèle actuel est word2vec proposé par fastText avec 3 possibles dimensions (50, 100, 300)
    """
    w2v = get_w2v()
    list_v = [[w2v[str(w).lower()] for w in exemple if str(w).lower() in w2v.keys()] for exemple in corpus]
    X = [np.sum(exemple, axis=0) for exemple in list_v]
    return np.array(X)

def sklearn_Tfidf_B(corpus, ngram_r):
    v = TfidfVectorizer(analyzer="word", ngram_range=(1, ngram_r))
    X = v.fit_transform(corpus)
    return X.toarray()

# ======================================================================
# It's the part of handmade TF-IDF:
def nb_word_duplicate(sent):
    nb_word_duplicate = []
    sent = sent.lower()
    words = sent.split()
    for word in words:
        count = sent.count(word)
        count = count/len(words)
        pair = (word, count)
        nb_word_duplicate.append(pair)
    return nb_word_duplicate

# here the tf_dico shows IN ONE DOCUMENT, how many times a word appears in a document (sentence)
def tf_dico(corpus):
    tf_dico = {}
    i = 0
    for sent in corpus:
        tf_dico[i] = nb_word_duplicate(" ".join(sent))
        i += 1
    return tf_dico

def vocab(corpus):
    vocab = []
    for exemple in corpus:
        for sent in exemple:
            for word in sent.split():
                if word not in vocab:
                    vocab.append(word.lower())
    return list(set(vocab))

# here the idf_dico shows A WORD, how many time it appears in all documents
# (calculate once even if it appears more than one time ine a document)
def idf_dico(corpus, vocab):
    idf_dico = {}
    for index, voc in enumerate(vocab):
        word_count = 0
        for sent in corpus:
            if voc in sent:
                word_count += 1
        idf_dico[voc.lower()] = word_count
    # print(idf_dico)
    return idf_dico

def idf(tf_dico, idf_dico, corpus):
    idf_ = {}
    for i in range(0, len(tf_dico)):
        list_new_pairs = []
        for pair in tf_dico[i]:
            nb_idf = 0
            if idf_dico[pair[0]] > 0:
                nb_idf = np.log(len(tf_dico)/idf_dico[pair[0]])
            new_pair = (pair[0], nb_idf)
            list_new_pairs.append(new_pair)
        idf_[i] = list_new_pairs

    return idf_

def tf_idf(tf_dico, idf):
    tf_idf = {}
    for i in range(0, len(tf_dico)):
        tf_paires = tf_dico[i]
        idf_paire = idf[i]
        new_list = []
        for j in range(0, len(tf_paires)):
            new_list.append((tf_paires[j][0], tf_paires[j][1]*idf_paire[j][1]))
        tf_idf[i] = new_list
    return tf_idf

def tf_idf_matrix(tf_idf, vocab):
    tf_idf_matrix = np.zeros((len(tf_idf), len(vocab)))
    # print(tf_idf_matrix)
    for i in range(0, len(tf_idf)):
        for pair in tf_idf[i]:
            index_vocab = vocab.index(pair[0].lower())
            tf_idf_matrix[i][index_vocab] = pair[1]
    return tf_idf_matrix

def get_tfidf_matrix(corpus, vocab):
    tf_dic = tf_dico(corpus)
    idf_dic = idf_dico(corpus, vocab)
    idf_ = idf(tf_dic, idf_dic, corpus)
    tf_idf_ = tf_idf(tf_dic, idf_)
    return tf_idf_matrix(tf_idf_, vocab)

#enconding: utf8
import numpy as np
from conll_text_manip import *
from vectorisation4wsd import *

from time import time

def vectoriser(data, t_vector, n):
    t0 = time()
    vocab_ = vocab(data)
    X = []
    if t_vector == 'tf-idf':
        corpus = [" ".join(sample) for sample in data]
        #X = get_tfidf_matrix(data, vocab_) #Faut voir si on utilise la A ou la B
        X = sklearn_Tfidf_B(corpus, n)
    elif t_vector == 'embeds':
        X = get_sum_Embs(data)
    print("Vectorisation par %s achevée pour %d examples (%0.3fs s)\n" % (str(t_vector), len(X), time() - t0))
    return X

class Lexeme:
    """
    Classe contenant les exemples pour WSD pour un lème donné et leur représentation vectorielle
    name = str : nom du lème
    conll_data = list : liste des occurrences du lème en format .conll
    tokens_id = list : liste contenant les tokens ids des lèmes dans leur exemples
    targets = list : liste conteant les classes golds des exemples
    t_vector = str : type de vectorization (est-ce qu'il s'agit d'une vectorization simple tfidf ou d'une double vectorization implémenté avec une pipeline ?)
    vector = ndarray : matrice de features (taille nb_exemples * nb_features)
    """

    def __init__(self, lexeme, t_exemple="lineaire", t_feature=1, t_modele="tf-idf", n=2, test=False):
        print("__"*50)
        print("\t\t\t\t| Lexeme : "+lexeme+" |")
        print("__"*50)
        general_path = "data/"+lexeme+"/"+lexeme

        #Phase 1 : chargement des données
        self.name = lexeme
        self.nb_elements = n #Nombre d'éléments à prendre en compte lors de la création d'exemples (n_grams pour linéaire et profondeur pour l'arbre syntaxique)

        #On charge les exemples d'apprentissage en format brut pour le lexeme passé en paramettres
        if test == True:
            print("Version de test en cours")
            conll_data = load_conll_file(general_path+".conll")[:5]
            self.tokens_id = read_one_val_per_line(general_path+".tok_ids")[:5]
            self.targets = read_one_val_per_line(general_path+".gold")[:5]
        else :
            conll_data = load_conll_file(general_path+".conll")
            self.tokens_id = read_one_val_per_line(general_path+".tok_ids")
            self.targets = read_one_val_per_line(general_path+".gold")

        #On parse une première fois l'information pour recupérer les champs qui nous intéréssent à vectoriser
        exemples_non_vec = make_exemples(conll_data, self.tokens_id, t_exemple, t_feature, n)

        #Finalement, on éffectue la vectorisation et on enregistre les exemples qui sont désormais prêt à passer à la clusterisation
        self.exemples = vectoriser(exemples_non_vec, t_modele, n)

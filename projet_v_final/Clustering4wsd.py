#enconding : utf8
from time import time
from collections import defaultdict
from kmeans_cluster import KMeans
from plots4wsd import *
from sklearn.cluster import AgglomerativeClustering

#Fonctions d'utilité pour calculer la précisiton
def list2dict(labels_list):
    """
    Transforme une liste où les clefs sont les valeurs d'entrées et les valeurs sont des listes contenant les indices associés
    """
    dico = defaultdict(list)
    for idx, val in enumerate(labels_list):
        dico[val].append(idx)
    return dico

def make_match_dict(dict1, dict2):
    """
    Construit un dictionnaire regrouppant les intersections des deux dictionnaires passés en arguments
    """
    dico_match = {}

    for gold_ind, gold_list in dict1.items():
        intersection = 0
        counter = 0

        for pred_ind, pred_list in dict2.items():
            intersection = len(set(gold_list).intersection(set(pred_list)))

            if intersection >= counter and pred_ind not in dico_match.values():
                counter = intersection
                dico_match[gold_ind] = pred_ind

    return dico_match

def accuracy(gold_class, labels_prediction, cm=False):
    """
    Calcule la précision des labels prédits par un clustering en utilisant la méthode de l'intersection
    """
    dico_prediction = list2dict(labels_prediction)
    dico_gold_class = list2dict(gold_class)

    dico_match = make_match_dict(dico_gold_class, dico_prediction)

    for k_pred, v_pred in dico_prediction.items():
        for k in dico_match:
            if k_pred == dico_match[k]:
                for i in v_pred:
                    labels_prediction[i] = k

    if cm == True :
        display_cm(gold_class, labels_prediction)

    corrects = len([y_true for y_true, y_pred in zip(gold_class, labels_prediction) if y_true==y_pred])

    return round(corrects/len(gold_class), 2)

def print_results(results):
    print("--"*50)
    print("\t\t\tResultats des méthodes de clustering")
    print("--"*50)
    print("\t\tMETHODE\t\tACCURACY\t\tPARAMETTRES")
    for result in results:
        name = str(result[0])
        accuracy = str(result[1])
        params = ""
        if name=="Hierarchical":
            params = "Mesure de distance : "+str(result[2]['affinity'])+"\tLinkage : "+str(result[2]['linkage'])
        print("\t\t"+name+"\t"+accuracy+"\t"+params)
    print("--"*50)

#Fonction principale
def run_clusters(k, X, gold_labels):
    """
    Éxécute KMeans et Hierarchical clustering sur les mêmes données afin de comparer les différents modèles avec différents paramettres
    k : int -> nombre de clusters à chercher
    X : np.array -> matrice de données
    gold_labels : list(int) -> labels gold
    """

    km = KMeans(k)

    hierarchical = [
        AgglomerativeClustering(k, affinity='euclidean', linkage='ward'),
        AgglomerativeClustering(k, affinity='euclidean', linkage='complete'),
        AgglomerativeClustering(k, affinity='euclidean', linkage='average'),
        AgglomerativeClustering(k, affinity='euclidean', linkage='single'),
        AgglomerativeClustering(k, affinity='manhattan', linkage='complete'),
        AgglomerativeClustering(k, affinity='manhattan', linkage='average'),
        AgglomerativeClustering(k, affinity='manhattan', linkage='single'),
        AgglomerativeClustering(k, affinity='cosine', linkage='complete'),
        AgglomerativeClustering(k, affinity='cosine', linkage='average'),
        AgglomerativeClustering(k, affinity='cosine', linkage='single'),
        ]

    labels_km = km.find_best_init(X)
    labels_hierar = [clustering.fit_predict(X) for clustering in hierarchical]

    results = [('Hierarchical', accuracy(labels_hierar[i], gold_labels), cluster.get_params()) for i, cluster in enumerate(hierarchical)]
    results.append(('KMeans', accuracy(labels_km, gold_labels)))

    print_results(results)

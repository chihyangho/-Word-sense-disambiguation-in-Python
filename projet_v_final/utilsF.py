#encoding: utf8

import numpy as np

def euclidean_dist(a, b):
    """
    Calcule la distance Euclidienne entre deux point supposées de même dimension
    """
    dist = np.sum([np.float_power(a[i]-b[i], 2) for i in range(len(a))])
    return np.sqrt(dist)

def manhattan_dist(a, b):
    """
    Calcule la distance de Manhattant entre deux points supposées de même dimension
    """
    return np.sum([np.absolute(a[i]-b[i]) for i in range(len(a))])

def cosinus_sim(a, b):
    """
    Calcule le cosinus entre deux vecteurs supposées de même dimesion
    """
    norm_a = np.sqrt(np.sum([x*x for x in a]))
    norm_b = np.sqrt(np.sum([x*x for x in b]))
    return np.dot(a, b) / (norm_a*norm_b)

def mean_point(points):
    """
    Retourne le centroïde moyen d'un nuage de points (list/array de points) supposés de même dimesion
    """
    centroid = [np.mean([p[i] for p in points]) for i in range(len(points[0]))]
    return np.array(centroid)

def complete_linkage(cluster1, cluster2, dist_func):
    """
    Calcule la distance maximum entre deux ensembles de données
    """
    return np.max([dist_func(a, b) for a, b in zip(cluster1, cluster2)])

def single_linkage(cluster1, cluster2, dist_func):
    """
    Calcule la distance minimale entre deux ensembles de données
    """
    return np.min([dist_func(a, b) for a, b in zip(cluster1, cluster2)])

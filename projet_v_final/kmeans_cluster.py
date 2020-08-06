#enconding : utf8
import numpy as np

from utilsF import euclidean_dist, mean_point

class KMeans:
    def __init__(self, k):
        self.nb_clusters = k
        self.centroides = []

    def assign_point_to_cluster(self, data_points):
        """
        A chaque point de Data assigne le centroide le plus proche formant ainsi les clusters
        """
        assignments = []
        for point in data_points:
            min_dist = np.Inf
            near_ind = 0
            for i_c, center in enumerate(self.centroides):
                temp_d = euclidean_dist(point, center)
                if min_dist > temp_d:
                    min_dist = temp_d
                    near_ind = i_c
            assignments.append(near_ind)
        return assignments

    def find_clusters(self, data):
        assignments = self.assign_point_to_cluster(data)
        clusters = []
        for i_c, _ in enumerate(self.centroides):
            c = [i_p for i_p, center in enumerate(assignments) if center==i_c]
            clusters.append(c)
        return (np.array(clusters), np.array(assignments))

    def update_centers(self, data, clusters_ind):
        """
        Calcule le point moyen pour chaque liste de point représentant un cluster
        """
        clusters = [[data[ind] for ind in cluster] for cluster in clusters_ind]
        return [mean_point(cluster) for cluster in clusters]

    def run(self, X, first_centers):

        self.centroides = first_centers
        old_clusters, _ = self.find_clusters(X)
        self.centroides = self.update_centers(X, old_clusters)
        new_clusters, labels = self.find_clusters(X)
        while (np.array(old_clusters) != np.array(new_clusters)).all():
            self.centroides = self.update_centers(X, new_clusters)
            temp_clusters, labels = self.find_clusters(X)
            old_clusters = new_clusters
            new_clusters = temp_clusters
        var_cluster = [np.var(cluster) for cluster in new_clusters]
        return (labels, np.sum(var_cluster))

    def find_best_init(self, X, max_iter=5):
        from time import time
        import random
        print("KMeans en cours d'éxécution")
        t0 = time()
        best_centers = []
        best_labels = []
        min_var = np.Inf
        i = 0
        for i in range(max_iter):
            #Initialisation aléatoire des centroides : on choisit k vecteurs de X de façon aléatoire
            centers = [X[ind] for ind in random.sample(range(len(X)), self.nb_clusters)]
            clusters, var = self.run(X, centers)
            if var < min_var:
                min_var = var
                best_centers = self.centroides
                best_labels = clusters
        self.centroides = best_centers
        print("-> KMeans éxécuté (%.3f sec)" % (time()-t0))
        return best_labels

    def predit_cluster(self, x):
        dist_to_centers = np.array([euclidean_dist(x, center) for center in self.centroides])
        return np.argmin(dist_to_centers)

"""
X = [[2,2],[3,1],[3,3],[-1,-1],[-1,-2],[-3,-1],[0,0]]
km = KMeans(2)
labels = km.find_best_init(X, max_iter = 2)
print("Labels\n", labels)
from plots4wsd import plot_kmeans
plot_kmeans(X, cluster)
"""

#encoding : utf-8
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram


def plot_kmeans(data, clusters):
    df_cluster = []
    for i_c, cluster in enumerate(clusters):
        for x, y in cluster:
            df_cluster.append([i_c, x, y])

    df_cluster = pd.DataFrame(data=np.array(df_cluster), columns=['cluster', 'x', 'y'])
    print(df_cluster)
    sns.scatterplot(x="x", y="y", data=df_cluster, hue="cluster")
    plt.show()
    data = np.array(data)
    normal_data = plt.scatter(data[:,0], data[:,1])
    plt.title("Points bruts")
    plt.grid(True)
    plt.show()


# !!! Cette fonction n'est pas à nous, elle vient de l'exemple Plot Hierarchical Clustering Dendrogram dans sklearn
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def plot_cluster(cluster):
    name, algo = cluster

    if name == 'Hierarchical' :
        plt.title('Hierarchical Clustering Dendrogram')
        # plot the top three levels of the dendrogram
        plot_dendrogram(algo, truncate_mode='level', p=3)

    plt.show()

def display_cm(gold_class, pred_labels):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(gold_class, pred_labels)
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('clusters predits par KMeans')
    plt.ylabel('gold class')
    plt.show()

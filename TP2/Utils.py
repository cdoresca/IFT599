import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score,adjusted_rand_score,normalized_mutual_info_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import numpy as np
import time
import tracemalloc
from sklearn.decomposition import PCA

def read_csv() -> pd.DataFrame:
    x = pd.read_csv("hiseq_data.csv",index_col=0)
    y = pd.read_csv("hiseq_labels.csv",index_col=0)
    return pd.merge(y, x, left_index=True, right_index=True, how='left')

def read_ecg() -> np.ndarray:
    data = np.load('ecg.npz')
    return data['ecg']

def evaluate_cluster_interne(data, cluster, name : str):
    print(f"Méthode: {name}")
    silhouette = silhouette_score(data, cluster)
    davies = davies_bouldin_score(data, cluster)
    calinski = calinski_harabasz_score(data,cluster)
    print("Silhouette Score: ", silhouette)
    print("Davies-Bouldin Index: ", davies)
    print("Indice de Calinski-Harabasz: ", calinski)
    print()
    return [silhouette, davies, calinski]


def  evaluate_cluster_externe(true, predict, name):
    print(f"Méthode: {name}")
    print("Adjusted Rand Index: ", adjusted_rand_score(true, predict))
    print("Information mutuelle normalisée: ",normalized_mutual_info_score(true,predict))

def evaluate_anomalie(true, predict, name):
    print(f"Méthode: {name}" )
    print("Exactitude: ",accuracy_score(true,predict))
    print("Rappel: ",recall_score(true,predict))
    print("Précision: ",precision_score(true,predict))
    print("F1-score: ",f1_score(true,predict))
    print("ROC-AUC: ",roc_auc_score(true,predict))
    print()

class Timer:
    def __init__(self):
        self.start = time.time()
    
    def stop(self, name:str):
        print(f"Temps d'exécution {name}: {time.time() - self.start:.4f} secondes")

class Memory:
    def __init__(self):
        tracemalloc.start()

    def stop(self):
        current, peak = tracemalloc.get_traced_memory()
        print(f"Mémoire actuelle : {current / 10**6:.2f} MB")
        print(f"Pic mémoire : {peak / 10**6:.2f} MB")
        tracemalloc.stop()


def plot_clusters(X : np.ndarray, labels: np.ndarray, title="Clusters"):
    """
    Trace un scatter plot 2D des clusters.
    
    Parameters:
    - X : np.array ou pd.DataFrame, les données (n_samples, n_features)
    - labels : array-like, labels des clusters
    - title : str, titre du graphique
    """
    # Réduction à 2D pour visualisation
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    plt.figure(figsize=(8,6))
    
    # On suppose 5 clusters
    for i in range(5):
        plt.scatter(
            X_2d[labels == i, 0], 
            X_2d[labels == i, 1], 
            label=f'Cluster {i+1}',
            alpha=0.7
        )
    
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(True)
    plt.show()
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
    print(f"Méthode interne: {name}")
    silhouette = silhouette_score(data, cluster)
    davies = davies_bouldin_score(data, cluster)
    calinski = calinski_harabasz_score(data,cluster)
    print("Silhouette Score: ", silhouette)
    print("Davies-Bouldin Index: ", davies)
    print("Indice de Calinski-Harabasz: ", calinski)
    print()
    return [('Silhouette Score: ',silhouette), ("Davies-Bouldin Index: ",davies), ("Indice de Calinski-Harabasz: ",calinski)]


def  evaluate_cluster_externe(true, predict, name):
    rand = adjusted_rand_score(true, predict)
    info = normalized_mutual_info_score(true,predict)
    print(f"Méthode externe: {name}")
    print("Indice de Rand Ajusté: ", rand)
    print("Information mutuelle normalisée: ",info)
    print()
    return [("Indice de Rand Ajusté: ",rand),("Information mutuelle normalisée: ",info)]

def evaluate_anomalie(true, predict, name):
    print(f"Méthode: {name}" )
    exactitude =accuracy_score(true,predict)
    rap = recall_score(true,predict)
    pre = precision_score(true,predict)
    f1 = f1_score(true,predict)
    roc = roc_auc_score(true,predict)
    print("Exactitude: ",exactitude)
    print("Rappel: ",rap)
    print("Précision: ",pre)
    print("F1-score: ",f1)
    print("ROC-AUC: ",roc)
    print()

    return [("Exactitude: ",exactitude),("Rappel: ",rap),("Précision: ",pre),("F1-score: ",f1),("ROC-AUC: ",roc)]

class Timer:
    def __init__(self):
        self.start = time.time()
    
    def begin(self):
        self.start = time.time()

    def stop(self, name:str):
        duree = time.time() - self.start
        print(f"Temps d'exécution {name}: {duree:.4f} secondes")
        print()
        return (f"Temps d'exécution {name} :",duree)

class Memory:
    def __init__(self):
        tracemalloc.start()

    def start(self): 
        tracemalloc.start()

    def stop(self,name:str):
        current, peak = tracemalloc.get_traced_memory()
        print(f"Mémoire actuelle {name}: {current / 10**6:.2f} MB")
        print(f"Pic mémoire {name}: {peak / 10**6:.2f} MB")
        tracemalloc.stop()
        print()
        return (f"Pic mémoire {name}:",peak / 10**6)


def plot_clusters(X : np.ndarray, labels: np.ndarray, title="Clusters"):
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

def moy(a,tab,unite= ''):
    for i in range(a):
        value = np.array([])
        for j in range(i,len(tab),a):
            name, val = tab[j]
            value = np.append(value,val)

        print(f' {name} {np.mean(value)} +/- {np.std(value)} {unite}')
    print()
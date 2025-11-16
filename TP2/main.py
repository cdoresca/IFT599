import Utils as ut
from Models import Cluster
from Preprocesses import Preprocesses

df_hiseq = ut.read_csv()



clean = Preprocesses(df_hiseq)

df_hiseq =clean.drop('Class')

df_hiseq = clean.StandardScaler()
df_hiseq =clean.pca()
'''
cluster_hiseq = Cluster(df_hiseq)


label_kmeans_hiseq = cluster_hiseq.Kmeans()
label_dbscan_hiseq = cluster_hiseq.dbscan()
label_sc_hiseq = cluster_hiseq.SpectralCluster()

cluster = [("KMeans - hiseq",label_kmeans_hiseq),("DBSACN - hiseq",label_dbscan_hiseq),("Spectral Cluster - hiseq",label_sc_hiseq)]

for name,label in cluster:
    ut.plot_clusters(df_hiseq,label, name)

'''
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(df_hiseq)
distances, indices = neighbors_fit.kneighbors(df_hiseq)

distances = np.sort(distances[:,4])
plt.plot(distances)
plt.show()
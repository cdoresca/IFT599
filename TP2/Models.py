from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.ensemble import IsolationForest


class Cluster:

    def __init__(self,data):
        self.data = data
    
    def Kmeans(self):
        k = KMeans(n_clusters=5,random_state=42)
        return k.fit_predict(self.data)
    
    def dbscan(self,ep = 0.5):
        dbscan = DBSCAN(ep)
        return dbscan.fit_predict(self.data)

    def SpectralCluster(self):
        spectral = SpectralClustering(n_clusters=5,affinity='nearest_neighbors',random_state=42)
        return spectral.fit_predict(self.data)
    
class IF:
    def __init__(self,args):
        self.model = IsolationForest(n_estimators = args, random_state=42)

    def train(self,data):
        self.model.fit(data)
        
    def score(self,data):
        return self.model.predict(data)
    
class Autocodeur:

    def __init__(self):
        pass
    
    

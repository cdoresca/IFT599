from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler

class Preprocesses:
    def __init__(self, data):
        self.data = data
    
    def pca(self,n = 100):
        pca = PCA(n_components=n)
        return  pca.fit_transform(self.data)
    
    def label_encoder(self, column):
        le = LabelEncoder()
        self.data[column + '_encoder'] = le.fit_transform(self.data[column])
        self.data = self.data.drop(columns=column)
        return self.data
    
    def StandardScaler(self):
        self.data = StandardScaler().fit_transform(self.data)
        return self.data
    
    def drop(self,column:str):
        self.data = self.data.drop(columns = column)
        return self.data
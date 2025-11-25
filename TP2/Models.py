from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.ensemble import IsolationForest
import torch.nn as nn
import torch
import torch.nn.functional as F


class Cluster:

    def __init__(self,data):
        self.data = data
    
    def Kmeans(self):
        k = KMeans(n_clusters=5)
        return k.fit_predict(self.data)
    
    def dbscan(self,ep = 0.5):
        dbscan = DBSCAN(ep)
        return dbscan.fit_predict(self.data)

    def SpectralCluster(self):
        spectral = SpectralClustering(n_clusters=5,affinity='nearest_neighbors')
        return spectral.fit_predict(self.data)
    
class IF:
    def __init__(self, n_estimators=100):
       
        
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination="auto",
        )

    def train(self, data):
        
        self.model.fit(data)

    def predict(self, data):
      
        return -self.model.predict(data)

    def score(self, data):

        return self.model.score_samples(data)

# copie du code du prof
class AE(nn.Module):

    def __init__(self, in_feature):
        super(AE, self).__init__()
    
        self.enc = nn.Sequential(
            nn.Linear(in_feature, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
            )
        self.dec = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, in_feature),
            )
        
    def forward(self, x):
        encode = self.enc(x)
        decode = self.dec(encode)
        return decode
    
# source : https://www.codecademy.com/article/variational-autoencoder-tutorial-vaes-explained
class VEncoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):  
        super(VEncoder, self).__init__()  
        self.fc1 = nn.Linear(input_dim, hidden_dim)  
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x): 
        h = torch.relu(self.fc1(x)) 
        mu = self.fc_mu(h) 
        logvar = self.fc_logvar(h) 
        return mu, logvar 

class VDecoder(nn.Module):  
    def __init__(self, latent_dim=20, hidden_dim=400, output_dim=784):  
        super(VDecoder, self).__init__()  
        self.fc1 = nn.Linear(latent_dim, hidden_dim)  
        self.fc2 = nn.Linear(hidden_dim, output_dim)  

    def forward(self, z):  
        h = torch.relu(self.fc1(z))  
        return torch.sigmoid(self.fc2(h)) 
 
class VAE(nn.Module):

    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        self.encoder = VEncoder(input_dim, hidden_dim, latent_dim)  
        self.decoder = VDecoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):  
        mu, logvar = self.encoder(x)  
        z = self.reparameterize(mu, logvar)  
        reconstructed = self.decoder(z)  
        return reconstructed, mu, logvar 


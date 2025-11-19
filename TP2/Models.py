from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.ensemble import IsolationForest
import torch.nn as nn
import torch
import torch.nn.functional as F


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
    def __init__(self, n_estimators=100, random_state=42):
        """
        Isolation Forest wrapper.
        
        n_estimators : nombre d’arbres
        random_state : graine aléatoire
        """
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination="auto",
            random_state=random_state
        )

    def train(self, data):
        """
        Entraîne le modèle.
        data : array-like (n_samples, n_features)
        """
        self.model.fit(data)

    def predict(self, data):
        """
        Retourne les prédictions :
        1 = anomalie
        -1 = normal
        """
        return -self.model.predict(data)

    def score(self, data):
        """
        Retourne les *anomaly scores* :
        Plus le score est GRAND → plus c'est NORMAL
        Pour un score inversé (plus grand = plus anormal), voir plus bas.
        """
        return self.model.score_samples(data)
    
class DAE(nn.Module):

    def __init__(self, in_feature):
        super(DAE, self).__init__()
    
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

# https://github.com/IParraMartin/Sparse-Autoencoder/blob/main/sae.py   
class SAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, latent_dim=32,rho =0.5):
        super().__init__()            
        self.rho = rho
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Sigmoid()            
        )
        
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
             nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def sparsity_penalty(self, encoded):
        rho_hat = torch.mean(encoded, dim=0)
        rho = self.rho
        epsilon = 1e-8
        rho_hat = torch.clamp(rho_hat, min=epsilon, max=1 - epsilon)
        kl_divergence = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
        sparsity_penalty = torch.sum(kl_divergence)
        return self.rho * sparsity_penalty

    def loss_function(self, x_hat, x, encoded):
        mse_loss = F.mse_loss(x_hat, x)
        sparsity_loss = self.sparsity_penalty(encoded)
        return mse_loss + sparsity_loss
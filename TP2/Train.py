import numpy as np
import torch

#Code ecrie avec l'aide de chatgpt pour le raffinement
class Trainer:

    def __init__(self, data: np.ndarray, shuffle=True):

        if shuffle:
            np.random.shuffle(data)

        self.allData = data[:,:-1]

        df_anomalie = data[data[:, -1] == 1][:, :-1]
        df_normale  = data[data[:, -1] == 0][:, :-1]

        n_norm = df_normale.shape[0]
        n_ano  = df_anomalie.shape[0]

        n_norm_60 = int(0.6 * n_norm)
        n_ano_80 = int(0.8 * n_ano)

        self.train = df_normale[:n_norm_60]

        test_norm = df_normale[n_norm_60 : n_norm_60 + int(0.3*n_norm)]
        test_ano  = df_anomalie[:n_ano_80]
        self.test = np.vstack([test_norm, test_ano])

        val_norm = df_normale[n_norm_60 + int(0.3*n_norm):]
        val_ano  = df_anomalie[n_ano_80:]
        self.validation = np.vstack([val_norm, val_ano])

    def train_autoencoder(self, model, epochs=50, lr=1e-3):

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        X_tensor = torch.tensor(self.train, dtype=torch.float32)

        for epoch in range(epochs):
            optimizer.zero_grad()
            reconstructed = model(X_tensor)
            loss = criterion(reconstructed, X_tensor)
            loss.backward()
            optimizer.step()

    def anomaly_score(self, model, on="test"):

        if on == "test":
            X = self.test
        elif on == "validation":
            X = self.validation

        X_tensor = torch.tensor(self.test, dtype=torch.float32)

        with torch.no_grad():
            recon = model(X_tensor)

        mse = torch.mean((recon - X_tensor)**2, dim=1)
        return mse.numpy()
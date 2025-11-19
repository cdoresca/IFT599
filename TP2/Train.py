import numpy as np

class Train:

    def __init__(self, data: np.ndarray):

        df_anomalie = data[data[:, -1] == 1]
        df_normale  = data[data[:, -1] == 0]

      
        n_norm = df_normale.shape[0]
        n_ano  = df_anomalie.shape[0]

        n_norm_60 = int(0.6 * n_norm)
        n_ano_80 = int(0.8 * n_ano)
       
        self.train = df_normale[:n_norm_60]
        test_norm  = df_normale[n_norm_60 : n_norm_60 + int(0.30*n_norm)]
        val_norm   = df_normale[n_norm_60 + int(0.30*n_norm):]

        
        test_ano = df_anomalie[:n_ano_80]
        val_ano  = df_anomalie[n_ano_80:]

        
        self.test  = np.vstack([test_norm, test_ano])
        self.validation   = np.vstack([val_norm,  val_ano])

    def train(self,ae):
        pass
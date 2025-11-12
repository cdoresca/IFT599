import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import Utils as ut
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

df_ecg = ut.read_ecg()
df_hiseq_y = ut.read_csv('hiseq_labels.csv')
df_hiseq_x = ut.read_csv('hiseq_data.csv')

df_hiseq = pd.merge(df_hiseq_y,df_hiseq_x, how='left',left_index=True, right_index=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_hiseq)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
plt.figure(figsize=(6,5))
sns.scatterplot(x='PC1', y='PC2', data=df_pca)
plt.title("Projection des données selon les 2 premières composantes principales")
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.show()



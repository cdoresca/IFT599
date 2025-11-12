import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_csv(data):
    return pd.read_csv(data,index_col=0)

def read_ecg():
    data = np.load('ecg.npz')
    return data['ecg']
#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.cm as cm

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import VarianceThreshold

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

from sklearn.pipeline import make_pipeline

from datetime import datetime
from astral import Astral

#%%

# Better visualization in console
pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

# Make the graphs a bit prettier, and bigger
plt.rcParams['figure.figsize'] = (15, 5)


#%%

octavio_df = pd.read_csv("/Users/David/Documents/CloudStation/2018_MCSBT/02_Term/ML_AI/Groupwork/01_Clustering/PY/octavio.csv", sep=',')
arthur_df = pd.read_csv("/Users/David/Documents/CloudStation/2018_MCSBT/02_Term/ML_AI/Groupwork/01_Clustering/PY/arthur.csv", sep=',')
alejandro_df = pd.read_csv("/Users/David/Documents/CloudStation/2018_MCSBT/02_Term/ML_AI/Groupwork/01_Clustering/PY/alejandro.csv", sep=',')
david_df = pd.read_csv("/Users/David/Documents/CloudStation/2018_MCSBT/02_Term/ML_AI/Groupwork/01_Clustering/PY/david-music-songs.csv", sep=',')
hannah_df = pd.read_csv("/Users/David/Documents/CloudStation/2018_MCSBT/02_Term/ML_AI/Groupwork/01_Clustering/PY/hannah.csv", sep=',')
leila_df = pd.read_csv("/Users/David/Documents/CloudStation/2018_MCSBT/02_Term/ML_AI/Groupwork/01_Clustering/PY/leila.csv", sep=',')

df = pd.concat([octavio_df, arthur_df, alejandro_df, david_df, hannah_df, leila_df])
df.keys()
df = df[['energy', 'liveness', 'speechiness', 'acousticness','instrumentalness', 'danceability','valence', 'uri','song_names', 'artist']]


df.to_csv('/Users/David/Documents/CloudStation/2018_MCSBT/02_Term/ML_AI/Groupwork/01_Clustering/PY/TEAM_C_SPOTIFY.csv', sep=';')


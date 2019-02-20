#LIBRARIES

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

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#DATASET 
#insert path below
audiofeats = pd.read_csv(r'C:\Users\Leila\Desktop\spotify-clustering\TEAM_C_SPOTIFY.csv', sep=';')

#%%
##########
#DECISION
##########




#%%
##########
#EXPLORATION
###########

# Correlation Matrix Heatmap

f, ax = plt.subplots(figsize=(10, 6))
corr = music.corr()
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="BuGn_r",fmt='.2f',
                 linewidths=.05)
f.subplots_adjust(top=0.93)
t= f.suptitle('Music Attributes Correlation Heatmap', fontsize=14)




#%%
###########
#CLEANING
#############

audiofeats = audiofeats.drop(['uri','artist','song_names','Unnamed: 0'], axis=1)
x = np.array(audiofeats)

plt.show()
mms = MinMaxScaler()
mms.fit(x)
data_transformed = mms.transform(x)

#%%
############
#CLUSTERING
############



model = KMeans(n_clusters=4).fit(X)
labels = model.fit_predict(X)
model.cluster_centers_
centers = np.array(model.cluster_centers_)
print(centers)
df_labels=pd.DataFrame({"labels":labels})

frames=[audiofeats, audiofeats_labels]

result = pd.concat(frames, axis = 1, sort=False)
result.info()
filter_0=result.loc[result['labels'] == 0]

filter_0.describe()



#%%
#################
#VIZUALIZATION
###################

plt.scatter(x=audiofeats.energy, y= audiofeats.danceability, c=kmeans.labels_, cmap='rainbow')


#%%
###################################
#LABELLING CLUSTERS/INTERPRETATION
###################################
audiofeats['cluster'] = kmeans.labels_
audiofeats.head(10)

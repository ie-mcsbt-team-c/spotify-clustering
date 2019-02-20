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
audiofeats = pd.read_csv(r'C:\Users\Leila\Desktop\AI & ML\audio-features-unsupervised-learning\audiofeatures.csv')


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



#dropping 'uri' because float datatype
x = np.array(audiofeats.drop(['uri'], 1).astype(float))


#%%
############
#CLUSTERING
############


plt.show()
mms = MinMaxScaler()
mms.fit(x)
data_transformed = mms.transform(x)

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

plt.scatter(x["energy"],x["danceability"], c=kmeans.labels_, cmap='rainbow')


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Prep Dataframe - Leila you probably have this already but I used it for my code to work
spot = pd.read_csv('/Users/hannaholdorf/Documents/Documents/CSBT/2nd Term/AI & ML/Statistical Learning & Predictions/Clustering/FINAL_TEAM_C.csv', sep=',')
spot1 =  spot.drop(['tempo', 
                    'time_signature',
                    'key',
                    'duration_ms',
                    'loudness',
                    'mode', 
                    'type', 
                    'uri',
                    'acousticness',
                    'song_names',
                    'artist'], axis=1)


spot1 = spot1.dropna(axis=0)
spot1 = np.array((spot1).astype(float))


kmeans = KMeans(n_clusters=4)  
kmeans.fit(spot1)

X = spot1[:,0]       #energy
Y = spot1[:,3]       #instrumentalness
Z = spot1[:,4]       #danceability

#ax.scatter(X, Y, Z, c='r', marker='o')
ax.scatter(X, Y, Z, c=kmeans.labels_, cmap='rainbow', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

#%%
###################################
#LABELLING CLUSTERS/INTERPRETATION
###################################
audiofeats['cluster'] = kmeans.labels_
audiofeats.head(10)

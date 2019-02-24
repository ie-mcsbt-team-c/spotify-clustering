#LIBRARIES
#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.cm as cm

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler

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

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#%%

#DATASET 
#insert path below
all_feats = pd.read_csv('https://raw.githubusercontent.com/ie-mcsbt-team-c/spotify-clustering/master/TEAM_C_SPOTIFY.csv',';')
audiofeats = pd.read_csv('https://raw.githubusercontent.com/ie-mcsbt-team-c/spotify-clustering/master/TEAM_C_SPOTIFY.csv',';')

#%%
##########
#DECISION
##########
#Spotify Dataset: The Dataset consist of the last 50 songs listened on Spotify extracted thanks to the Spotify API.
#The Data contains 228 lignes and 7 features to describe the music : Energy, Liveness, Speechiness, 
#Acousticness, Instrumentalness, Danceability and Valence. 
#N.B : Each feature used are explained in the PPT document. 

#This data only contains numerical value. 
audiofeats.select_dtypes(include= 'number')

##########
#%%EXPLORATION
###########

#Caracteristic of the Dataset 
audiofeats.describe() 
audiofeats = audiofeats.drop("Unnamed: 0", axis = 1)


#Histogram 
#Visualizing data in One Dimension (1-D) thanks to histogram. Here, we are 
# only concerned with analyzing one data attribute or variable and visualizing the same (one dimension)

audiofeats.hist(bins=15, color='green', edgecolor='black', linewidth=1.0,
           xlabelsize=8, ylabelsize=8, grid=False)    
plt.tight_layout(rect=(0, 0, 1.2, 1.2))    

#Conclusion: The plot give a good idea of data distribution of each feature. 


#Visualizing data in Two Dimensions (2-D):  Let's now check out potential relationships 
# or correlations amongst the different data attributes

# The pair-wise Correlation Matrix Heatmap

f, ax = plt.subplots(figsize=(10, 6))
corr = audiofeats.corr()
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="BuGn_r",fmt='.2f',
                 linewidths=.05)
f.subplots_adjust(top=0.93)
t= f.suptitle('Music Attributes Correlation Heatmap', fontsize=14)

#Conclusion : Acoustiness and Energy seems to be negatively correlated. 

#%%

# Box Plots
f, (ax) = plt.subplots(1, 1, figsize=(12, 4))
f.suptitle('Energy - Acousticness of Music', fontsize=14)

sns.boxplot(x="acousticness", y="energy", data=audiofeats)
ax.set_xlabel("Energy",size = 20,alpha=0.8)
ax.set_ylabel("Acousticness",size = 20,alpha=0.8)

#Conclusion : 
#%%
###########
#CLEANING
#############

audiofeats = audiofeats.drop(['uri','artist','song_names'], axis=1)
audio_array = np.array(audiofeats)
#%% 
plt.show()

#Scale : Transforms features by scaling each feature to a given range.

mms = MinMaxScaler()
mms.fit(audio_array)
audio_array_scaled = mms.transform(audio_array)

#%%
############
#CLUSTERING
############

model = KMeans(n_clusters=4).fit(audio_array_scaled)
labels = model.fit_predict(audio_array_scaled)

#David, I don´t understand why there is so much stuff in this code ? 
#I rewrote it in 2 lines for clarity but feel free to add anything 
# or add why it´s there for in comments pliz

#model = KMeans(n_clusters=4).fit(audio_array)
#labels = model.fit_predict(X)
#model.cluster_centers_
#centers = np.array(model.cluster_centers_)
#print(centers)
#audiofeats_labels=pd.DataFrame({"labels":labels})
#
#frames=[audiofeats, audiofeats_labels]
#
#result = pd.concat(frames, axis = 1, sort=False)
#result.info()
#filter_0=result.loc[result['labels'] == 0]
#
#filter_0.describe()



#%%
#################
#VIZUALIZATION
###################



plt.scatter(x=audiofeats.energy,y=audiofeats.danceability, c=model.labels_, cmap='rainbow')

#%%

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

audio_3d = np.array((audiofeats).astype(float))
#
#kmeans = KMeans(n_clusters=4)  
#kmeans.fit(spot1)

X = audio_3d[:,0]       #energy
Y = audio_3d[:,3]       #instrumentalness
Z = audio_3d[:,4]       #danceability

#ax.scatter(X, Y, Z, c='r', marker='o')
ax.scatter(X, Y, Z, c=model.labels_, cmap='rainbow', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

#%%
###################################
#LABELLING CLUSTERS/INTERPRETATION
###################################
audiofeats['cluster'] = model.labels_
audio_withinfo = audiofeats.merge(all_feats[['artist','song_names']], left_index=True, right_index=True)

audio_withinfo.head(10)
c1= audio_withinfo.loc[audiofeats['cluster'] == 0]
c2= audio_withinfo.loc[audiofeats['cluster'] == 1]
c3= audio_withinfo.loc[audiofeats['cluster'] == 2]
c4= audio_withinfo.loc[audiofeats['cluster'] == 3]




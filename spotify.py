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

#Here I still need to figure out the explanation why k-means would work well

#Our first goal is to find what number of clusters we need. We will use Elbow method and back it up with silohoutte. 


#%%

#ELBOW METHOD#
#If you graph the percentage of variance explained by 
#the clusters against the number of clusters, the first clusters will 
#add much information (explain a lot of variance), but at some point 
#the marginal gain will drop, giving an angle in the graph. 
#The number of clusters are chosen at this point, hence the “elbow criterion”.

ks = range(1, 10)
inertias = []

def number_of_clusters():
    
    for k in ks:
        model = KMeans(n_clusters=k, random_state=None)
        model.fit(audio_array_scaled)
        inertias.append(model.inertia_)
        print({"Number of cluster":k})
        print({"Model Inertia Value":model.inertia_})


# We can see there is no a significant different after 5 clusters
# Inertia refers to within-cluster sum-of-squares of all clusters. Then show how well packed the clusters. 
# Our goal is to         


number_of_clusters()

plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('Percentage of variance explained')
plt.figtext(.5, .8, "Explanation:")
plt.figtext(.5, .75, "How much intra-cluster variance")
plt.figtext(.5, .7, "is explained when we add")
plt.figtext(.5, .65, "a new cluster")
plt.xticks(ks)
plt.show()

#%%

#SILHOUETTE ANALYSIS#

#Silhouette coefficients 
#(as these values are referred to as) near +1 
#indicate that the sample is far away from the
# neighboring clusters. A value of 0 indicates 
#that the sample is on or very close to the decision 
#boundary between two neighboring clusters and negative 
#values indicate that those samples might have been assigned 
#to the wrong cluster.
#
#Avg Silhouette score: I get the average Silhouette score of the cluster k.
#Silhouette samples score: And, for all the samples belonging to a given cluster 
#(from 1 to k), I calculate the individual silhouette score of each sample belong 
#to that cluster.

# Important #

#Negative values means that they are not in the right cluster since they might
#be closer to other centroids from other clusters. 

#Then I sort the scores of all samples belonging to each cluster. 
#This is needed so that I can plot the score in an ascending order.

#s(i) = {b(i) - a(i)} / {\max\{a(i),b(i)\}}
#


#Explnation here https://www.youtube.com/watch?v=5TPldC_dC0s


range_n_clusters = [4,5,6]
scores=[]

def number_of_clusters_silhouette():
        
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
         
        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(audio_array_scaled) + (n_clusters + 1) * 10])
        
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(audio_array_scaled)
        
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(audio_array_scaled, cluster_labels)
        scores.append(silhouette_avg)
        print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
#        
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(audio_array_scaled, cluster_labels)
        
#        If you want to see the values for each point of the dataset
#        print(sample_silhouette_values)
        
        y_lower = 10
        
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]
#    
            ith_cluster_silhouette_values.sort()
#    
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
#    
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
    
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
    
            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")
        
            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        
            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        
            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(audio_array_scaled[:, 0], audio_array_scaled[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                        c=colors, edgecolor='k')
        
            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                        c="white", alpha=1, s=200, edgecolor='k')
        
        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')
    
            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")
    
            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')
            
            
            
#    plt.figure()
    


#    plt.show()
    
   
        
        
number_of_clusters_silhouette()

#%%

#model. Clustering the dataset with 

model = KMeans(n_clusters=5, init="k-means++", n_init=228, precompute_distances = True, random_state=None, max_iter=300).fit(audio_array_scaled)
labels = model.fit_predict(audio_array_scaled)


#%%
def clustering_centers_A():
    model = KMeans(n_clusters=5, init="k-means++", n_init=228, precompute_distances = True, random_state=50, max_iter=300).fit(audio_array_scaled)
#    labels = model.fit_predict(audio_array_scaled)


    model.cluster_centers_
    centers = np.array(model.cluster_centers_)
    counter = len(centers)
    i = 0
    for r in range(counter):
        i+=1
        if i <= counter: 
            centers_0 = print ({"The center of cluster "+str(r):centers[r]})
    return centers_0
            
clustering_centers_A()

#%%

def clustering_centers_B():
    model = KMeans(n_clusters=5, init="k-means++", n_init=1000, precompute_distances = True, random_state=20, max_iter=300).fit(audio_array_scaled)
#    labels = model.fit_predict(audio_array_scaled)


    model.cluster_centers_
    centers = np.array(model.cluster_centers_)
    counter = len(centers)
    i = 0
    for r in range(counter):
        i+=1
        if i <= counter: 
            print ({"The center of cluster "+str(r):centers[r]})
            
clustering_centers_B()
#%%

def clustering_centers_C():
    model = KMeans(n_clusters=5, init="k-means++", n_init=228, precompute_distances = True, random_state=100, max_iter=300).fit(audio_array_scaled)
#    labels = model.fit_predict(audio_array_scaled)


    model.cluster_centers_
    centers = np.array(model.cluster_centers_)
    counter = len(centers)
    i = 0
    for r in range(counter):
        i+=1
        if i <= counter: 
            print ({"The center of cluster "+str(r):centers[r]})
            
clustering_centers_C()
#%%

def clustering_centers_D():
    model = KMeans(n_clusters=5, init="k-means++", n_init=228, precompute_distances = True, random_state=None, max_iter=300).fit(audio_array_scaled)
#    labels = model.fit_predict(audio_array_scaled)


    model.cluster_centers_
    centers = np.array(model.cluster_centers_)
    counter = len(centers)
    i = 0
    for r in range(counter):
        i+=1
        if i <= counter: 
            print ({"The center of cluster "+str(r):centers[r]})
            
clustering_centers_D()

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




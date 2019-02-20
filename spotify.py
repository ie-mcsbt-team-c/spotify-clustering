#LIBRARIES

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#DECISION


#EXPLORATION

#insert path below
audiofeats = pd.read_csv(r'C:\Users\Leila\Desktop\AI & ML\audio-features-unsupervised-learning\audiofeatures.csv')

#CLEANING

#dropping 'uri' because float datatype
x = np.array(audiofeats.drop(['uri'], 1).astype(float))

#CLUSTERING

plt.show()
mms = MinMaxScaler()
mms.fit(x)
data_transformed = mms.transform(x)

kmeans = KMeans(n_clusters=5)  
kmeans.fit(x)

#VIZUALIZATION

plt.scatter(x["energy"],x["danceability"], c=kmeans.labels_, cmap='rainbow')

#LABELLING CLUSTERS/INTERPRETATION

audiofeats['cluster'] = kmeans.labels_
audiofeats.head(10)

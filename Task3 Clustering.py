#!/usr/bin/env python
# coding: utf-8

# ## required library

# In[197]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from glob import glob


# In[138]:


data_files = sorted(glob('data*.csv'))


# In[139]:


data_files


# In[140]:


dataset = pd.concat(pd.read_csv(data)
                      for data in data_files)


# In[141]:


dataset


# In[142]:


dataset.head()


# In[143]:


dataset.tail()


# In[144]:


dataset.shape


# In[145]:


dataset.size


# In[146]:


dataset.info()


# In[147]:


dataset.describe(include= 'all')


# In[148]:


dataset.isna().sum()


# In[149]:


corr = dataset.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# In[150]:


from sklearn.preprocessing import StandardScaler


# In[151]:


features=['AT','AP','AH','AFDP','GTEP','TIT','TAT','TEY','CDP','CO',
         'NOX']


# In[152]:


data_norm=dataset.loc[:, features].values


# In[153]:


data_norm= StandardScaler().fit_transform(data_norm)


# In[154]:


data_norm


# In[155]:


from sklearn.cluster import KMeans
wcss=[]
for i in range(1,8):
    kmeans = KMeans(n_clusters = i, init = 'k-means++',random_state=42)
    kmeans.fit(data_norm)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,8),wcss,'bx-')
plt.title('The Elbow Method for optimal K')
plt.xlabel('Number of clusters')
plt.ylabel('sum of sequare distances')
plt.show()


# In[156]:


Model=KMeans(n_clusters = 3, init='k-means++',random_state=42)
y=Model.fit_predict(data_norm)


# In[157]:


print('labels',Model.labels_)


# In[158]:


dataset['Cluster']= y
dataset.head()


# In[159]:


print('Centroids',Model.cluster_centers_)


# In[160]:


plt.figure(figsize=(8,8))
plt.scatter(data_norm[y==0,0],data_norm[y==0,1],s=100,c='blue',label='Cluster1')
plt.scatter(data_norm[y==1,0],data_norm[y==1,1],s=100,c='green',label='Cluster2')
plt.scatter(data_norm[y==2,0],data_norm[y==2,1],s=100,c='brown',label='Cluster3')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='purple',marker='*',label='centeroied')
plt.title ('Gas Emissions')
plt.legend()
plt.show()


# In[161]:


from sklearn.decomposition import PCA
pca=PCA(n_components=3)
principlecomponents=pca.fit_transform(data_norm)


# In[162]:


pca.explained_variance_ratio_


# In[163]:


sum(pca.explained_variance_ratio_)


# In[164]:


principlecomponents


# In[165]:


principaldataset_df=pd.DataFrame(data=principlecomponents
                              ,columns=['principle components_1','principle components_2','principle components_3'])


# In[166]:


principaldataset_df.head()


# In[167]:


principaldataset_df['Cluster']=Model.labels_


# In[168]:


principaldataset_df


# In[169]:


principaldataset_df.shape


# In[170]:


sns.countplot(x=y)


# In[171]:


import plotly.express as px
fig = px.scatter_3d(principaldataset_df,x='principle components_1',y='principle components_2',z='principle components_3',
                    symbol=y,color=y,size_max=12)
fig.update_layout(margin = dict (l=1,r=0,b=0,t=0))
fig.show()


# In[172]:


# performance metricus
metrics.silhouette_score(principlecomponents,Model.labels_,metric='euclidean')


# ## DBSCAN

# In[173]:


# we use standerdscaler to make all points inside the dataset between (0,1)
from sklearn.preprocessing import StandardScaler
data_norm=dataset.loc[:, features].values
data_norm_s=StandardScaler()
data_norm=data_norm_s.fit_transform(data_norm)


# In[174]:


data_norm


# In[175]:


#from this diagram we will chose the eps
from sklearn.neighbors import NearestNeighbors
neighbors = NearestNeighbors(n_neighbors=2)
distances,indices = neighbors.fit(data_norm).kneighbors(data_norm)
distances = distances[:,1]
distances = np.sort(distances,axis=0)
plt.plot(distances)


# In[176]:


distances


# In[177]:


neigh = NearestNeighbors(n_neighbors=2)
neigh.fit(data_norm)


# In[178]:


# which two numbers calculator the distance
indices


# In[179]:


from sklearn.cluster import DBSCAN
dbscan = DBSCAN (eps=0.9,min_samples=22)
y_dbscan=dbscan.fit_predict(data_norm)


# In[180]:


dbscancluster = DBSCAN(eps=0.9,min_samples=22)
dbscancluster.fit(data_norm)


# In[181]:


clusters=dbscancluster.labels_
len(set(clusters))


# In[182]:


from sklearn.metrics import silhouette_score


# In[183]:


# to know how many clusters we have 
np.unique(y_dbscan)


# In[184]:


y_dbscan


# In[185]:


plt.figure(figsize=(8,8))
plt.scatter(data_norm[y_dbscan==0,0],data_norm[y_dbscan==0,1],s=100,c='red',label='Cluster1')
plt.scatter(data_norm[y_dbscan==1,0],data_norm[y_dbscan==1,1],s=100,c='blue',label='Cluster2')
plt.scatter(data_norm[y_dbscan==-1,0],data_norm[y_dbscan==-1,1],s=100,c='purple',label='Noise')
plt.title('Number of companies')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
plt.show()


# In[186]:


from sklearn.decomposition import PCA
pca=PCA(n_components=3)
principlecomponents=pca.fit_transform(data_norm)


# In[187]:


pca.explained_variance_ratio_


# In[188]:


sum(pca.explained_variance_ratio_)


# In[189]:


principlecomponents


# In[190]:


principaldataset_df=pd.DataFrame(data=principlecomponents
                              ,columns=['principle components_1','principle components_2','principle components_3'])


# In[191]:


principaldataset_df.head()


# In[192]:


principaldataset_df.shape


# In[193]:


sns.countplot(x=y)


# In[194]:


import plotly.express as px
fig = px.scatter_3d(principaldataset_df,x='principle components_1',y='principle components_2',z='principle components_3',
                    symbol=y,color=y,size_max=12)
fig.update_layout(margin = dict (l=1,r=0,b=0,t=0))
fig.show()


# In[199]:


#performance metric
metrics.silhouette_score(principlecomponents, y_dbscan,metric='euclidean')


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


names = ['type','Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline'] 
file = pd.read_csv('wine.data',names=names)

y = file['type']
y.head()


# In[3]:




file =file.drop(columns='type',axis=1)


# In[4]:


file.head()


# In[5]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_std = sc.fit_transform(file)


# In[6]:


from sklearn.decomposition import PCA
pca = PCA(n_components=None)
X_tr = pca.fit_transform(X_std)
features  = range(pca.n_components_)
plt.bar(features,pca.explained_variance_ratio_,color = 'black')
plt.xlabel('PCA features')
plt.ylabel('variace %')
plt.xticks(features)


# In[7]:


pca_comp = pd.DataFrame(X_tr)


# In[8]:


plt.scatter(pca_comp[0],pca_comp[1],alpha=.1,color='black')
plt.xlabel('PCA 1')
plt.ylabel("PCA 2")


# In[9]:


from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmean = KMeans(n_clusters=i)
    kmean.fit(pca_comp.iloc[:,0:2])
    wcss.append(kmean.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('of clusters')
plt.ylabel('WCSS')
plt.show()


# In[16]:


kmean = KMeans(n_clusters=3)
y_kmeans =kmean.fit_predict(pca_comp.iloc[:,0:2])
X =pca_comp.iloc[:,0:2].values


# In[17]:


plt.scatter(X[y_kmeans  == 0,0],X[y_kmeans == 0,1],s=10,c='red',label='Class 1')
plt.scatter(X[y_kmeans  == 1,0],X[y_kmeans == 1,1],s=10,c='green',label='Class 2')
plt.scatter(X[y_kmeans  == 2,0],X[y_kmeans == 2,1],s=10,c='purple',label='Class 3')
plt.scatter(kmean.cluster_centers_[:,0],kmean.cluster_centers_[:,1],s=100,c='yellow',label='Centroids')
plt.title('Cluster of classes')
plt.legend()
plt.show()


# In[18]:


from sklearn.metrics import silhouette_score
# silhouette score
silhouette_avg3 = silhouette_score(X, (kmean.labels_), metric='euclidean')
print("The silhouette score is ", silhouette_avg3)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # Anika Bizla

# ## Prediction using Unsupervised ML

# ## Task 2 of Data Science and Business Analytics

# In[1]:


#importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import datasets


# In[3]:


#Reading and loading the Iris Dataset
iris=datasets.load_iris()
i_data=pd.DataFrame(iris.data,columns=iris.feature_names)
print(i_data.shape)
i_data.head()


# In[5]:


#Predicting the optimum number of clusterss by k means classification
from sklearn.cluster import KMeans
X=i_data.iloc[:,[0,1,2,3]].values


# In[6]:


kmeans=KMeans(n_clusters=9)
kmeans.fit(X)


# In[7]:


y_kmeans=kmeans.predict(X)


# In[10]:


#Plotting the dataset for better observation
plt.scatter(X[:,0],X[:,1],c=y_kmeans,s=10,cmap='inferno')
centers=kmeans.cluster_centers_
plt.scatter(centers[:,0],centers[:,1],c='green',s=500,alpha=0.7)


# #### We can see that that optimum number of clusters should be 2 or 3. So, now we shall take the help of the Elbow Method to confirm the suitable number here

# In[11]:


#Predicting the optimum number of clusters using algo method
kmeans.inertia_


# In[13]:


sse_=[]   #Within-cluster sums of squared error
for k in range(1,11):
    kmeans=KMeans(n_clusters=k).fit(X)
    sse_.append([k,kmeans.inertia_])


# In[15]:


#Plotting the results of this method to observe the "Elbows(s)" in a line graph
plt.plot(pd.DataFrame(sse_)[0],pd.DataFrame(sse_)[1])
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("Within clusters sums of sqaured error")
plt.show()


# In[18]:


#Predicting the optimum number of clusters using Silhouette Analysis
from sklearn.metrics import silhouette_score


# In[21]:


sse_=[]
for k in range(2,11):
    kmeans=KMeans(n_clusters=k).fit(X)
    sse_.append([k,silhouette_score(X,kmeans.labels_)])


# In[23]:


#Plotting the results of this analysis procedure for better understanding
plt.plot(pd.DataFrame(sse_)[0],pd.DataFrame(sse_)[1])
plt.title("The Silhouette Analysis")
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette Score")
plt.show()


# #### Concluding from both the analysis, i have decided to choose the optimum number of clusters as 3. So now i am going to represent the  clusters visually along with thier respective centroids

# In[24]:


#Applying K_Means Classifier to the IRIS Dataset.
kmeans=KMeans(n_clusters=3,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(X)


# In[26]:


#Representing the clusters for visualisation
plt.figure(figsize=(8,6))
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,marker='*',c='Brown',label='Iris-Setosa')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,marker='*',c='darkgreen',label='Iris-Versicolor')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,marker='*',c='Indigo',label='Iris-Virginica')
#Plotting the respective centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,marker=',',c="gold",label='centroids')
plt.legend()
plt.title("Clusters with thier centroids")
            


# #### Above Scatter Plot is the visual representation for the given iris dataset where we can observe that the optimum number of clusters is 3 for this data

# # ThankYou

# In[ ]:





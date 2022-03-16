from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


data = pd.read_csv('iris-dataset.csv')

plt.scatter(data['sepal_length'], data['sepal_width'])
plt.xlabel('Length of sepal')
plt.ylabel('Width of sepal')
plt.show()


# # Unscaled
x = data.copy()
kmeans = KMeans(3)
kmeans.fit(x)

clusters = data.copy()
clusters['cluster_pred'] = kmeans.fit_predict(x)
plt.scatter(clusters['sepal_length'], clusters['sepal_width'], c=clusters['cluster_pred'], cmap='rainbow')
# A scatter plot with unscaled values


# # Scaled
xscaled = data.copy()
x_scaled = preprocessing.scale(xscaled)

# The Elbow Method
wcss = []
cl_num = 10
# 'cl_num' is an arbitrary number.
for i in range(1, cl_num):
    # Range is 1 to 9
    kmeans = KMeans(i)
    kmeans.fit(x_scaled)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)
number_clusters = range(1, cl_num)
plt.plot(number_clusters, wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster Sum of Squares')

kmeans_scaled = KMeans(3)
kmeans_scaled.fit(x_scaled)
clusters_scaled = data.copy()
clusters_scaled['cluster_pred'] = kmeans_scaled.fit_predict(x_scaled)
# After several tries, I settled with 3 Clusters, just like what I did with the clustered unscaled data.
plt.scatter(clusters_scaled['sepal_length'], clusters_scaled['sepal_width'], c=clusters_scaled['cluster_pred'], cmap='rainbow')


# # Checking my Solution
real_data = pd.read_csv('iris-with-answers.csv')
real_data
# Here I loaded the dataset containing the species of the Iris flower to check whether I did a correct Clustering of the Iris Dataset. .

real_data['species'].unique()
# To check the data in the feature species, I used the unique() method.
data_mapped = real_data.copy()
data_mapped['species'] = data_mapped['species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2})
data_mapped
# Here I simply mapped the categorical data into numerical variables to plot them on a graph. 

x = data_mapped.iloc[:, 4:]
x
# I used the .iloc indexer to select my desired position in the dataframe.

kmeans = KMeans(3)
kmeans.fit(x)

identified_clusters = kmeans.fit_predict(x)
data_with_clusters = data_mapped.copy()
data_with_clusters['Cluster'] = identified_clusters
data_with_clusters

# ## Sepal Size
plt.scatter(data_with_clusters['sepal_length'], data_with_clusters['sepal_width'], c=data_with_clusters['Cluster'], cmap='rainbow')
# I learned that Clustering cannot be trusted at all times even with the Elbow Method. The method doesn't show the optimal number of clusters to use - only the different distance between different clusters and within clusters

# ## Petal Size
plt.scatter(data_with_clusters['petal_length'], data_with_clusters['petal_width'], c=data_with_clusters['Cluster'], cmap='rainbow')

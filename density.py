import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

# Generate a dataset with clusters of different densities
data1, _ = make_blobs(n_samples=100, centers=[[0, 0]], cluster_std=0.5, random_state=42)
data2, _ = make_blobs(n_samples=100, centers=[[4, 4]], cluster_std=1.5, random_state=42)
data3, _ = make_blobs(n_samples=100, centers=[[8, 0]], cluster_std=0.8, random_state=42)
data4, _ = make_blobs(n_samples=100, centers=[[8, 8]], cluster_std=1.2, random_state=42)

# Combine the clusters
data = np.vstack([data1, data2, data3, data4])

# K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(data)

# Hierarchical clustering
hierarchical = AgglomerativeClustering(n_clusters=4, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(data)

# DBSCAN clustering
dbscan = DBSCAN(eps=0.8, min_samples=5)
dbscan_labels = dbscan.fit_predict(data)

# Plotting
plt.figure(figsize=(15, 5))

# K-Means plot
plt.subplot(1, 3, 1)
plt.scatter(data[:, 0], data[:, 1], c=kmeans_labels, cmap='viridis', edgecolors='k', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='X', s=200, color='red', label='Centroids')
plt.title('K-Means Clustering')
plt.legend()

# Hierarchical plot
plt.subplot(1, 3, 2)
plt.scatter(data[:, 0], data[:, 1], c=hierarchical_labels, cmap='viridis', edgecolors='k', s=50)
plt.title('Hierarchical Clustering')

# DBSCAN plot
plt.subplot(1, 3, 3)
plt.scatter(data[:, 0], data[:, 1], c=dbscan_labels, cmap='viridis', edgecolors='k', s=50)
plt.title('DBSCAN Clustering')

plt.tight_layout()
plt.show()
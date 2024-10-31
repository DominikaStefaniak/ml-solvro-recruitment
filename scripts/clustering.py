import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from umap import UMAP
from sklearn.metrics import silhouette_score, davies_bouldin_score
import os

# To avoid user warning from KMeans model
os.environ['OMP_NUM_THREADS'] = '1'

# Import the prepared data
df = pd.read_parquet("../data/final_cocktail_dataset.parquet")

# Scale data to improve models efficiency
scaler = StandardScaler()
df[["numIngredients", "numAlcoholicIngredients"]] = scaler.fit_transform(df[["numIngredients", "numAlcoholicIngredients"]])

# Exclude numeric data for dimensionality reduction
numeric_data = df.select_dtypes(include=['int64', 'float64'])

# Apply umap for dimensionality reduction to 10 components
reducer10D = UMAP(n_components=10, random_state=42, n_jobs=1)
umap_data = reducer10D.fit_transform(numeric_data)

# Create a new dataframe with columns reduced by umap
df_umap = pd.DataFrame(umap_data, columns=[f"UMAP{i+1}" for i in range(10)])

# KMeans model

# Search for the best performance of KMeans for different number of clusters using both the Silhouette and Bouldin metric
best_score = -1 # Worst possible score
best_eps_score = 2 # Lowest possible

for number in range(2, 11):
    kmeans = KMeans(n_clusters=number, random_state=42)
    clusters = kmeans.fit_predict(df_umap)

    silhouette_avg = silhouette_score(df_umap, clusters)
    davies_bouldin = davies_bouldin_score(df_umap, clusters)

    # Best when Silhouette is close to 1 and Bouldin is close to 0
    # Strive for largest difference between these two metrics
    if silhouette_avg - davies_bouldin > best_score:
        best_score = silhouette_avg - davies_bouldin
        best_eps_score = number

# Use KMeans with the best-performing number of clusters
kmeans = KMeans(n_clusters=best_eps_score, random_state=42)
df_umap["KMeans_clusters"] = kmeans.fit_predict(df_umap)

# Evaluate the final KMeans model
silhouette_avg_kmeans = silhouette_score(df_umap.iloc[:, :-1], df_umap["KMeans_clusters"])
davies_bouldin_kmeans = davies_bouldin_score(df_umap.iloc[:, :-1], df_umap["KMeans_clusters"])

# Print the results of the used metrics for KMeans
print(f'Number of clusters for best KMeans performance: {best_eps_score}')
print(f'Silhouette score for KMeans: {silhouette_avg_kmeans}, Davies Bouldin score: {davies_bouldin_kmeans}')

#DBSCAN model

# Search for the best performance of DBSCAN for different epsilon using both the Silhouette and Bouldin metric
best_score = -1 # Worst possible score
best_eps_score = 1 # Lowest possible

for number in range(1, 12):
    dbscan = DBSCAN(eps=number*0.1, min_samples=5)
    clusters = dbscan.fit_predict(df_umap.iloc[:, :-1])

    # Ensure that we have more than 1 cluster for silhouette score calculation
    if len(np.unique(clusters)) > 1:
        silhouette_avg = silhouette_score(df_umap.iloc[:, :-1], clusters)
        davies_bouldin = davies_bouldin_score(df_umap.iloc[:, :-1], clusters)

        # Check the combined metric like in KMeans
        if silhouette_avg - davies_bouldin > best_score:
            best_score = silhouette_avg - davies_bouldin
            best_eps_score = number

# Create DBSCAN model with the best-performing eps value
dbscan = DBSCAN(eps=best_eps_score * 0.1, min_samples=5)
df_umap['DBSCAN_clusters'] = dbscan.fit_predict(df_umap.iloc[:, :-1])

# Evaluate the final DBSCAN model
silhouette_avg_dbscan = silhouette_score(df_umap.iloc[:, :-2], df_umap["DBSCAN_clusters"])
davies_bouldin_dbscan = davies_bouldin_score(df_umap.iloc[:, :-2], df_umap["DBSCAN_clusters"])

# Print the results of the used metrics for DBSCAN model
print(f'Clusters created by DBSCAN: {df_umap["DBSCAN_clusters"].unique()}')
print(f'Silhouette score for DBSCAN: {silhouette_avg_dbscan}, Davies Bouldin score: {davies_bouldin_dbscan}')

# Prepare the data for quality evaluation
reducer2D = UMAP(n_components=2, random_state=42, n_jobs=1)
umap_2d = reducer2D.fit_transform(df_umap.iloc[:, :-2])

# Convert to DataFrame for saving
df_umap_2d = pd.DataFrame(umap_2d, columns=["UMAP1", "UMAP2"])
df_umap_2d["KMeans_clusters"] = df_umap["KMeans_clusters"]
df_umap_2d["DBSCAN_clusters"] = df_umap["DBSCAN_clusters"]

# Save the reduced dataframe consisting of two dimensions and the results of both clustering models
df_umap_2d.to_parquet("../data/final_cocktail_dataset_with_clusters.parquet")

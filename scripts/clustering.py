import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap
from umap import UMAP
from sklearn.metrics import silhouette_score, davies_bouldin_score
import os

# To avoid user warning coming from KMeans model
os.environ['OMP_NUM_THREADS'] = '1'

# Import the prepared data
df = pd.read_parquet("../data/final_cocktail_dataset.parquet")

# Scale the data to improve the efficiency of the models
scaler = StandardScaler()
df[["numIngredients", "numAlcoholicIngredients"]] = scaler.fit_transform(df[["numIngredients", "numAlcoholicIngredients"]])

# Exclude numeric data for dimensionality reduction
numeric_data = df.select_dtypes(include=['int64', 'float64'])

# Apply umap for dimensionality reduction to 10 components
reducer10D = UMAP(n_components=10, random_state=42, n_jobs=1)
umap_data = reducer10D.fit_transform(numeric_data)

# Create a new dataframe with columns reduced by umap
df_umap = pd.DataFrame(umap_data, columns=[f"UMAP{i+1}" for i in range(10)])

# Search for the best performance of KMeans for different number of clusters using both the Silhouette and Bouldin metric
best_score = -1 # Worst possible score
best_n_score = 2 # Lowest possible

for number in range(2, 11):
    kmeans = KMeans(n_clusters=number, random_state=42)
    clusters = kmeans.fit_predict(df_umap)

    silhouette_avg = silhouette_score(df_umap, clusters)
    davies_bouldin = davies_bouldin_score(df_umap, clusters)

    # It's best when the Silhouette metric is close to 1 and Bouldin metric close to 0
    # Strive for the biggest difference of these two metrics
    if silhouette_avg - davies_bouldin > best_score:
        best_score = silhouette_avg - davies_bouldin
        best_n_score = number

print(best_score, best_n_score)

# Use KMeans with the number of clusters that performed best according to the used metrics
kmeans = KMeans(n_clusters=best_n_score, random_state=42)
df_umap["KMeans_clusters"] = kmeans.fit_predict(df_umap)

# Evaluate the final KMeans model
silhouette_avg = silhouette_score(df_umap.iloc[:, :-1], df_umap["KMeans_clusters"])
davies_bouldin = davies_bouldin_score(df_umap.iloc[:, :-1], df_umap["KMeans_clusters"])

# Print the results of the used metrics
print(f'Number of clusters for best KMeans performance: {best_n_score}')
print(f'Silhouette score: {silhouette_avg}, Davies Bouldin score: {davies_bouldin}')

# Prepare the data for quality evaluation
reducer2D = UMAP(n_components=2, random_state=42, n_jobs=1)
umap_2d = reducer2D.fit_transform(df_umap.iloc[:, :-1])

# Convert to DataFrame for saving
df_umap_2d = pd.DataFrame(umap_2d, columns=["UMAP1", "UMAP2"])
df_umap_2d["KMeans_clusters"] = df_umap["KMeans_clusters"]

# Save the dataframe
df_umap_2d.to_parquet("../data/final_cocktail_dataset_with_clusters.parquet")

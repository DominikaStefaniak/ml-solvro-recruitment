import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap
from umap import UMAP

df = pd.read_parquet("../data/final_cocktail_dataset.parquet")

# Scale the data to improve the effect of the model
scaler = StandardScaler()
df[["numIngredients", "numAlcoholicIngredients"]] = scaler.fit_transform(df[["numIngredients", "numAlcoholicIngredients"]])

# Exclude numeric data for dimensionality reduction and clustering model
numeric_data = df.select_dtypes(include=['int64', 'float64'])

# Apply umap for dimensionality reduction
reducer = umap.UMAP(n_components=2, random_state=42)
umap_embedding = reducer.fit_transform(numeric_data)

df_umap = pd.DataFrame(umap_embedding, columns=['UMAP1', 'UMAP2'])

# KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
df_umap["clusters"] = kmeans.fit_predict(df_umap)

df_umap.to_parquet("../data/final_cocktail_dataset_with_clusters.parquet")

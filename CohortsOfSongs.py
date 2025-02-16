import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv(r'C:\Users\shalini.annam\OneDrive - Qentelli\Desktop\AIMLprojects\CohortsOfSongs\rolling_stones_spotify.csv')  # Replace with actual path

# View basic information about the dataset
print(df.info())  # Data types, number of non-null values
print(df.describe())  # Statistical summary of numeric features

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# Check for missing values
missing_values = df.isnull().sum()
print(f"Missing values:\n{missing_values}")

# Check for erroneous/irrelevant values (e.g., negative popularity, energy > 1, etc.)
erroneous_values = df[(df['popularity'] < 0) | (df['popularity'] > 100)]  # Example condition
print(f"Erroneous values:\n{erroneous_values}")

# Remove duplicates and erroneous values
df_clean = df.drop_duplicates()
df_clean = df_clean[(df_clean['popularity'] >= 0) & (df_clean['popularity'] <= 100)]

# Handle missing values: Option 1: Drop rows with missing values or Option 2: Fill with median
df_clean = df_clean.dropna()  # or df_clean.fillna(df_clean.median(), inplace=True)

# Check for outliers (e.g., using z-scores for numerical features)
from scipy.stats import zscore
df_clean['zscore_popularity'] = zscore(df_clean['popularity'])
outliers = df_clean[df_clean['zscore_popularity'].abs() > 3]
print(f"Outliers detected:\n{outliers}")

# Univariate analysis (distribution of features)
plt.figure(figsize=(10, 6))
sns.histplot(df_clean['popularity'], kde=True, bins=30)
plt.title('Distribution of Song Popularity')
plt.xlabel('Popularity')
plt.ylabel('Frequency')
plt.show()

# Bivariate analysis: Popularity vs Energy
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_clean, x='energy', y='popularity')
plt.title('Energy vs Popularity')
plt.xlabel('Energy')
plt.ylabel('Popularity')
plt.show()

# Boxplot to visualize how popularity is distributed by album
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_clean, x='album', y='popularity')
plt.title('Popularity Distribution by Album')
plt.xticks(rotation=90)
plt.show()

popular_threshold = 80
popular_songs_album = df_clean[df_clean['popularity'] > popular_threshold].groupby('album').size()

# Plotting albums with the number of popular songs
plt.figure(figsize=(12, 6))
# popular_songs_album.sort_values(ascending=False).head(10).plot(kind='bar')
plt.title('Top 10 Albums with Most Popular Songs')
plt.ylabel('Number of Popular Songs')
plt.xlabel('Album Name')
plt.show()

# Popularity vs Danceability
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_clean, x='danceability', y='popularity')
plt.title('Danceability vs Popularity')
plt.xlabel('Danceability')
plt.ylabel('Popularity')
plt.show()

# Popularity vs Tempo
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_clean, x='tempo', y='popularity')
plt.title('Tempo vs Popularity')
plt.xlabel('Tempo')
plt.ylabel('Popularity')
plt.show()

# Selecting features for PCA
features = ['danceability', 'energy', 'tempo', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'valence']
X = df_clean[features]

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA(n_components=2)  # Reduce to 2 principal components for visualization
pca_components = pca.fit_transform(X_scaled)

# Plotting the first two principal components
plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1], hue=df_clean['album'], palette='Set1', s=80)
plt.title('PCA: Songs in 2D (Reduced Dimensions)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Explained variance ratio of the principal components
print(f'Explained Variance by Principal Components: {pca.explained_variance_ratio_}')

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Determine optimal number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Apply K-means with the optimal number of clusters (let's assume k=3 here for example)
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Assign clusters to the dataset
df_clean['Cluster'] = clusters

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1], hue=df_clean['Cluster'], palette='Set2', s=80)
plt.title('Clustered Songs Using K-Means (PCA Projection)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


# Select features for clustering
features = ['danceability', 'energy', 'tempo', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'valence']
X = df[features]

# Standardizing the features (important for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method to find the optimal number of clusters
wcss = []  # Within-cluster sum of squares (inertia)
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Method graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Within-cluster sum of squares)')
plt.show()

# Based on the Elbow Method, choose the optimal k (let's assume k=3 for this example)

# --- Step 2: Apply K-Means Clustering ---
optimal_k = 3  # Replace with the chosen number from the elbow method
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# --- Step 3: Define Each Cluster Based on Features ---

# We can analyze the cluster centers (centroids) to define the characteristics of each cluster
centroids = kmeans.cluster_centers_

# Create a DataFrame with centroids for easier interpretation
centroids_df = pd.DataFrame(centroids, columns=features)
print("Cluster Centroids (Mean Feature Values for Each Cluster):")
print(centroids_df)

# To further analyze, visualize the clusters in a reduced dimensionality space (PCA)
pca = PCA(n_components=2)  # Reduce to 2D for visualization
pca_components = pca.fit_transform(X_scaled)

# Plot the clusters in 2D (PCA projection)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1], hue=df['Cluster'], palette='Set2', s=80)
plt.title(f'K-Means Clusters in 2D (PCA Projection, k={optimal_k})')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# --- Step 4: Evaluate the Clustering Quality (Silhouette Score) ---

# Compute the silhouette score to evaluate how well-separated the clusters are
sil_score = silhouette_score(X_scaled, df['Cluster'])
print(f"Silhouette Score: {sil_score}")
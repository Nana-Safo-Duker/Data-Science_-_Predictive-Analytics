"""
Machine Learning Analysis
Customer Data Analysis - Customer Segmentation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

# Set paths
data_path = Path('../../data/Customers.csv')
results_path = Path('../../results')
results_path.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(data_path)

def prepare_data():
    """Prepare data for ML analysis"""
    print("=" * 50)
    print("DATA PREPARATION")
    print("=" * 50)
    
    # Encode categorical variables
    le_country = LabelEncoder()
    le_city = LabelEncoder()
    
    df_encoded = df.copy()
    df_encoded['Country_encoded'] = le_country.fit_transform(df['Country'])
    df_encoded['City_encoded'] = le_city.fit_transform(df['City'])
    
    # Select features
    features = ['CustomerID', 'Country_encoded', 'City_encoded']
    X = df_encoded[features].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Data shape: {X_scaled.shape}")
    print(f"Features: {features}")
    
    return X_scaled, df_encoded, le_country, le_city, features

def determine_optimal_clusters(X_scaled, max_clusters=10):
    """Determine optimal number of clusters using Elbow method and Silhouette score"""
    print("\n" + "=" * 50)
    print("DETERMINING OPTIMAL NUMBER OF CLUSTERS")
    print("=" * 50)
    
    inertias = []
    silhouette_scores = []
    K_range = range(2, min(max_clusters + 1, len(X_scaled)))
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    # Plot Elbow method
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Inertia')
    axes[0].set_title('Elbow Method for Optimal k')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Score for Optimal k')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_path / 'optimal_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Find optimal k (highest silhouette score)
    optimal_k = K_range[np.argmax(silhouette_scores)]
    print(f"\nOptimal number of clusters: {optimal_k}")
    print(f"Silhouette score: {max(silhouette_scores):.4f}")
    
    return optimal_k

def kmeans_clustering(X_scaled, n_clusters, df_encoded):
    """Perform K-Means clustering"""
    print("\n" + "=" * 50)
    print("K-MEANS CLUSTERING")
    print("=" * 50)
    
    # Fit K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to dataframe
    df_clustered = df_encoded.copy()
    df_clustered['Cluster'] = clusters
    
    # Evaluate clustering
    silhouette = silhouette_score(X_scaled, clusters)
    davies_bouldin = davies_bouldin_score(X_scaled, clusters)
    calinski_harabasz = calinski_harabasz_score(X_scaled, clusters)
    
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Davies-Bouldin Score: {davies_bouldin:.4f}")
    print(f"Calinski-Harabasz Score: {calinski_harabasz:.4f}")
    
    # Cluster statistics
    print("\n=== Cluster Statistics ===")
    cluster_stats = df_clustered.groupby('Cluster').agg({
        'CustomerID': 'count',
        'Country_encoded': 'mean',
        'City_encoded': 'mean'
    }).round(2)
    cluster_stats.columns = ['Customer_Count', 'Avg_Country_encoded', 'Avg_City_encoded']
    print(cluster_stats)
    
    # Country distribution by cluster
    print("\n=== Country Distribution by Cluster ===")
    country_cluster = pd.crosstab(df_clustered['Cluster'], df_clustered['Country'])
    print(country_cluster)
    
    # Visualize clusters using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6, s=100)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                c='red', marker='x', s=200, linewidths=3, label='Centroids')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title(f'K-Means Clustering (k={n_clusters})')
    plt.colorbar(scatter, label='Cluster')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_path / 'kmeans_clustering.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return df_clustered, kmeans

def hierarchical_clustering(X_scaled, n_clusters, df_encoded):
    """Perform Hierarchical Clustering"""
    print("\n" + "=" * 50)
    print("HIERARCHICAL CLUSTERING")
    print("=" * 50)
    
    # Fit Hierarchical Clustering
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    clusters = hierarchical.fit_predict(X_scaled)
    
    # Add cluster labels
    df_hierarchical = df_encoded.copy()
    df_hierarchical['Cluster'] = clusters
    
    # Evaluate clustering
    silhouette = silhouette_score(X_scaled, clusters)
    davies_bouldin = davies_bouldin_score(X_scaled, clusters)
    calinski_harabasz = calinski_harabasz_score(X_scaled, clusters)
    
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Davies-Bouldin Score: {davies_bouldin:.4f}")
    print(f"Calinski-Harabasz Score: {calinski_harabasz:.4f}")
    
    # Visualize clusters using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(12, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='plasma', alpha=0.6, s=100)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title(f'Hierarchical Clustering (k={n_clusters})')
    plt.colorbar(label='Cluster')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_path / 'hierarchical_clustering.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return df_hierarchical

def cluster_analysis(df_clustered):
    """Analyze cluster characteristics"""
    print("\n" + "=" * 50)
    print("CLUSTER ANALYSIS")
    print("=" * 50)
    
    # Country distribution by cluster
    print("\n=== Top Countries by Cluster ===")
    for cluster_id in sorted(df_clustered['Cluster'].unique()):
        cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
        top_countries = cluster_data['Country'].value_counts().head(5)
        print(f"\nCluster {cluster_id} ({len(cluster_data)} customers):")
        for country, count in top_countries.items():
            print(f"  {country}: {count} customers ({(count/len(cluster_data)*100):.1f}%)")
    
    # City distribution by cluster
    print("\n=== Top Cities by Cluster ===")
    for cluster_id in sorted(df_clustered['Cluster'].unique()):
        cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
        top_cities = cluster_data['City'].value_counts().head(5)
        print(f"\nCluster {cluster_id}:")
        for city, count in top_cities.items():
            print(f"  {city}: {count} customers ({(count/len(cluster_data)*100):.1f}%)")
    
    # Visualize cluster composition
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Country distribution
    country_cluster = pd.crosstab(df_clustered['Cluster'], df_clustered['Country'])
    country_cluster.plot(kind='bar', ax=axes[0], stacked=True, colormap='tab20')
    axes[0].set_xlabel('Cluster')
    axes[0].set_ylabel('Number of Customers')
    axes[0].set_title('Country Distribution by Cluster')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Cluster sizes
    cluster_sizes = df_clustered['Cluster'].value_counts().sort_index()
    cluster_sizes.plot(kind='bar', ax=axes[1], color='steelblue', edgecolor='black')
    axes[1].set_xlabel('Cluster')
    axes[1].set_ylabel('Number of Customers')
    axes[1].set_title('Cluster Sizes')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_path / 'cluster_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function"""
    # Prepare data
    X_scaled, df_encoded, le_country, le_city, features = prepare_data()
    
    # Determine optimal number of clusters
    optimal_k = determine_optimal_clusters(X_scaled, max_clusters=min(8, len(X_scaled)//2))
    
    # Perform K-Means clustering
    df_clustered, kmeans = kmeans_clustering(X_scaled, optimal_k, df_encoded)
    
    # Perform Hierarchical clustering
    df_hierarchical = hierarchical_clustering(X_scaled, optimal_k, df_encoded)
    
    # Cluster analysis
    cluster_analysis(df_clustered)
    
    # Save results
    df_clustered.to_csv(results_path / 'clustered_customers.csv', index=False)
    print(f"\n✓ Clustered data saved to {results_path / 'clustered_customers.csv'}")
    
    print("\n" + "=" * 50)
    print("ML ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"\n1. Optimal number of clusters: {optimal_k}")
    print("2. Performed K-Means clustering")
    print("3. Performed Hierarchical clustering")
    print("4. Analyzed cluster characteristics")
    print("5. Generated visualizations and saved results")

if __name__ == "__main__":
    main()


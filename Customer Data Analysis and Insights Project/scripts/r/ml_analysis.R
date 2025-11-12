# Machine Learning Analysis
# Customer Data Analysis - Customer Segmentation

# Load libraries
library(tidyverse)
library(ggplot2)
library(dplyr)
library(cluster)
library(factoextra)
library(NbClust)
library(corrplot)

# Set paths
data_path <- "../../data/Customers.csv"
results_path <- "../../results"

# Create results directory if it doesn't exist
if (!dir.exists(results_path)) {
  dir.create(results_path, recursive = TRUE)
}

# Load data
df <- read.csv(data_path, stringsAsFactors = FALSE)

cat("==================================================\n")
cat("MACHINE LEARNING ANALYSIS\n")
cat("==================================================\n\n")

# 1. Data Preparation
cat("=== DATA PREPARATION ===\n")

# Encode categorical variables
df_encoded <- df
if ("Country" %in% colnames(df)) {
  df_encoded$Country_encoded <- as.numeric(as.factor(df$Country))
}
if ("City" %in% colnames(df)) {
  df_encoded$City_encoded <- as.numeric(as.factor(df$City))
}

# Select features
features <- c("CustomerID")
if ("Country_encoded" %in% colnames(df_encoded)) {
  features <- c(features, "Country_encoded")
}
if ("City_encoded" %in% colnames(df_encoded)) {
  features <- c(features, "City_encoded")
}

X <- df_encoded[, features, drop = FALSE]

# Standardize features
X_scaled <- scale(X)

cat("Data shape:", dim(X_scaled), "\n")
cat("Features:", paste(features, collapse = ", "), "\n")

# 2. Determine Optimal Number of Clusters
cat("\n=== DETERMINING OPTIMAL NUMBER OF CLUSTERS ===\n")

# Elbow method and Silhouette score
max_clusters <- min(8, nrow(X_scaled) %/% 2)
K_range <- 2:max_clusters

inertias <- numeric(length(K_range))
silhouette_scores <- numeric(length(K_range))

for (i in seq_along(K_range)) {
  k <- K_range[i]
  kmeans_result <- kmeans(X_scaled, centers = k, nstart = 10)
  inertias[i] <- kmeans_result$tot.withinss
  silhouette_scores[i] <- mean(silhouette(kmeans_result$cluster, dist(X_scaled))[, 3])
}

# Plot Elbow method and Silhouette score
png(file = file.path(results_path, "optimal_clusters.png"), 
    width = 1600, height = 600, res = 300)
par(mfrow = c(1, 2))
plot(K_range, inertias, type = "b", pch = 19, col = "steelblue",
     xlab = "Number of Clusters (k)", ylab = "Inertia",
     main = "Elbow Method for Optimal k")
plot(K_range, silhouette_scores, type = "b", pch = 19, col = "coral",
     xlab = "Number of Clusters (k)", ylab = "Silhouette Score",
     main = "Silhouette Score for Optimal k")
dev.off()

# Find optimal k
optimal_k <- K_range[which.max(silhouette_scores)]
cat("Optimal number of clusters:", optimal_k, "\n")
cat("Silhouette score:", max(silhouette_scores), "\n")

# 3. K-Means Clustering
cat("\n=== K-MEANS CLUSTERING ===\n")

kmeans_result <- kmeans(X_scaled, centers = optimal_k, nstart = 10)
clusters <- kmeans_result$cluster

# Add cluster labels to dataframe
df_clustered <- df_encoded
df_clustered$Cluster <- clusters

# Evaluate clustering
silhouette_score <- mean(silhouette(clusters, dist(X_scaled))[, 3])
cat("Silhouette Score:", silhouette_score, "\n")

# Cluster statistics
cat("\n=== Cluster Statistics ===\n")
cluster_stats <- df_clustered %>%
  group_by(Cluster) %>%
  summarise(
    Customer_Count = n(),
    Avg_Country_encoded = mean(Country_encoded, na.rm = TRUE),
    Avg_City_encoded = mean(City_encoded, na.rm = TRUE),
    .groups = "drop"
  )
print(cluster_stats)

# Country distribution by cluster
if ("Country" %in% colnames(df_clustered)) {
  cat("\n=== Country Distribution by Cluster ===\n")
  country_cluster <- table(df_clustered$Cluster, df_clustered$Country)
  print(country_cluster)
}

# Visualize clusters using PCA
pca_result <- prcomp(X_scaled)
X_pca <- pca_result$x[, 1:2]

png(file = file.path(results_path, "kmeans_clustering.png"), 
    width = 1200, height = 800, res = 300)
plot(X_pca, col = clusters, pch = 19, 
     main = paste("K-Means Clustering (k =", optimal_k, ")"),
     xlab = paste("PC1 (", round(summary(pca_result)$importance[2, 1] * 100, 2), "% variance)", sep = ""),
     ylab = paste("PC2 (", round(summary(pca_result)$importance[2, 2] * 100, 2), "% variance)", sep = ""))
points(kmeans_result$centers[, 1:2], col = "red", pch = 4, cex = 2, lwd = 3)
legend("topright", legend = paste("Cluster", 1:optimal_k), 
       col = 1:optimal_k, pch = 19, title = "Clusters")
dev.off()

# 4. Hierarchical Clustering
cat("\n=== HIERARCHICAL CLUSTERING ===\n")

# Compute distance matrix
dist_matrix <- dist(X_scaled, method = "euclidean")

# Perform hierarchical clustering
hierarchical_result <- hclust(dist_matrix, method = "ward.D2")
clusters_hierarchical <- cutree(hierarchical_result, k = optimal_k)

# Add cluster labels
df_hierarchical <- df_encoded
df_hierarchical$Cluster <- clusters_hierarchical

# Evaluate clustering
silhouette_score_hierarchical <- mean(silhouette(clusters_hierarchical, dist_matrix)[, 3])
cat("Silhouette Score:", silhouette_score_hierarchical, "\n")

# Visualize dendrogram
png(file = file.path(results_path, "hierarchical_dendrogram.png"), 
    width = 1400, height = 800, res = 300)
plot(hierarchical_result, main = "Hierarchical Clustering Dendrogram",
     xlab = "Customers", sub = "", cex = 0.6)
rect.hclust(hierarchical_result, k = optimal_k, border = "red")
dev.off()

# Visualize clusters
png(file = file.path(results_path, "hierarchical_clustering.png"), 
    width = 1200, height = 800, res = 300)
plot(X_pca, col = clusters_hierarchical, pch = 19,
     main = paste("Hierarchical Clustering (k =", optimal_k, ")"),
     xlab = paste("PC1 (", round(summary(pca_result)$importance[2, 1] * 100, 2), "% variance)", sep = ""),
     ylab = paste("PC2 (", round(summary(pca_result)$importance[2, 2] * 100, 2), "% variance)", sep = ""))
legend("topright", legend = paste("Cluster", 1:optimal_k), 
       col = 1:optimal_k, pch = 19, title = "Clusters")
dev.off()

# 5. Cluster Analysis
cat("\n=== CLUSTER ANALYSIS ===\n")

# Country distribution by cluster
if ("Country" %in% colnames(df_clustered)) {
  cat("\n=== Top Countries by Cluster ===\n")
  for (cluster_id in sort(unique(df_clustered$Cluster))) {
    cluster_data <- df_clustered[df_clustered$Cluster == cluster_id, ]
    top_countries <- cluster_data %>%
      count(Country) %>%
      arrange(desc(n)) %>%
      head(5)
    
    cat("\nCluster", cluster_id, "(", nrow(cluster_data), "customers):\n", sep = "")
    for (i in 1:nrow(top_countries)) {
      country <- top_countries$Country[i]
      count <- top_countries$n[i]
      percentage <- (count / nrow(cluster_data)) * 100
      cat("  ", country, ": ", count, " customers (", round(percentage, 1), "%)\n", sep = "")
    }
  }
}

# City distribution by cluster
if ("City" %in% colnames(df_clustered)) {
  cat("\n=== Top Cities by Cluster ===\n")
  for (cluster_id in sort(unique(df_clustered$Cluster))) {
    cluster_data <- df_clustered[df_clustered$Cluster == cluster_id, ]
    top_cities <- cluster_data %>%
      count(City) %>%
      arrange(desc(n)) %>%
      head(5)
    
    cat("\nCluster", cluster_id, ":\n", sep = "")
    for (i in 1:nrow(top_cities)) {
      city <- top_cities$City[i]
      count <- top_cities$n[i]
      percentage <- (count / nrow(cluster_data)) * 100
      cat("  ", city, ": ", count, " customers (", round(percentage, 1), "%)\n", sep = "")
    }
  }
}

# Visualize cluster composition
if ("Country" %in% colnames(df_clustered)) {
  png(file = file.path(results_path, "cluster_analysis.png"), 
      width = 1600, height = 600, res = 300)
  par(mfrow = c(1, 2))
  
  # Country distribution
  country_cluster_table <- table(df_clustered$Cluster, df_clustered$Country)
  barplot(country_cluster_table, main = "Country Distribution by Cluster",
          xlab = "Country", ylab = "Number of Customers", 
          legend.text = paste("Cluster", 1:optimal_k),
          args.legend = list(x = "topright", cex = 0.8))
  
  # Cluster sizes
  cluster_sizes <- table(df_clustered$Cluster)
  barplot(cluster_sizes, main = "Cluster Sizes",
          xlab = "Cluster", ylab = "Number of Customers",
          col = "steelblue", border = "black")
  
  dev.off()
}

# Save results
write.csv(df_clustered, file = file.path(results_path, "clustered_customers.csv"), row.names = FALSE)
cat("\n✓ Clustered data saved to", file.path(results_path, "clustered_customers.csv"), "\n")

cat("\n==================================================\n")
cat("ML ANALYSIS SUMMARY\n")
cat("==================================================\n")
cat("\n1. Optimal number of clusters:", optimal_k, "\n")
cat("2. Performed K-Means clustering\n")
cat("3. Performed Hierarchical clustering\n")
cat("4. Analyzed cluster characteristics\n")
cat("5. Generated visualizations and saved results\n")

cat("\n✓ ML analysis completed successfully!\n")


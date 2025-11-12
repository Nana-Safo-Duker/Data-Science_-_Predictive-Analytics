# Univariate, Bivariate, and Multivariate Analysis
# Customer Data Analysis

# Load libraries
library(tidyverse)
library(ggplot2)
library(dplyr)
library(corrplot)
library(FactoMineR)
library(factoextra)

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
cat("UNIVARIATE, BIVARIATE, AND MULTIVARIATE ANALYSIS\n")
cat("==================================================\n\n")

# 1. Univariate Analysis
cat("=== UNIVARIATE ANALYSIS ===\n\n")

# Numerical variables
numerical_cols <- sapply(df, is.numeric)
if (sum(numerical_cols) > 0) {
  cat("=== Numerical Variables Univariate Analysis ===\n")
  for (col in names(df)[numerical_cols]) {
    cat("\n", col, ":\n", sep = "")
    cat("  Mean:", round(mean(df[[col]], na.rm = TRUE), 2), "\n")
    cat("  Median:", round(median(df[[col]], na.rm = TRUE), 2), "\n")
    cat("  Std Dev:", round(sd(df[[col]], na.rm = TRUE), 2), "\n")
    cat("  Min:", min(df[[col]], na.rm = TRUE), "\n")
    cat("  Max:", max(df[[col]], na.rm = TRUE), "\n")
    cat("  Range:", max(df[[col]], na.rm = TRUE) - min(df[[col]], na.rm = TRUE), "\n")
    
    # Visualization
    png(file = file.path(results_path, paste0("univariate_", col, ".png")), 
        width = 1800, height = 600, res = 300)
    par(mfrow = c(1, 3))
    hist(df[[col]], breaks = 20, main = paste("Histogram of", col),
         xlab = col, ylab = "Frequency", col = "steelblue", border = "black")
    boxplot(df[[col]], main = paste("Box Plot of", col), ylab = col, col = "lightblue")
    plot(density(df[[col]], na.rm = TRUE), main = paste("Density Plot of", col),
         xlab = col, col = "steelblue", lwd = 2)
    dev.off()
  }
}

# Categorical variables
cat("\n=== Categorical Variables Univariate Analysis ===\n")
categorical_cols <- sapply(df, is.character) | sapply(df, is.factor)
for (col in names(df)[categorical_cols]) {
  cat("\n", col, ":\n", sep = "")
  value_counts <- table(df[[col]])
  cat("  Total categories:", length(value_counts), "\n")
  cat("  Most frequent:", names(sort(value_counts, decreasing = TRUE)[1]), "\n")
  cat("  Frequency:", max(value_counts), "\n")
  cat("  Frequency percentage:", round((max(value_counts) / nrow(df)) * 100, 2), "%\n")
  
  # Visualization
  value_counts_df <- data.frame(
    Category = names(value_counts),
    Count = as.numeric(value_counts)
  ) %>%
    arrange(desc(Count))
  
  if (nrow(value_counts_df) > 20) {
    value_counts_df <- head(value_counts_df, 20)
  }
  
  png(file = file.path(results_path, paste0("univariate_", col, ".png")), 
      width = 1400, height = 800, res = 300)
  ggplot(value_counts_df, aes(x = reorder(Category, Count), y = Count)) +
    geom_bar(stat = "identity", fill = "coral", color = "black") +
    coord_flip() +
    labs(x = col, y = "Frequency", title = paste(col, "Distribution")) +
    theme_minimal() +
    theme(axis.text.y = element_text(size = 8))
  dev.off()
}

# 2. Bivariate Analysis
cat("\n==================================================\n")
cat("BIVARIATE ANALYSIS\n")
cat("==================================================\n\n")

# Country vs City
if ("Country" %in% colnames(df) && "City" %in% colnames(df)) {
  cat("=== Country vs City Analysis ===\n")
  country_city <- table(df$Country, df$City)
  cat("Contingency table shape:", dim(country_city), "\n")
  
  # Flatten and sort
  country_city_df <- as.data.frame.table(country_city) %>%
    filter(Freq > 0) %>%
    arrange(desc(Freq))
  colnames(country_city_df) <- c("Country", "City", "Count")
  
  cat("\nTop country-city combinations:\n")
  print(head(country_city_df, 10))
  
  # Visualization
  png(file = file.path(results_path, "bivariate_country_city.png"), 
      width = 1600, height = 1000, res = 300)
  country_city_long <- country_city_df %>%
    filter(Count > 0) %>%
    head(50)  # Limit for readability
  ggplot(country_city_long, aes(x = City, y = Country, fill = Count)) +
    geom_tile() +
    scale_fill_gradient(low = "white", high = "red") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 6),
          axis.text.y = element_text(size = 6)) +
    labs(title = "Country vs City Heatmap")
  dev.off()
  
  # Chi-square test
  cat("\n=== Chi-Square Test: Country vs City ===\n")
  chi2_test <- chisq.test(country_city)
  cat("Chi-square statistic:", round(chi2_test$statistic, 4), "\n")
  cat("p-value:", round(chi2_test$p.value, 4), "\n")
  cat("Degrees of freedom:", chi2_test$parameter, "\n")
  
  alpha <- 0.05
  if (chi2_test$p.value < alpha) {
    cat("Result: Significant association between Country and City (p < ", alpha, ")\n", sep = "")
  } else {
    cat("Result: No significant association (p >= ", alpha, ")\n", sep = "")
  }
}

# Customer ID vs Country
if ("CustomerID" %in% colnames(df) && "Country" %in% colnames(df)) {
  cat("\n=== Customer ID Distribution by Country ===\n")
  country_customer <- df %>%
    group_by(Country) %>%
    summarise(Customer_Count = n()) %>%
    arrange(desc(Customer_Count))
  print(country_customer)
  
  png(file = file.path(results_path, "bivariate_customer_country.png"), 
      width = 1400, height = 800, res = 300)
  ggplot(country_customer, aes(x = reorder(Country, Customer_Count), y = Customer_Count)) +
    geom_bar(stat = "identity", fill = "steelblue", color = "black") +
    coord_flip() +
    labs(x = "Country", y = "Number of Customers", 
         title = "Customer Distribution by Country") +
    theme_minimal() +
    theme(axis.text.y = element_text(size = 8))
  dev.off()
}

# 3. Multivariate Analysis
cat("\n==================================================\n")
cat("MULTIVARIATE ANALYSIS\n")
cat("==================================================\n\n")

# Encode categorical variables
df_encoded <- df
if ("Country" %in% colnames(df)) {
  df_encoded$Country_encoded <- as.numeric(as.factor(df$Country))
}
if ("City" %in% colnames(df)) {
  df_encoded$City_encoded <- as.numeric(as.factor(df$City))
}

# Select features for multivariate analysis
features <- c("CustomerID")
if ("Country_encoded" %in% colnames(df_encoded)) {
  features <- c(features, "Country_encoded")
}
if ("City_encoded" %in% colnames(df_encoded)) {
  features <- c(features, "City_encoded")
}

X <- df_encoded[, features, drop = FALSE]

# Correlation matrix
cat("=== Correlation Matrix ===\n")
corr_matrix <- cor(X)
print(corr_matrix)

# Visualization
png(file = file.path(results_path, "multivariate_correlation.png"), 
    width = 1000, height = 800, res = 300)
corrplot(corr_matrix, method = "circle", type = "upper", 
         order = "hclust", tl.cex = 0.8, tl.col = "black")
dev.off()

# PCA Analysis
cat("\n=== Principal Component Analysis (PCA) ===\n")
X_scaled <- scale(X)
pca_result <- prcomp(X_scaled)

cat("Standard deviations:\n")
print(pca_result$sdev)
cat("\nProportion of variance:\n")
print(summary(pca_result)$importance[2, ])
cat("\nCumulative proportion:\n")
print(summary(pca_result)$importance[3, ])

# PCA Visualization
png(file = file.path(results_path, "multivariate_pca.png"), 
    width = 1200, height = 800, res = 300)
fviz_pca_biplot(pca_result, repel = TRUE, col.var = "steelblue", 
                col.ind = "gray", title = "PCA: Biplot")
dev.off()

# Country-City-CustomerID relationship
if ("Country" %in% colnames(df) && "City" %in% colnames(df)) {
  cat("\n=== Country-City-CustomerID Relationship ===\n")
  country_city_summary <- df %>%
    group_by(Country, City) %>%
    summarise(
      Customer_Count = n(),
      Min_CustomerID = min(CustomerID),
      Max_CustomerID = max(CustomerID),
      .groups = "drop"
    ) %>%
    arrange(desc(Customer_Count))
  print(head(country_city_summary, 10))
}

cat("\n==================================================\n")
cat("ANALYSIS SUMMARY\n")
cat("==================================================\n")
cat("\n1. Univariate Analysis:\n")
cat("   • Analyzed individual variables (numerical and categorical)\n")
cat("   • Generated distribution plots and summary statistics\n")
cat("\n2. Bivariate Analysis:\n")
cat("   • Examined relationships between two variables\n")
cat("   • Performed chi-square tests for independence\n")
cat("   • Created contingency tables and heatmaps\n")
cat("\n3. Multivariate Analysis:\n")
cat("   • Analyzed relationships among multiple variables\n")
cat("   • Performed PCA for dimensionality reduction\n")
cat("   • Created correlation matrices\n")

cat("\n✓ Analysis completed successfully!\n")


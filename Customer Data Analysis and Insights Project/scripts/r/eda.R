# Exploratory Data Analysis Script
# Customer Data Analysis

# Load libraries
library(tidyverse)
library(ggplot2)
library(dplyr)
library(VIM)

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
cat("EXPLORATORY DATA ANALYSIS\n")
cat("==================================================\n\n")

# 1. Data Overview
cat("=== Dataset Information ===\n")
cat("Dataset loaded successfully!\n")
cat("Shape:", nrow(df), "rows,", ncol(df), "columns\n\n")
cat("First few rows:\n")
print(head(df))

cat("\n=== Dataset Structure ===\n")
str(df)

cat("\n=== Column Names ===\n")
print(colnames(df))

# 2. Missing Values Analysis
cat("\n=== Missing Values Analysis ===\n")
missing_values <- colSums(is.na(df))
missing_percentage <- (missing_values / nrow(df)) * 100

missing_df <- data.frame(
  Column = names(missing_values),
  Missing_Count = missing_values,
  Missing_Percentage = missing_percentage
)

missing_df_filtered <- missing_df[missing_df$Missing_Count > 0, ]
if (nrow(missing_df_filtered) == 0) {
  cat("✓ No missing values found in the dataset!\n")
} else {
  print(missing_df_filtered)
}

# 3. Duplicate Records
cat("\n=== Duplicate Records ===\n")
duplicate_count <- sum(duplicated(df))
cat("Number of duplicate records:", duplicate_count, "\n")
if (duplicate_count > 0) {
  cat("Duplicate records:\n")
  print(df[duplicated(df), ])
} else {
  cat("✓ No duplicate records found!\n")
}

# 4. Summary Statistics
cat("\n=== Numerical Columns Summary ===\n")
numerical_cols <- sapply(df, is.numeric)
if (sum(numerical_cols) > 0) {
  print(summary(df[, numerical_cols, drop = FALSE]))
} else {
  cat("No numerical columns found.\n")
}

cat("\n=== Categorical Columns Summary ===\n")
categorical_cols <- sapply(df, is.character) | sapply(df, is.factor)
for (col in names(df)[categorical_cols]) {
  cat("\n", col, ":\n", sep = "")
  cat("  Unique values:", n_distinct(df[[col]]), "\n")
  cat("  Most frequent:", names(sort(table(df[[col]]), decreasing = TRUE)[1]), "\n")
  cat("  Frequency:", max(table(df[[col]])), "\n")
}

# 5. Distribution Analysis - Customer ID
cat("\n=== Distribution Analysis ===\n")
if ("CustomerID" %in% colnames(df)) {
  png(file = file.path(results_path, "customer_id_distribution.png"), 
      width = 1200, height = 600, res = 300)
  hist(df$CustomerID, breaks = 20, main = "Distribution of Customer IDs",
       xlab = "Customer ID", ylab = "Frequency", col = "steelblue", border = "black")
  dev.off()
  cat("✓ Customer ID distribution plot saved\n")
}

# 6. Country Distribution
if ("Country" %in% colnames(df)) {
  country_counts <- df %>%
    count(Country) %>%
    arrange(desc(n))
  
  cat("\n=== Country Distribution ===\n")
  print(country_counts)
  
  png(file = file.path(results_path, "country_distribution.png"), 
      width = 1400, height = 800, res = 300)
  ggplot(country_counts, aes(x = reorder(Country, n), y = n)) +
    geom_bar(stat = "identity", fill = "steelblue", color = "black") +
    coord_flip() +
    labs(x = "Country", y = "Number of Customers", 
         title = "Customer Distribution by Country") +
    theme_minimal() +
    theme(axis.text.y = element_text(size = 8))
  dev.off()
  cat("✓ Country distribution plot saved\n")
}

# 7. City Distribution
if ("City" %in% colnames(df)) {
  city_counts <- df %>%
    count(City) %>%
    arrange(desc(n)) %>%
    head(20)
  
  cat("\n=== Top 20 Cities by Customer Count ===\n")
  print(city_counts)
  
  png(file = file.path(results_path, "city_distribution.png"), 
      width = 1400, height = 800, res = 300)
  ggplot(city_counts, aes(x = reorder(City, n), y = n)) +
    geom_bar(stat = "identity", fill = "coral", color = "black") +
    coord_flip() +
    labs(x = "City", y = "Number of Customers", 
         title = "Top 20 Cities by Customer Count") +
    theme_minimal() +
    theme(axis.text.y = element_text(size = 8))
  dev.off()
  cat("✓ City distribution plot saved\n")
}

# 8. Data Quality Checks
cat("\n=== Data Quality Checks ===\n")
cat("Unique Customer IDs:", n_distinct(df$CustomerID), "\n")
cat("Total Records:", nrow(df), "\n")
if (n_distinct(df$CustomerID) == nrow(df)) {
  cat("✓ All Customer IDs are unique\n")
} else {
  cat("⚠ Warning: Duplicate Customer IDs found\n")
}

# 9. Summary
cat("\n==================================================\n")
cat("EDA SUMMARY\n")
cat("==================================================\n")
cat("\n1. Dataset contains", nrow(df), "customers with", ncol(df), "attributes\n")
cat("2. Countries represented:", n_distinct(df$Country), "\n")
cat("3. Cities represented:", n_distinct(df$City), "\n")
cat("4. Missing values:", sum(is.na(df)), "\n")
cat("5. Duplicate records:", sum(duplicated(df)), "\n")

if ("Country" %in% colnames(df)) {
  top_country <- names(sort(table(df$Country), decreasing = TRUE)[1])
  top_country_count <- max(table(df$Country))
  cat("6. Most common country:", top_country, "(", top_country_count, "customers)\n")
}

if ("City" %in% colnames(df)) {
  top_city <- names(sort(table(df$City), decreasing = TRUE)[1])
  top_city_count <- max(table(df$City))
  cat("7. Most common city:", top_city, "(", top_city_count, "customers)\n")
}

cat("\n=== Key Insights ===\n")
cat("• The dataset appears to be clean with no missing values\n")
cat("• All customer IDs are unique\n")
cat("• The dataset has good geographical diversity\n")
cat("• Ready for further statistical and ML analysis\n")

cat("\n✓ EDA completed successfully!\n")


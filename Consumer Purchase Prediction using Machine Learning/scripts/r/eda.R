# Exploratory Data Analysis Script
# Consumer Purchase Prediction Project

# Load necessary libraries
library(dplyr)
library(ggplot2)
library(corrplot)
library(VIM)

# Function to find project root by looking for data directory
find_project_root <- function() {
  # Start from current working directory
  current_dir <- getwd()
  max_levels <- 10
  project_marker <- file.path("Consumer Purchase Prediction", "Consumer Purchase Prediction", "data", "Advertisement.csv")
  
  for (i in 1:max_levels) {
    # Check for the data file in nested structure
    if (file.exists(file.path(current_dir, project_marker))) {
      return(current_dir)
    }
    # Check for data file in current directory structure
    if (file.exists(file.path(current_dir, "data", "Advertisement.csv"))) {
      return(current_dir)
    }
    # Check if we're in the Consumer Purchase Prediction directory
    if (basename(current_dir) == "Consumer Purchase Prediction") {
      if (file.exists(file.path(current_dir, "Consumer Purchase Prediction", "data", "Advertisement.csv"))) {
        return(current_dir)
      }
      if (file.exists(file.path(current_dir, "data", "Advertisement.csv"))) {
        return(current_dir)
      }
    }
    # Move up one level
    parent_dir <- dirname(current_dir)
    if (parent_dir == current_dir) break  # Reached root
    current_dir <- parent_dir
  }
  return(NULL)
}

# Set working directory to project root
project_root <- find_project_root()
if (!is.null(project_root)) {
  setwd(project_root)
  cat("Working directory set to:", getwd(), "\n")
} else {
  cat("Warning: Could not find project root. Using current directory:", getwd(), "\n")
}

# Load the dataset - try multiple possible paths
data_paths <- c(
  file.path("Consumer Purchase Prediction", "Consumer Purchase Prediction", "data", "Advertisement.csv"),
  file.path("data", "Advertisement.csv"),
  "Advertisement.csv"
)

data_path <- NULL
for (path in data_paths) {
  if (file.exists(path)) {
    data_path <- path
    break
  }
}

if (is.null(data_path)) {
  stop(paste("Cannot find Advertisement.csv. Searched in:\n",
             paste("  -", data_paths, collapse = "\n"),
             "\nCurrent working directory:", getwd()))
}

df <- read.csv(data_path, stringsAsFactors = TRUE)

cat("Dataset loaded successfully from:", data_path, "\n")
cat("Shape:", nrow(df), "rows,", ncol(df), "columns\n\n")

# 1. Data Overview
cat(paste(rep("=", 50), collapse=""), "\n")
cat("DATA OVERVIEW\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

cat("First 5 rows:\n")
print(head(df, 5))

cat("\nDataset Structure:\n")
str(df)

cat("\nStatistical Summary:\n")
print(summary(df))

cat("\nMissing Values:\n")
print(colSums(is.na(df)))

cat("\nDuplicate Rows:", sum(duplicated(df)), "\n\n")

# 2. Target Variable Analysis
cat(paste(rep("=", 50), collapse=""), "\n")
cat("TARGET VARIABLE ANALYSIS\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

purchased_counts <- table(df$Purchased)
cat("Purchased Distribution:\n")
print(purchased_counts)
cat("\nPercentage:\n")
print(prop.table(purchased_counts) * 100)

# Create output directory if it doesn't exist
output_paths <- c(
  file.path("Consumer Purchase Prediction", "Consumer Purchase Prediction", "output"),
  "output"
)

output_dir <- NULL
for (path in output_paths) {
  if (dir.exists(path)) {
    output_dir <- path
    break
  }
}

if (is.null(output_dir)) {
  # Create the first possible output directory
  output_dir <- output_paths[1]
  if (!dir.exists(dirname(output_dir))) {
    dir.create(dirname(output_dir), recursive = TRUE)
  }
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  cat("Created output directory:", output_dir, "\n")
}

# Visualize target variable
png(file.path(output_dir, "target_distribution_r.png"), width = 1200, height = 600, res = 300)
par(mfrow = c(1, 2))
barplot(purchased_counts, names.arg = c("No", "Yes"), 
        col = c("skyblue", "coral"),
        main = "Purchased Distribution (Bar Chart)",
        xlab = "Purchased", ylab = "Count")
pie(purchased_counts, labels = c("No", "Yes"), 
    main = "Purchased Distribution (Pie Chart)",
    col = c("skyblue", "coral"))
dev.off()

# 3. Numerical Variables Analysis
cat("\n", paste(rep("=", 50), collapse=""), "\n")
cat("NUMERICAL VARIABLES ANALYSIS\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

png(file.path(output_dir, "numerical_distributions_r.png"), width = 1500, height = 1000, res = 300)
par(mfrow = c(2, 2))
hist(df$Age, main = "Age Distribution (Histogram)", xlab = "Age", col = "lightblue", breaks = 30)
boxplot(df$Age, main = "Age Distribution (Box Plot)", ylab = "Age", col = "lightblue")
hist(df$EstimatedSalary, main = "Estimated Salary Distribution (Histogram)", 
     xlab = "Estimated Salary", col = "lightgreen", breaks = 30)
boxplot(df$EstimatedSalary, main = "Estimated Salary Distribution (Box Plot)", 
        ylab = "Estimated Salary", col = "lightgreen")
dev.off()

# 4. Categorical Variables Analysis
cat(paste(rep("=", 50), collapse=""), "\n")
cat("CATEGORICAL VARIABLES ANALYSIS\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

gender_counts <- table(df$Gender)
cat("Gender Distribution:\n")
print(gender_counts)
cat("\nPercentage:\n")
print(prop.table(gender_counts) * 100)

png(file.path(output_dir, "categorical_distributions_r.png"), width = 1200, height = 600, res = 300)
par(mfrow = c(1, 2))
barplot(gender_counts, col = c("lightblue", "lightpink"),
        main = "Gender Distribution", xlab = "Gender", ylab = "Count")
pie(gender_counts, labels = names(gender_counts),
    main = "Gender Distribution (Pie Chart)",
    col = c("lightblue", "lightpink"))
dev.off()

# 5. Relationship Analysis
cat("\n", paste(rep("=", 50), collapse=""), "\n")
cat("RELATIONSHIP ANALYSIS\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

png(file.path(output_dir, "relationships_r.png"), width = 1500, height = 600, res = 300)
par(mfrow = c(1, 2))
boxplot(Age ~ Purchased, data = df, main = "Age Distribution by Purchase Status",
        xlab = "Purchased", ylab = "Age", names = c("No", "Yes"))
boxplot(EstimatedSalary ~ Purchased, data = df, 
        main = "Salary Distribution by Purchase Status",
        xlab = "Purchased", ylab = "Estimated Salary", names = c("No", "Yes"))
dev.off()

# Scatter plot
png(file.path(output_dir, "scatter_plot_r.png"), width = 1200, height = 800, res = 300)
plot(df$Age, df$EstimatedSalary, 
     col = ifelse(df$Purchased == 1, 
                  adjustcolor("red", alpha.f = 0.6), 
                  adjustcolor("blue", alpha.f = 0.6)),
     pch = 19,
     xlab = "Age", ylab = "Estimated Salary",
     main = "Age vs Estimated Salary (colored by Purchase Status)")
legend("topright", legend = c("No Purchase", "Purchase"), 
       col = c("blue", "red"), pch = 19)
dev.off()

# 6. Correlation Analysis
cat(paste(rep("=", 50), collapse=""), "\n")
cat("CORRELATION ANALYSIS\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

numeric_df <- df[, c("Age", "EstimatedSalary", "Purchased")]
correlation_matrix <- cor(numeric_df)
cat("Correlation Matrix:\n")
print(correlation_matrix)
cat("\nCorrelation with Purchased:\n")
print(sort(correlation_matrix[, "Purchased"], decreasing = TRUE))

png(file.path(output_dir, "correlation_matrix_r.png"), width = 1000, height = 800, res = 300)
corrplot(correlation_matrix, method = "color", type = "upper",
         order = "hclust", tl.cex = 0.8, tl.col = "black",
         addCoef.col = "black", number.cex = 0.7)
dev.off()

cat("\n", paste(rep("=", 50), collapse=""), "\n")
cat("EDA COMPLETED SUCCESSFULLY!\n")
cat(paste(rep("=", 50), collapse=""), "\n")

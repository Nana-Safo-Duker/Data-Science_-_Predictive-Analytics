# Univariate, Bivariate, and Multivariate Analysis Script
# Consumer Purchase Prediction Project

# Load necessary libraries
library(dplyr)
library(ggplot2)
library(corrplot)
library(psych)
library(e1071)

# Function to find project root by looking for data directory
find_project_root <- function() {
  current_dir <- getwd()
  max_levels <- 10
  project_marker <- file.path("Consumer Purchase Prediction", "Consumer Purchase Prediction", "data", "Advertisement.csv")
  
  for (i in 1:max_levels) {
    if (file.exists(file.path(current_dir, project_marker))) {
      return(current_dir)
    }
    if (file.exists(file.path(current_dir, "data", "Advertisement.csv"))) {
      return(current_dir)
    }
    if (basename(current_dir) == "Consumer Purchase Prediction") {
      if (file.exists(file.path(current_dir, "Consumer Purchase Prediction", "data", "Advertisement.csv"))) {
        return(current_dir)
      }
      if (file.exists(file.path(current_dir, "data", "Advertisement.csv"))) {
        return(current_dir)
      }
    }
    parent_dir <- dirname(current_dir)
    if (parent_dir == current_dir) break
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
cat("UNIVARIATE, BIVARIATE, AND MULTIVARIATE ANALYSIS\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

# Get output directory
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
  output_dir <- output_paths[1]
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  cat("Created output directory:", output_dir, "\n")
}

# 1. Univariate Analysis
cat("1. UNIVARIATE ANALYSIS\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

# Age
png(file.path(output_dir, "univariate_age_r.png"), width = 1500, height = 1000, res = 300)
par(mfrow = c(2, 2))
hist(df$Age, main = "Age Distribution (Histogram)", xlab = "Age", col = "lightblue", breaks = 30)
boxplot(df$Age, main = "Age Distribution (Box Plot)", ylab = "Age", col = "lightblue")
qqnorm(df$Age, main = "Age Q-Q Plot")
qqline(df$Age, col = "red")
plot(density(df$Age), main = "Age Density Plot", xlab = "Age")
dev.off()

cat("Age Statistics:\n")
cat("Mean:", mean(df$Age), "\n")
cat("Median:", median(df$Age), "\n")
cat("Std:", sd(df$Age), "\n")
cat("Skewness:", skewness(df$Age), "\n")
cat("Kurtosis:", kurtosis(df$Age), "\n\n")

# Estimated Salary
png(file.path(output_dir, "univariate_salary_r.png"), width = 1500, height = 1000, res = 300)
par(mfrow = c(2, 2))
hist(df$EstimatedSalary, main = "Estimated Salary Distribution (Histogram)", 
     xlab = "Estimated Salary", col = "lightgreen", breaks = 30)
boxplot(df$EstimatedSalary, main = "Estimated Salary Distribution (Box Plot)", 
        ylab = "Estimated Salary", col = "lightgreen")
qqnorm(df$EstimatedSalary, main = "Estimated Salary Q-Q Plot")
qqline(df$EstimatedSalary, col = "red")
plot(density(df$EstimatedSalary), main = "Estimated Salary Density Plot", 
     xlab = "Estimated Salary")
dev.off()

cat("Estimated Salary Statistics:\n")
cat("Mean:", mean(df$EstimatedSalary), "\n")
cat("Median:", median(df$EstimatedSalary), "\n")
cat("Std:", sd(df$EstimatedSalary), "\n")
cat("Skewness:", skewness(df$EstimatedSalary), "\n")
cat("Kurtosis:", kurtosis(df$EstimatedSalary), "\n\n")

# 2. Bivariate Analysis
cat("2. BIVARIATE ANALYSIS\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

# Age vs Purchased
png(file.path(output_dir, "bivariate_age_purchased_r.png"), width = 1800, height = 600, res = 300)
par(mfrow = c(1, 3))
boxplot(Age ~ Purchased, data = df, main = "Age by Purchase Status",
        xlab = "Purchased", ylab = "Age", names = c("No", "Yes"))
stripchart(Age ~ Purchased, data = df, method = "jitter", 
           pch = 19, col = c("blue", "red"), vertical = TRUE,
           main = "Age Distribution by Purchase Status (Strip)")
plot(df$Age[df$Purchased == 0], df$Age[df$Purchased == 1], 
     xlab = "No Purchase", ylab = "Purchase", 
     main = "Age Comparison")
dev.off()

# Salary vs Purchased
png(file.path(output_dir, "bivariate_salary_purchased_r.png"), width = 1800, height = 600, res = 300)
par(mfrow = c(1, 3))
boxplot(EstimatedSalary ~ Purchased, data = df, 
        main = "Estimated Salary by Purchase Status",
        xlab = "Purchased", ylab = "Estimated Salary", names = c("No", "Yes"))
stripchart(EstimatedSalary ~ Purchased, data = df, method = "jitter", 
           pch = 19, col = c("blue", "red"), vertical = TRUE,
           main = "Estimated Salary Distribution by Purchase Status (Strip)")
plot(df$EstimatedSalary[df$Purchased == 0], df$EstimatedSalary[df$Purchased == 1], 
     xlab = "No Purchase", ylab = "Purchase", 
     main = "Estimated Salary Comparison")
dev.off()

# Age vs Salary
png(file.path(output_dir, "bivariate_age_salary_r.png"), width = 1200, height = 800, res = 300)
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

# Statistical tests
t_test_age <- t.test(Age ~ Purchased, data = df)
cat("Age T-test: t =", t_test_age$statistic, ", p-value =", t_test_age$p.value, "\n")

t_test_salary <- t.test(EstimatedSalary ~ Purchased, data = df)
cat("Salary T-test: t =", t_test_salary$statistic, ", p-value =", t_test_salary$p.value, "\n\n")

# 3. Multivariate Analysis
cat("3. MULTIVARIATE ANALYSIS\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

# Correlation matrix
numeric_df <- df[, c("Age", "EstimatedSalary", "Purchased")]
correlation_matrix <- cor(numeric_df)

png(file.path(output_dir, "multivariate_correlation_r.png"), width = 1000, height = 800, res = 300)
corrplot(correlation_matrix, method = "color", type = "upper",
         order = "hclust", tl.cex = 0.8, tl.col = "black",
         addCoef.col = "black", number.cex = 0.7)
dev.off()

cat("Correlation Matrix:\n")
print(correlation_matrix)

# Multivariate visualization by Gender
png(file.path(output_dir, "multivariate_by_gender_r.png"), width = 1600, height = 600, res = 300)
par(mfrow = c(1, 2))
for (gender in unique(df$Gender)) {
  gender_df <- df[df$Gender == gender, ]
  plot(gender_df$Age, gender_df$EstimatedSalary, 
       col = ifelse(gender_df$Purchased == 1, 
                    adjustcolor("red", alpha.f = 0.6), 
                    adjustcolor("blue", alpha.f = 0.6)),
       pch = 19,
       xlab = "Age", ylab = "Estimated Salary",
       main = paste(gender, ": Age vs Estimated Salary"))
  legend("topright", legend = c("No Purchase", "Purchase"), 
         col = c("blue", "red"), pch = 19)
}
dev.off()

cat("\n", paste(rep("=", 50), collapse=""), "\n")
cat("ANALYSIS COMPLETED SUCCESSFULLY!\n")
cat(paste(rep("=", 50), collapse=""), "\n")

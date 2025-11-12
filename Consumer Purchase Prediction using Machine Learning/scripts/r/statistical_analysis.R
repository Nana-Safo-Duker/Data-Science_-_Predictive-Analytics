# Statistical Analysis Script
# Consumer Purchase Prediction Project

# Load necessary libraries
library(dplyr)
library(ggplot2)
library(car)
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
cat("STATISTICAL ANALYSIS\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

# 1. Descriptive Statistics
cat("1. DESCRIPTIVE STATISTICS\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

cat("Age Statistics:\n")
print(summary(df$Age))
cat("Skewness:", skewness(df$Age), "\n")
cat("Kurtosis:", kurtosis(df$Age), "\n\n")

cat("Estimated Salary Statistics:\n")
print(summary(df$EstimatedSalary))
cat("Skewness:", skewness(df$EstimatedSalary), "\n")
cat("Kurtosis:", kurtosis(df$EstimatedSalary), "\n\n")

# By Purchase Status
cat("Descriptive Statistics by Purchase Status:\n")
print(describeBy(df[, c("Age", "EstimatedSalary")], df$Purchased))

# 2. Normality Tests
cat("\n2. NORMALITY TESTS\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

shapiro_age <- shapiro.test(df$Age)
cat("Age - Shapiro-Wilk Test:\n")
cat("  W =", shapiro_age$statistic, ", p-value =", shapiro_age$p.value, "\n")
if (shapiro_age$p.value > 0.05) {
  cat("  Result: Data appears to be normally distributed (p > 0.05)\n")
} else {
  cat("  Result: Data does not appear to be normally distributed (p <= 0.05)\n")
}

shapiro_salary <- shapiro.test(df$EstimatedSalary)
cat("\nEstimated Salary - Shapiro-Wilk Test:\n")
cat("  W =", shapiro_salary$statistic, ", p-value =", shapiro_salary$p.value, "\n")
if (shapiro_salary$p.value > 0.05) {
  cat("  Result: Data appears to be normally distributed (p > 0.05)\n")
} else {
  cat("  Result: Data does not appear to be normally distributed (p <= 0.05)\n")
}

# 3. Hypothesis Testing
cat("\n3. HYPOTHESIS TESTING\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

# Age: Independent samples t-test
age_purchased <- df$Age[df$Purchased == 1]
age_not_purchased <- df$Age[df$Purchased == 0]

# Check normality first
shapiro_age1 <- shapiro.test(age_purchased)
shapiro_age2 <- shapiro.test(age_not_purchased)

if (shapiro_age1$p.value > 0.05 && shapiro_age2$p.value > 0.05) {
  # Use t-test
  t_test_age <- t.test(age_purchased, age_not_purchased)
  cat("Age Difference (t-test):\n")
  cat("  t =", t_test_age$statistic, ", p-value =", t_test_age$p.value, "\n")
} else {
  # Use Mann-Whitney U test
  wilcox_test_age <- wilcox.test(age_purchased, age_not_purchased)
  cat("Age Difference (Mann-Whitney U test):\n")
  cat("  W =", wilcox_test_age$statistic, ", p-value =", wilcox_test_age$p.value, "\n")
}

# Salary: Independent samples t-test
salary_purchased <- df$EstimatedSalary[df$Purchased == 1]
salary_not_purchased <- df$EstimatedSalary[df$Purchased == 0]

shapiro_sal1 <- shapiro.test(salary_purchased)
shapiro_sal2 <- shapiro.test(salary_not_purchased)

if (shapiro_sal1$p.value > 0.05 && shapiro_sal2$p.value > 0.05) {
  t_test_salary <- t.test(salary_purchased, salary_not_purchased)
  cat("\nSalary Difference (t-test):\n")
  cat("  t =", t_test_salary$statistic, ", p-value =", t_test_salary$p.value, "\n")
} else {
  wilcox_test_salary <- wilcox.test(salary_purchased, salary_not_purchased)
  cat("\nSalary Difference (Mann-Whitney U test):\n")
  cat("  W =", wilcox_test_salary$statistic, ", p-value =", wilcox_test_salary$p.value, "\n")
}

# 4. Chi-Square Test
cat("\n4. CHI-SQUARE TEST\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

contingency_table <- table(df$Gender, df$Purchased)
cat("Contingency Table: Gender vs Purchased\n")
print(contingency_table)

chi2_test <- chisq.test(contingency_table)
cat("\nChi-square Test:\n")
cat("  Chi-square =", chi2_test$statistic, ", p-value =", chi2_test$p.value, "\n")
cat("  Degrees of freedom =", chi2_test$parameter, "\n")

# 5. Correlation Analysis
cat("\n5. CORRELATION ANALYSIS\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

numeric_df <- df[, c("Age", "EstimatedSalary", "Purchased")]
pearson_corr <- cor(numeric_df, method = "pearson")
cat("Pearson Correlation:\n")
print(pearson_corr)

spearman_corr <- cor(numeric_df, method = "spearman")
cat("\nSpearman Correlation:\n")
print(spearman_corr)

# Correlation tests
cor_test_age <- cor.test(df$Age, df$Purchased)
cor_test_salary <- cor.test(df$EstimatedSalary, df$Purchased)

cat("\nCorrelation Tests:\n")
cat("Age vs Purchased: r =", cor_test_age$estimate, ", p =", cor_test_age$p.value, "\n")
cat("Salary vs Purchased: r =", cor_test_salary$estimate, ", p =", cor_test_salary$p.value, "\n")

# 6. ANOVA
cat("\n6. ANOVA\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

# One-way ANOVA: Age by Purchase status
anova_age <- aov(Age ~ as.factor(Purchased), data = df)
cat("ANOVA: Age by Purchase Status\n")
print(summary(anova_age))

# One-way ANOVA: Salary by Purchase status
anova_salary <- aov(EstimatedSalary ~ as.factor(Purchased), data = df)
cat("\nANOVA: Estimated Salary by Purchase Status\n")
print(summary(anova_salary))

cat("\n", paste(rep("=", 50), collapse=""), "\n")
cat("STATISTICAL ANALYSIS COMPLETED SUCCESSFULLY!\n")
cat(paste(rep("=", 50), collapse=""), "\n")

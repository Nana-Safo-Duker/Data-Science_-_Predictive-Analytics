# Statistical Analysis Script
# Performs descriptive, inferential, and exploratory statistical analysis

# Load necessary libraries
library(dplyr)
library(psych)
library(car)

# Suppress warnings
options(warn = -1)

# Load the dataset
data_path <- file.path("..", "..", "data", "FuelConsumption.csv")
df <- read.csv(data_path, stringsAsFactors = FALSE)

# Clean column names
colnames(df) <- trimws(colnames(df))

cat("==================================================\n")
cat("STATISTICAL ANALYSIS\n")
cat("==================================================\n\n")

# Descriptive Statistics
cat("==================================================\n")
cat("DESCRIPTIVE STATISTICS\n")
cat("==================================================\n\n")

numerical_cols <- c("ENGINE.SIZE", "CYLINDERS", "FUEL.CONSUMPTION", "COEMISSIONS")

# Calculate descriptive statistics
descriptive_stats <- df %>%
  select(all_of(numerical_cols)) %>%
  describe() %>%
  select(mean, median, sd, var, min, max, skew, kurtosis)

cat("Descriptive Statistics:\n")
print(round(descriptive_stats, 3))

# Quartiles and IQR
cat("\nQuartiles and IQR:\n")
for(col in numerical_cols) {
  Q1 <- quantile(df[[col]], 0.25, na.rm = TRUE)
  Q2 <- quantile(df[[col]], 0.50, na.rm = TRUE)
  Q3 <- quantile(df[[col]], 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  cat("\n", col, ":\n")
  cat("  Q1:", round(Q1, 2), ", Q2:", round(Q2, 2), ", Q3:", round(Q3, 2), "\n")
  cat("  IQR:", round(IQR, 2), "\n")
}

# Inferential Statistics
cat("\n==================================================\n")
cat("INFERENTIAL STATISTICS\n")
cat("==================================================\n\n")

# Normality Tests
cat("Normality Tests:\n")
for(col in numerical_cols) {
  data <- df[[col]][!is.na(df[[col]])]
  if(length(data) < 5000) {
    test_result <- shapiro.test(data)
    test_name <- "Shapiro-Wilk"
    stat <- test_result$statistic
    p_value <- test_result$p.value
  } else {
    test_result <- ks.test(data, "pnorm", mean(data), sd(data))
    test_name <- "Kolmogorov-Smirnov"
    stat <- test_result$statistic
    p_value <- test_result$p.value
  }
  
  cat("\n", col, " (", test_name, "):\n")
  cat("  Statistic:", round(stat, 4), ", p-value:", round(p_value, 4), "\n")
  cat("  Normal:", ifelse(p_value > 0.05, "Yes", "No"), "\n")
}

# T-tests: Compare fuel consumption by fuel type
cat("\nT-test: Fuel Consumption by Fuel Type\n")
fuel_types <- unique(df$FUEL)
if(length(fuel_types) >= 2) {
  group1 <- df[df$FUEL == fuel_types[1], "FUEL.CONSUMPTION"]
  group2 <- df[df$FUEL == fuel_types[2], "FUEL.CONSUMPTION"]
  t_test <- t.test(group1, group2)
  cat("  T-statistic:", round(t_test$statistic, 4), "\n")
  cat("  P-value:", round(t_test$p.value, 4), "\n")
  cat("  Significant difference:", ifelse(t_test$p.value < 0.05, "Yes", "No"), "\n")
}

# ANOVA: Compare fuel consumption across vehicle classes
cat("\nANOVA: Fuel Consumption by Vehicle Class (Top 5)\n")
top_classes <- df %>%
  count(VEHICLE.CLASS, sort = TRUE) %>%
  head(5) %>%
  pull(VEHICLE.CLASS)

groups <- lapply(top_classes, function(x) {
  df[df$VEHICLE.CLASS == x, "FUEL.CONSUMPTION"]
})

anova_result <- aov(FUEL.CONSUMPTION ~ VEHICLE.CLASS, 
                    data = df[df$VEHICLE.CLASS %in% top_classes, ])
anova_summary <- summary(anova_result)
cat("  F-statistic:", round(anova_summary[[1]][["F value"]][1], 4), "\n")
cat("  P-value:", round(anova_summary[[1]][["Pr(>F)"]][1], 4), "\n")
cat("  Significant difference:", 
    ifelse(anova_summary[[1]][["Pr(>F)"]][1] < 0.05, "Yes", "No"), "\n")

# Exploratory Statistical Analysis
cat("\n==================================================\n")
cat("EXPLORATORY STATISTICAL ANALYSIS\n")
cat("==================================================\n\n")

# Confidence Intervals
cat("95% Confidence Intervals for Mean:\n")
for(col in numerical_cols) {
  data <- df[[col]][!is.na(df[[col]])]
  mean_val <- mean(data)
  n <- length(data)
  se <- sd(data) / sqrt(n)
  ci_lower <- mean_val - qt(0.975, n-1) * se
  ci_upper <- mean_val + qt(0.975, n-1) * se
  
  cat("\n", col, ":\n")
  cat("  Mean:", round(mean_val, 2), "\n")
  cat("  95% CI: [", round(ci_lower, 2), ", ", round(ci_upper, 2), "]\n")
}

# Correlation Analysis with p-values
cat("\nCorrelation Analysis with Significance:\n")
target <- "COEMISSIONS"
for(col in c("ENGINE.SIZE", "CYLINDERS", "FUEL.CONSUMPTION")) {
  data_subset <- df[, c(col, target)]
  data_subset <- data_subset[complete.cases(data_subset), ]
  
  pearson_test <- cor.test(data_subset[[col]], data_subset[[target]], 
                           method = "pearson")
  spearman_test <- cor.test(data_subset[[col]], data_subset[[target]], 
                            method = "spearman")
  
  cat("\n", col, " vs ", target, ":\n")
  cat("  Pearson: r =", round(pearson_test$estimate, 4), 
      ", p =", round(pearson_test$p.value, 4), "\n")
  cat("  Spearman: ρ =", round(spearman_test$estimate, 4), 
      ", p =", round(spearman_test$p.value, 4), "\n")
  cat("  Significant:", ifelse(pearson_test$p.value < 0.05, "Yes", "No"), "\n")
}

cat("\n==================================================\n")
cat("STATISTICAL ANALYSIS COMPLETE!\n")
cat("==================================================\n")

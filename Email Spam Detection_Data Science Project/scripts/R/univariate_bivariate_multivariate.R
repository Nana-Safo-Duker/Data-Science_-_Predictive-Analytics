# Univariate, Bivariate, and Multivariate Analysis
# Email Spam Detection Dataset - R Script

# Load required libraries
library(tidyverse)
library(ggplot2)
library(corrplot)
library(GGally)

# Load data
tryCatch({
  df <- read.csv("../../data/emails_spam_processed.csv", stringsAsFactors = FALSE)
}, error = function(e) {
  df <- read.csv("../../data/emails_spam_clean.csv", stringsAsFactors = FALSE)
  df$text_length <- nchar(df$text)
  df$word_count <- str_count(df$text, "\\S+")
})

numeric_cols <- c("text_length", "word_count", "sentence_count", "avg_word_length")
numeric_cols <- numeric_cols[numeric_cols %in% names(df)]

# Helper function for string concatenation
`%+%` <- function(a, b) paste0(a, b)

# Univariate Analysis
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("UNIVARIATE ANALYSIS\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

for (col in numeric_cols) {
  cat("\n1. Analysis of", col, ":\n")
  cat("   Mean:", mean(df[[col]], na.rm = TRUE), "\n")
  cat("   Median:", median(df[[col]], na.rm = TRUE), "\n")
  cat("   Std Dev:", sd(df[[col]], na.rm = TRUE), "\n")
  cat("   Min:", min(df[[col]], na.rm = TRUE), "\n")
  cat("   Max:", max(df[[col]], na.rm = TRUE), "\n")
  cat("   Q1:", quantile(df[[col]], 0.25, na.rm = TRUE), "\n")
  cat("   Q2 (Median):", quantile(df[[col]], 0.50, na.rm = TRUE), "\n")
  cat("   Q3:", quantile(df[[col]], 0.75, na.rm = TRUE), "\n")
  cat("   IQR:", IQR(df[[col]], na.rm = TRUE), "\n")
}

# Univariate visualizations
png("../../output/figures/univariate_analysis_R.png", width = 1600, height = 1200, res = 300)
par(mfrow = c(2, 2))

for (i in 1:min(4, length(numeric_cols))) {
  col <- numeric_cols[i]
  hist(df[[col]], breaks = 50, main = paste("Distribution of", col),
       xlab = col, ylab = "Frequency", col = "steelblue", border = "black")
  abline(v = mean(df[[col]], na.rm = TRUE), col = "red", lty = 2, lwd = 2)
  abline(v = median(df[[col]], na.rm = TRUE), col = "green", lty = 2, lwd = 2)
  legend("topright", legend = c("Mean", "Median"), 
         col = c("red", "green"), lty = 2, lwd = 2)
}

dev.off()

# Box plots
png("../../output/figures/univariate_boxplots_R.png", width = 1600, height = 1200, res = 300)
par(mfrow = c(2, 2))

for (i in 1:min(4, length(numeric_cols))) {
  col <- numeric_cols[i]
  boxplot(df[[col]], main = paste("Box Plot of", col), ylab = col, col = "steelblue")
}

dev.off()

# Bivariate Analysis
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("BIVARIATE ANALYSIS\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

cat("\n1. Relationship between Features and Target Variable (Spam):\n")
for (col in numeric_cols) {
  spam_mean <- mean(df[[col]][df$spam == 1], na.rm = TRUE)
  ham_mean <- mean(df[[col]][df$spam == 0], na.rm = TRUE)
  difference <- spam_mean - ham_mean
  percent_diff <- (difference / ham_mean) * 100
  
  cat("\n", col, ":\n", sep = "")
  cat("   Spam Mean:", spam_mean, "\n")
  cat("   Ham Mean:", ham_mean, "\n")
  cat("   Difference:", difference, "(", percent_diff, "%)\n", sep = "")
  
  # T-test
  test_result <- t.test(df[[col]][df$spam == 1], df[[col]][df$spam == 0])
  cat("   T-test p-value:", test_result$p.value, "\n")
  cat("   Significant:", ifelse(test_result$p.value < 0.05, "Yes", "No"), "\n")
}

# Scatter plots
png("../../output/figures/bivariate_scatter_R.png", width = 1600, height = 1200, res = 300)
par(mfrow = c(2, 2))

for (i in 1:min(4, length(numeric_cols))) {
  col <- numeric_cols[i]
  plot(df[[col]][df$spam == 0], df$spam[df$spam == 0], 
       col = rgb(0.5, 0.7, 1, 0.5), pch = 19, cex = 0.5,
       main = paste(col, "vs Spam"), xlab = col, ylab = "Spam (0=Ham, 1=Spam)")
  points(df[[col]][df$spam == 1], df$spam[df$spam == 1], 
         col = rgb(1, 0.5, 0.5, 0.5), pch = 19, cex = 0.5)
  legend("topright", legend = c("Ham", "Spam"), 
         col = c(rgb(0.5, 0.7, 1, 0.5), rgb(1, 0.5, 0.5, 0.5)), pch = 19)
}

dev.off()

# Violin plots
png("../../output/figures/bivariate_violin_R.png", width = 1600, height = 1200, res = 300)
par(mfrow = c(2, 2))

for (i in 1:min(4, length(numeric_cols))) {
  col <- numeric_cols[i]
  boxplot(df[[col]] ~ df$spam, main = paste("Distribution of", col, "by Spam/Ham"),
          xlab = "Spam (0=Ham, 1=Spam)", ylab = col,
          col = c("skyblue", "salmon"), names = c("Ham", "Spam"))
}

dev.off()

# Correlation with target
cat("\n2. Correlation with Target Variable:\n")
for (col in numeric_cols) {
  correlation <- cor(df[[col]], df$spam, use = "complete.obs")
  cat("  ", col, "- Spam correlation:", correlation, "\n")
}

# Multivariate Analysis
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("MULTIVARIATE ANALYSIS\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

# Correlation matrix
cat("\n1. Correlation Matrix:\n")
correlation_matrix <- cor(df[c(numeric_cols, "spam")], use = "complete.obs")
print(correlation_matrix)

# Visualize correlation matrix
png("../../output/figures/multivariate_correlation_R.png", width = 1000, height = 1000, res = 300)
corrplot(correlation_matrix, method = "color", type = "upper", 
         order = "hclust", tl.cex = 0.8, tl.col = "black", addCoef.col = "black")
dev.off()

# Pair plots (sample if too large)
cat("\n2. Pairwise Relationships:\n")
sample_df <- df[sample(nrow(df), min(1000, nrow(df))), ]
pair_cols <- c(numeric_cols[1:min(4, length(numeric_cols))], "spam")

# Create pair plot using ggplot2
png("../../output/figures/multivariate_pairplot_R.png", width = 2000, height = 2000, res = 300)
if (require(GGally)) {
  sample_df$spam <- as.factor(sample_df$spam)
  ggpairs(sample_df[pair_cols], aes(color = spam, alpha = 0.5))
} else {
  pairs(sample_df[pair_cols], col = sample_df$spam + 1, pch = 19, cex = 0.5)
}
dev.off()

# PCA
cat("\n3. Principal Component Analysis:\n")
X <- df[numeric_cols]
X <- X[complete.cases(X), ]

# Standardize
X_scaled <- scale(X)

# Apply PCA
pca_result <- prcomp(X_scaled)

# Visualize PCA
png("../../output/figures/multivariate_pca_R.png", width = 1200, height = 800, res = 300)
plot(pca_result$x[, 1], pca_result$x[, 2], 
     col = df$spam[complete.cases(df[numeric_cols])] + 1,
     main = "PCA - Multivariate Analysis",
     xlab = paste("PC1 (", round(summary(pca_result)$importance[2, 1] * 100, 2), "%)", sep = ""),
     ylab = paste("PC2 (", round(summary(pca_result)$importance[2, 2] * 100, 2), "%)", sep = ""),
     pch = 19, cex = 0.5)
legend("topright", legend = c("Ham", "Spam"), col = c(1, 2), pch = 19)
dev.off()

cat("   Explained variance by PC1:", round(summary(pca_result)$importance[2, 1] * 100, 2), "%\n")
cat("   Explained variance by PC2:", round(summary(pca_result)$importance[2, 2] * 100, 2), "%\n")
cat("   Total explained variance:", round(sum(summary(pca_result)$importance[2, 1:2]) * 100, 2), "%\n")

# Feature interactions
cat("\n4. Feature Interactions:\n")
if ("text_length" %in% names(df) && "word_count" %in% names(df)) {
  df$text_length_word_count_interaction <- df$text_length * df$word_count
  interaction_corr <- cor(df$text_length_word_count_interaction, df$spam, use = "complete.obs")
  cat("   Text Length × Word Count interaction - Spam correlation:", interaction_corr, "\n")
}

cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("Analysis Complete!\n")
cat(paste(rep("=", 60), collapse = ""), "\n")


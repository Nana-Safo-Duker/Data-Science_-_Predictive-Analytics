# Descriptive, Inferential, and Exploratory Statistical Analysis
# Email Spam Detection Dataset - R Script

# Load required libraries
library(tidyverse)
library(ggplot2)
library(corrplot)
library(psych)
library(Hmisc)

# Load data
tryCatch({
  df <- read.csv("../../data/emails_spam_processed.csv", stringsAsFactors = FALSE)
}, error = function(e) {
  df <- read.csv("../../data/emails_spam_clean.csv", stringsAsFactors = FALSE)
  df$text_length <- nchar(df$text)
  df$word_count <- str_count(df$text, "\\S+")
})

# Helper function for string concatenation
`%+%` <- function(a, b) paste0(a, b)

# Descriptive Statistics
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("DESCRIPTIVE STATISTICS\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

numeric_cols <- c("text_length", "word_count", "sentence_count", "avg_word_length")
numeric_cols <- numeric_cols[numeric_cols %in% names(df)]

cat("\n1. Overall Descriptive Statistics:\n")
print(describe(df[numeric_cols]))

cat("\n2. Descriptive Statistics by Class (Spam vs Ham):\n")
for (col in numeric_cols) {
  cat("\n", col, ":\n", sep = "")
  print(df %>% 
    group_by(spam) %>% 
    summarise(
      mean = mean(.data[[col]], na.rm = TRUE),
      median = median(.data[[col]], na.rm = TRUE),
      sd = sd(.data[[col]], na.rm = TRUE),
      min = min(.data[[col]], na.rm = TRUE),
      max = max(.data[[col]], na.rm = TRUE),
      q25 = quantile(.data[[col]], 0.25, na.rm = TRUE),
      q75 = quantile(.data[[col]], 0.75, na.rm = TRUE),
      .groups = 'drop'
    ))
}

# Inferential Statistics
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("INFERENTIAL STATISTICS\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

cat("\n1. Independent Samples T-Test:\n")
cat("Testing if there's a significant difference in means between Spam and Ham\n")
cat("\nHypotheses:\n")
cat("H0: μ_spam = μ_ham (no difference in means)\n")
cat("H1: μ_spam ≠ μ_ham (significant difference in means)\n")
cat("\nSignificance level: α = 0.05\n")

for (col in numeric_cols) {
  spam_data <- df[[col]][df$spam == 1]
  ham_data <- df[[col]][df$spam == 0]
  
  # Remove NA values
  spam_data <- spam_data[!is.na(spam_data)]
  ham_data <- ham_data[!is.na(ham_data)]
  
  # Perform t-test
  test_result <- t.test(spam_data, ham_data)
  
  cat("\n", col, ":\n", sep = "")
  cat("  T-statistic:", test_result$statistic, "\n")
  cat("  P-value:", test_result$p.value, "\n")
  cat("  Mean (Spam):", mean(spam_data), "\n")
  cat("  Mean (Ham):", mean(ham_data), "\n")
  
  if (test_result$p.value < 0.05) {
    cat("  Result: Reject H0 - Significant difference (p < 0.05)\n")
  } else {
    cat("  Result: Fail to reject H0 - No significant difference (p >= 0.05)\n")
  }
}

# Mann-Whitney U test
cat("\n2. Mann-Whitney U Test (Non-parametric):\n")
for (col in numeric_cols) {
  spam_data <- df[[col]][df$spam == 1]
  ham_data <- df[[col]][df$spam == 0]
  
  spam_data <- spam_data[!is.na(spam_data)]
  ham_data <- ham_data[!is.na(ham_data)]
  
  test_result <- wilcox.test(spam_data, ham_data)
  
  cat("\n", col, ":\n", sep = "")
  cat("  W-statistic:", test_result$statistic, "\n")
  cat("  P-value:", test_result$p.value, "\n")
  
  if (test_result$p.value < 0.05) {
    cat("  Result: Reject H0 - Significant difference (p < 0.05)\n")
  } else {
    cat("  Result: Fail to reject H0 - No significant difference (p >= 0.05)\n")
  }
}

# Chi-square test
cat("\n3. Chi-Square Test for Independence:\n")
df$text_length_category <- cut(df$text_length, breaks = 3, labels = c("Short", "Medium", "Long"))
contingency_table <- table(df$spam, df$text_length_category)

cat("\nContingency Table:\n")
print(contingency_table)

test_result <- chisq.test(contingency_table)
cat("\nChi-square statistic:", test_result$statistic, "\n")
cat("P-value:", test_result$p.value, "\n")
cat("Degrees of freedom:", test_result$parameter, "\n")

if (test_result$p.value < 0.05) {
  cat("Result: Reject H0 - Text length category and spam are associated (p < 0.05)\n")
} else {
  cat("Result: Fail to reject H0 - No association (p >= 0.05)\n")
}

# Exploratory Statistical Analysis
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("EXPLORATORY STATISTICAL ANALYSIS\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

# Correlation analysis
cat("\n1. Correlation Analysis:\n")
correlation_matrix <- cor(df[c(numeric_cols, "spam")], use = "complete.obs")
print(correlation_matrix)

# Visualize correlation matrix
png("../../output/figures/correlation_matrix_R.png", width = 1000, height = 1000, res = 300)
corrplot(correlation_matrix, method = "color", type = "upper", 
         order = "hclust", tl.cex = 0.8, tl.col = "black")
dev.off()

# Distribution analysis
cat("\n2. Distribution Analysis:\n")
png("../../output/figures/distributions_R.png", width = 1600, height = 1200, res = 300)
par(mfrow = c(2, 2))

for (col in numeric_cols[1:min(4, length(numeric_cols))]) {
  hist(df[[col]][df$spam == 0], breaks = 30, col = rgb(0.5, 0.7, 1, 0.6),
       main = paste("Distribution of", col), xlab = col, ylab = "Frequency")
  hist(df[[col]][df$spam == 1], breaks = 30, col = rgb(1, 0.5, 0.5, 0.6), add = TRUE)
  legend("topright", legend = c("Ham", "Spam"), 
         fill = c(rgb(0.5, 0.7, 1, 0.6), rgb(1, 0.5, 0.5, 0.6)))
}

dev.off()

# Normality tests
cat("\n3. Normality Tests (Shapiro-Wilk):\n")
for (col in numeric_cols) {
  spam_data <- df[[col]][df$spam == 1]
  ham_data <- df[[col]][df$spam == 0]
  
  spam_data <- spam_data[!is.na(spam_data)]
  ham_data <- ham_data[!is.na(ham_data)]
  
  # Sample if too large
  if (length(spam_data) > 5000) spam_data <- sample(spam_data, 5000)
  if (length(ham_data) > 5000) ham_data <- sample(ham_data, 5000)
  
  test_spam <- shapiro.test(spam_data)
  test_ham <- shapiro.test(ham_data)
  
  cat("\n", col, ":\n", sep = "")
  cat("  Spam - Statistic:", test_spam$statistic, ", P-value:", test_spam$p.value, "\n")
  cat("    ", ifelse(test_spam$p.value > 0.05, "Normal", "Not Normal"), " distribution\n", sep = "")
  cat("  Ham - Statistic:", test_ham$statistic, ", P-value:", test_ham$p.value, "\n")
  cat("    ", ifelse(test_ham$p.value > 0.05, "Normal", "Not Normal"), " distribution\n", sep = "")
}

# Confidence intervals
cat("\n4. Confidence Intervals (95%):\n")
for (col in numeric_cols) {
  spam_data <- df[[col]][df$spam == 1]
  ham_data <- df[[col]][df$spam == 0]
  
  spam_data <- spam_data[!is.na(spam_data)]
  ham_data <- ham_data[!is.na(ham_data)]
  
  spam_ci <- t.test(spam_data)$conf.int
  ham_ci <- t.test(ham_data)$conf.int
  
  cat("\n", col, ":\n", sep = "")
  cat("  Spam Mean:", mean(spam_data), ", 95% CI: [", spam_ci[1], ", ", spam_ci[2], "]\n", sep = "")
  cat("  Ham Mean:", mean(ham_data), ", 95% CI: [", ham_ci[1], ", ", ham_ci[2], "]\n", sep = "")
}

cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("Statistical Analysis Complete!\n")
cat(paste(rep("=", 60), collapse = ""), "\n")


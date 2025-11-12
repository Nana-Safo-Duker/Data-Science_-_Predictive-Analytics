# Statistical Analysis for Employee Dataset
# Comprehensive descriptive, inferential, and exploratory statistical analysis

# Load required libraries
library(tidyverse)
library(ggplot2)
library(dplyr)
library(corrplot)
library(psych)
library(car)

# Set working directory to project root
# This script should be run from the project root directory
# Or navigate to project root if running from scripts/r directory
if (basename(getwd()) == "r" && basename(dirname(getwd())) == "scripts") {
  setwd(dirname(dirname(getwd())))
} else if (basename(getwd()) == "scripts") {
  setwd(dirname(getwd()))
}
cat("Working directory:", getwd(), "\n")
# Verify we're in the right place
if (!file.exists("data/processed/employees_cleaned.csv")) {
  stop("Please run EDA script first to create cleaned dataset, or ensure data/processed/employees_cleaned.csv exists")
}

# Create results directories
dir.create("results/plots", recursive = TRUE, showWarnings = FALSE)
dir.create("results/tables", recursive = TRUE, showWarnings = FALSE)

cat("===============================================================================\n")
cat("STATISTICAL ANALYSIS - EMPLOYEE DATASET\n")
cat("===============================================================================\n")

# Load cleaned dataset
df <- read.csv("data/processed/employees_cleaned.csv", stringsAsFactors = FALSE)

# =============================================================================
# 1. DESCRIPTIVE STATISTICS
# =============================================================================
cat("\n1. DESCRIPTIVE STATISTICS\n")
cat("===============================================================================\n")

numerical_cols <- c("Salary", "Bonus_pct", "Years_of_Service")
  
  # Basic descriptive statistics
  cat("\nBasic Descriptive Statistics:\n")
desc_stats <- describe(df[numerical_cols])
  print(desc_stats)
write.csv(desc_stats, "results/tables/descriptive_statistics.csv", row.names = FALSE)
  
  # Additional descriptive statistics
  cat("\nAdditional Descriptive Statistics:\n")
additional_stats <- data.frame(
  Variable = numerical_cols,
  Skewness = sapply(df[numerical_cols], function(x) psych::skew(x, na.rm = TRUE)),
  Kurtosis = sapply(df[numerical_cols], function(x) psych::kurtosi(x, na.rm = TRUE)),
  Variance = sapply(df[numerical_cols], function(x) var(x, na.rm = TRUE)),
  Coefficient_of_Variation = sapply(df[numerical_cols], function(x) sd(x, na.rm = TRUE) / mean(x, na.rm = TRUE) * 100)
)
print(additional_stats)
write.csv(additional_stats, "results/tables/additional_descriptive_statistics.csv", row.names = FALSE)

# =============================================================================
# 2. NORMALITY TESTS
# =============================================================================
cat("\n2. NORMALITY TESTS\n")
cat("===============================================================================\n")

normality_results <- data.frame()

for (col in numerical_cols) {
    data <- df[[col]][!is.na(df[[col]])]
  
  # Shapiro-Wilk test (for smaller samples)
  if (length(data) <= 5000) {
    shapiro_test <- shapiro.test(data)
    normality_results <- rbind(normality_results, data.frame(
      Variable = col,
      Test = "Shapiro-Wilk",
      Statistic = shapiro_test$statistic,
      P_value = shapiro_test$p.value,
      Normal = ifelse(shapiro_test$p.value > 0.05, "Yes", "No")
    ))
  }
  
  # Anderson-Darling normality test
  if (requireNamespace("nortest", quietly = TRUE)) {
    ad_test <- nortest::ad.test(data)
  } else {
    # Fallback to Kolmogorov-Smirnov test if nortest is not available
    ad_test <- list(statistic = NA, p.value = ks.test(data, "pnorm", mean(data), sd(data))$p.value)
  }
  normality_results <- rbind(normality_results, data.frame(
    Variable = col,
    Test = "Anderson-Darling",
    Statistic = ad_test$statistic,
    P_value = ad_test$p.value,
    Normal = ifelse(ad_test$p.value > 0.05, "Yes", "No")
  ))
}

cat("\nNormality Test Results:\n")
print(normality_results)
write.csv(normality_results, "results/tables/normality_tests.csv", row.names = FALSE)

# Q-Q plots for normality
png("results/plots/qq_plots_normality.png", width = 2400, height = 800, res = 300)
par(mfrow = c(1, 3))
for (col in numerical_cols) {
  qqnorm(df[[col]], main = paste("Q-Q Plot:", col))
  qqline(df[[col]], col = "red")
}
dev.off()

# =============================================================================
# 3. INFERENTIAL STATISTICS - T-TESTS
# =============================================================================
cat("\n3. INFERENTIAL STATISTICS - T-TESTS\n")
cat("===============================================================================\n")

# T-test: Salary by Gender
cat("\n3.1. T-test: Salary by Gender\n")
  male_salary <- df$Salary[df$Gender == "Male" & !is.na(df$Salary)]
  female_salary <- df$Salary[df$Gender == "Female" & !is.na(df$Salary)]
  
t_test_gender <- t.test(male_salary, female_salary)
cat("T-statistic:", t_test_gender$statistic, "\n")
cat("P-value:", t_test_gender$p.value, "\n")
cat("Mean Male Salary: $", round(mean(male_salary), 2), "\n", sep = "")
cat("Mean Female Salary: $", round(mean(female_salary), 2), "\n", sep = "")
cat("Significant difference:", ifelse(t_test_gender$p.value < 0.05, "Yes", "No"), "\n")

# T-test: Salary by Senior Management
cat("\n3.2. T-test: Salary by Senior Management\n")
sm_salary <- df$Salary[df$Senior_Management == TRUE & !is.na(df$Salary)]
non_sm_salary <- df$Salary[df$Senior_Management == FALSE & !is.na(df$Salary)]

t_test_sm <- t.test(sm_salary, non_sm_salary)
cat("T-statistic:", t_test_sm$statistic, "\n")
cat("P-value:", t_test_sm$p.value, "\n")
cat("Mean Senior Management Salary: $", round(mean(sm_salary), 2), "\n", sep = "")
cat("Mean Non-Senior Management Salary: $", round(mean(non_sm_salary), 2), "\n", sep = "")
cat("Significant difference:", ifelse(t_test_sm$p.value < 0.05, "Yes", "No"), "\n")

# Save t-test results
ttest_results <- data.frame(
  Test = c("Salary by Gender", "Salary by Senior Management"),
  T_statistic = c(t_test_gender$statistic, t_test_sm$statistic),
  P_value = c(t_test_gender$p.value, t_test_sm$p.value),
  Significant = c(ifelse(t_test_gender$p.value < 0.05, "Yes", "No"),
                  ifelse(t_test_sm$p.value < 0.05, "Yes", "No"))
)
write.csv(ttest_results, "results/tables/ttest_results.csv", row.names = FALSE)

# =============================================================================
# 4. NON-PARAMETRIC TESTS - MANN-WHITNEY U TEST
# =============================================================================
cat("\n4. NON-PARAMETRIC TESTS - MANN-WHITNEY U TEST\n")
cat("===============================================================================\n")

# Mann-Whitney U test: Salary by Gender
cat("\n4.1. Mann-Whitney U Test: Salary by Gender\n")
u_test_gender <- wilcox.test(male_salary, female_salary)
cat("U-statistic:", u_test_gender$statistic, "\n")
cat("P-value:", u_test_gender$p.value, "\n")
cat("Significant difference:", ifelse(u_test_gender$p.value < 0.05, "Yes", "No"), "\n")

# Mann-Whitney U test: Salary by Senior Management
cat("\n4.2. Mann-Whitney U Test: Salary by Senior Management\n")
u_test_sm <- wilcox.test(sm_salary, non_sm_salary)
cat("U-statistic:", u_test_sm$statistic, "\n")
cat("P-value:", u_test_sm$p.value, "\n")
cat("Significant difference:", ifelse(u_test_sm$p.value < 0.05, "Yes", "No"), "\n")

# =============================================================================
# 5. CHI-SQUARE TESTS
# =============================================================================
cat("\n5. CHI-SQUARE TESTS\n")
cat("===============================================================================\n")

# Chi-square test: Gender and Senior Management
cat("\n5.1. Chi-square Test: Gender and Senior Management\n")
  contingency_table <- table(df$Gender, df$Senior_Management)
  cat("\nContingency Table:\n")
  print(contingency_table)
  
chi2_test <- chisq.test(contingency_table)
cat("\nChi-square statistic:", chi2_test$statistic, "\n")
cat("P-value:", chi2_test$p.value, "\n")
cat("Degrees of freedom:", chi2_test$parameter, "\n")
cat("Significant association:", ifelse(chi2_test$p.value < 0.05, "Yes", "No"), "\n")

# Chi-square test: Gender and Team (top teams)
cat("\n5.2. Chi-square Test: Gender and Team (Top 5 Teams)\n")
top_teams <- names(head(sort(table(df$Team), decreasing = TRUE), 5))
df_top_teams <- df[df$Team %in% top_teams, ]
contingency_table_team <- table(df_top_teams$Gender, df_top_teams$Team)
cat("\nContingency Table:\n")
print(contingency_table_team)

chi2_test_team <- chisq.test(contingency_table_team)
cat("\nChi-square statistic:", chi2_test_team$statistic, "\n")
cat("P-value:", chi2_test_team$p.value, "\n")
cat("Degrees of freedom:", chi2_test_team$parameter, "\n")
cat("Significant association:", ifelse(chi2_test_team$p.value < 0.05, "Yes", "No"), "\n")

# Save chi-square results
chisquare_results <- data.frame(
  Test = c("Gender and Senior Management", "Gender and Team"),
  Chi_square = c(chi2_test$statistic, chi2_test_team$statistic),
  P_value = c(chi2_test$p.value, chi2_test_team$p.value),
  Significant = c(ifelse(chi2_test$p.value < 0.05, "Yes", "No"),
                  ifelse(chi2_test_team$p.value < 0.05, "Yes", "No"))
)
write.csv(chisquare_results, "results/tables/chisquare_results.csv", row.names = FALSE)

# =============================================================================
# 6. ANOVA - ANALYSIS OF VARIANCE
# =============================================================================
cat("\n6. ANOVA - ANALYSIS OF VARIANCE\n")
cat("===============================================================================\n")

# ANOVA: Salary across Teams (top 10 teams)
cat("\n6.1. ANOVA: Salary across Teams (Top 10 Teams)\n")
top_teams_anova <- names(head(sort(table(df$Team), decreasing = TRUE), 10))
df_anova <- df[df$Team %in% top_teams_anova & !is.na(df$Salary), ]

anova_model <- aov(Salary ~ Team, data = df_anova)
anova_summary <- summary(anova_model)
cat("F-statistic:", anova_summary[[1]]$`F value`[1], "\n")
cat("P-value:", anova_summary[[1]]$`Pr(>F)`[1], "\n")
cat("Significant difference:", ifelse(anova_summary[[1]]$`Pr(>F)`[1] < 0.05, "Yes", "No"), "\n")

# Post-hoc analysis: Pairwise comparisons
cat("\n6.2. Post-hoc Analysis: Pairwise T-tests (Bonferroni correction)\n")
pairwise_results <- pairwise.t.test(df_anova$Salary, df_anova$Team, p.adjust.method = "bonferroni")
cat("\nPairwise comparisons (Bonferroni corrected):\n")
print(pairwise_results$p.value)

# =============================================================================
# 7. CORRELATION ANALYSIS WITH SIGNIFICANCE TESTING
# =============================================================================
cat("\n7. CORRELATION ANALYSIS WITH SIGNIFICANCE TESTING\n")
cat("===============================================================================\n")

# Pearson correlation with significance
correlation_results <- data.frame()
for (i in 1:(length(numerical_cols)-1)) {
  for (j in (i+1):length(numerical_cols)) {
    col1 <- numerical_cols[i]
    col2 <- numerical_cols[j]
    
    data1 <- df[[col1]][!is.na(df[[col1]]) & !is.na(df[[col2]])]
    data2 <- df[[col2]][!is.na(df[[col1]]) & !is.na(df[[col2]])]
    
    if (length(data1) > 2) {
      cor_test <- cor.test(data1, data2)
      correlation_results <- rbind(correlation_results, data.frame(
        Variable1 = col1,
        Variable2 = col2,
        Correlation = cor_test$estimate,
        P_value = cor_test$p.value,
        Significant = ifelse(cor_test$p.value < 0.05, "Yes", "No")
      ))
    }
  }
}

cat("\nCorrelation Analysis Results:\n")
print(correlation_results)
write.csv(correlation_results, "results/tables/correlation_analysis.csv", row.names = FALSE)

# Visualize correlations
png("results/plots/correlation_matrix_statistical.png", width = 1600, height = 1600, res = 300)
corr_matrix <- cor(df[numerical_cols], use = "complete.obs")
corrplot(corr_matrix, method = "color", type = "upper", order = "hclust", 
         tl.cex = 0.8, tl.col = "black", tl.srt = 45, addCoef.col = "black")
  dev.off()
  
# =============================================================================
# 8. CONFIDENCE INTERVALS
# =============================================================================
cat("\n8. CONFIDENCE INTERVALS\n")
cat("===============================================================================\n")

confidence_intervals <- data.frame()
confidence_level <- 0.95

for (col in numerical_cols) {
    data <- df[[col]][!is.na(df[[col]])]
  n <- length(data)
    mean_val <- mean(data)
  std_err <- sd(data) / sqrt(n)
  t_critical <- qt((1 + confidence_level) / 2, df = n - 1)
  margin_error <- t_critical * std_err
  
  confidence_intervals <- rbind(confidence_intervals, data.frame(
    Variable = col,
    Mean = mean_val,
    Std_Error = std_err,
    Lower_CI_95 = mean_val - margin_error,
    Upper_CI_95 = mean_val + margin_error,
    Margin_of_Error = margin_error
  ))
}

cat("\n95% Confidence Intervals:\n")
print(confidence_intervals)
write.csv(confidence_intervals, "results/tables/confidence_intervals.csv", row.names = FALSE)

# Visualize confidence intervals
png("results/plots/confidence_intervals.png", width = 2400, height = 800, res = 300)
par(mfrow = c(1, 3))
for (i in 1:nrow(confidence_intervals)) {
  ci <- confidence_intervals[i, ]
  plot(1, ci$Mean, xlim = c(0.5, 1.5), ylim = c(ci$Lower_CI_95 - ci$Margin_of_Error, 
                                                   ci$Upper_CI_95 + ci$Margin_of_Error),
       main = paste("95% CI for", ci$Variable), xlab = "", ylab = ci$Variable,
       pch = 19, col = "steelblue", xaxt = "n")
  arrows(1, ci$Lower_CI_95, 1, ci$Upper_CI_95, angle = 90, code = 3, length = 0.1, col = "steelblue", lwd = 2)
  abline(h = ci$Mean, col = "red", lty = 2, lwd = 2)
  grid()
}
dev.off()

# =============================================================================
# 9. SUMMARY OF STATISTICAL TESTS
# =============================================================================
cat("\n9. SUMMARY OF STATISTICAL TESTS\n")
cat("===============================================================================\n")

summary <- data.frame(
  Test_Type = c("T-test", "T-test", "Mann-Whitney U", "Chi-square", "ANOVA"),
  Hypothesis = c("Salary differs by Gender", "Salary differs by Senior Management",
                 "Salary differs by Gender (non-parametric)", 
                 "Association between Gender and Senior Management",
                 "Salary differs across Teams"),
  Result = c(ifelse(t_test_gender$p.value < 0.05, "Significant", "Not Significant"),
             ifelse(t_test_sm$p.value < 0.05, "Significant", "Not Significant"),
             ifelse(u_test_gender$p.value < 0.05, "Significant", "Not Significant"),
             ifelse(chi2_test$p.value < 0.05, "Significant", "Not Significant"),
             ifelse(anova_summary[[1]]$`Pr(>F)`[1] < 0.05, "Significant", "Not Significant"))
)

cat("\nStatistical Test Summary:\n")
print(summary)
write.csv(summary, "results/tables/statistical_tests_summary.csv", row.names = FALSE)

cat("\n===============================================================================\n")
cat("STATISTICAL ANALYSIS COMPLETED SUCCESSFULLY!\n")
cat("===============================================================================\n")
cat("\nResults saved in:\n")
cat("- Tables: results/tables/\n")
cat("- Plots: results/plots/\n")

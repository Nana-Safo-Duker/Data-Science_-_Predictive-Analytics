# Univariate, Bivariate, and Multivariate Analysis for Employee Dataset
# Comprehensive analysis of individual variables, pairs of variables, and multiple variables

# Load required libraries
library(tidyverse)
library(ggplot2)
library(dplyr)
library(corrplot)
library(GGally)
library(plotly)

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
cat("UNIVARIATE, BIVARIATE, AND MULTIVARIATE ANALYSIS - EMPLOYEE DATASET\n")
cat("===============================================================================\n")

# Load cleaned dataset
df <- read.csv("data/processed/employees_cleaned.csv", stringsAsFactors = FALSE)

numerical_cols <- c("Salary", "Bonus_pct", "Years_of_Service")
categorical_cols <- c("Gender", "Senior_Management", "Team")

# =============================================================================
# 1. UNIVARIATE ANALYSIS
# =============================================================================
cat("\n1. UNIVARIATE ANALYSIS\n")
cat("===============================================================================\n")

# 1.1 Numerical Variables
cat("\n1.1. Numerical Variables Univariate Analysis\n")

for (col in numerical_cols) {
  cat("\n---", col, "---\n")
  data <- df[[col]][!is.na(df[[col]])]
  cat("Mean:", round(mean(data), 2), "\n")
  cat("Median:", round(median(data), 2), "\n")
  cat("Mode:", names(sort(table(data), decreasing = TRUE))[1], "\n")
  cat("Standard Deviation:", round(sd(data), 2), "\n")
  cat("Variance:", round(var(data), 2), "\n")
  cat("Skewness:", round(psych::skew(data), 2), "\n")
  cat("Kurtosis:", round(psych::kurtosi(data), 2), "\n")
  cat("Min:", round(min(data), 2), "\n")
  cat("Max:", round(max(data), 2), "\n")
  cat("Range:", round(max(data) - min(data), 2), "\n")
  cat("Q1:", round(quantile(data, 0.25), 2), "\n")
  cat("Q3:", round(quantile(data, 0.75), 2), "\n")
  cat("IQR:", round(quantile(data, 0.75) - quantile(data, 0.25), 2), "\n")
}

# Visualizations for numerical variables
png("results/plots/univariate_numerical.png", width = 2400, height = 1800, res = 300)
par(mfrow = c(3, 4))
for (col in numerical_cols) {
  data <- df[[col]][!is.na(df[[col]])]
  
  # Histogram
  hist(data, breaks = 50, main = paste("Histogram:", col), xlab = col, 
       col = "steelblue", border = "black")
  abline(v = mean(data), col = "red", lty = 2, lwd = 2)
  abline(v = median(data), col = "green", lty = 2, lwd = 2)
  legend("topright", legend = c(paste("Mean:", round(mean(data), 2)),
                                paste("Median:", round(median(data), 2))),
         col = c("red", "green"), lty = 2, cex = 0.8)
  
  # Box plot
  boxplot(data, main = paste("Box Plot:", col), ylab = col, col = "lightblue")
  
  # Q-Q plot
  qqnorm(data, main = paste("Q-Q Plot:", col))
  qqline(data, col = "red")
  
  # Violin plot (simulated with density)
  density_data <- density(data)
  plot(density_data, main = paste("Density Plot:", col), xlab = col, type = "l", lwd = 2)
  polygon(density_data, col = "steelblue", border = "black")
}
dev.off()

# 1.2 Categorical Variables
cat("\n1.2. Categorical Variables Univariate Analysis\n")

for (col in categorical_cols) {
  cat("\n---", col, "---\n")
  counts <- table(df[[col]], useNA = "always")
  percentages <- prop.table(counts) * 100
  cat("Counts:\n")
  print(counts)
  cat("\nPercentages:\n")
  print(round(percentages, 2))
  cat("Mode:", names(sort(counts, decreasing = TRUE))[1], "\n")
  cat("Number of unique values:", length(unique(df[[col]])), "\n")
}

# Visualizations for categorical variables
png("results/plots/univariate_categorical.png", width = 2400, height = 800, res = 300)
par(mfrow = c(1, 3))

# Gender distribution
gender_counts <- table(df$Gender)
barplot(gender_counts, main = "Gender Distribution", xlab = "Gender", ylab = "Count",
        col = c("skyblue", "pink", "lightgray"), border = "black")

# Senior Management distribution
sm_counts <- table(df$Senior_Management)
barplot(sm_counts, main = "Senior Management Distribution", xlab = "Senior Management",
        ylab = "Count", col = c("lightcoral", "lightgreen"), border = "black",
        names.arg = c("False", "True"))

# Team distribution (top 10)
team_counts <- head(sort(table(df$Team), decreasing = TRUE), 10)
barplot(team_counts, main = "Top 10 Teams by Employee Count", xlab = "Count",
        ylab = "Team", col = "steelblue", border = "black", horiz = TRUE, las = 1)
dev.off()

# =============================================================================
# 2. BIVARIATE ANALYSIS
# =============================================================================
cat("\n2. BIVARIATE ANALYSIS\n")
cat("===============================================================================\n")

# 2.1 Numerical vs Numerical
cat("\n2.1. Numerical vs Numerical Analysis\n")

# Correlation analysis
correlation_matrix <- cor(df[numerical_cols], use = "complete.obs")
cat("\nCorrelation Matrix:\n")
print(correlation_matrix)
write.csv(correlation_matrix, "results/tables/bivariate_correlation_matrix.csv")

# Scatter plots
png("results/plots/bivariate_numerical_numerical.png", width = 2400, height = 800, res = 300)
par(mfrow = c(1, 3))

# Salary vs Bonus %
plot(df$Salary, df$Bonus_pct, main = "Salary vs Bonus %", xlab = "Salary",
     ylab = "Bonus %", pch = 19, alpha = 0.5, col = "steelblue")
cor_sb <- cor(df$Salary, df$Bonus_pct, use = "complete.obs")
text(par("usr")[1] + 0.05 * diff(par("usr")[1:2]),
     par("usr")[4] - 0.05 * diff(par("usr")[3:4]),
     paste("Correlation:", round(cor_sb, 3)), adj = c(0, 1), cex = 1.2,
     bg = "wheat", box.col = "black")
grid()

# Salary vs Years of Service
plot(df$Years_of_Service, df$Salary, main = "Salary vs Years of Service",
     xlab = "Years of Service", ylab = "Salary", pch = 19, alpha = 0.5, col = "steelblue")
cor_ys <- cor(df$Years_of_Service, df$Salary, use = "complete.obs")
text(par("usr")[1] + 0.05 * diff(par("usr")[1:2]),
     par("usr")[4] - 0.05 * diff(par("usr")[3:4]),
     paste("Correlation:", round(cor_ys, 3)), adj = c(0, 1), cex = 1.2,
     bg = "wheat", box.col = "black")
grid()

# Bonus % vs Years of Service
plot(df$Years_of_Service, df$Bonus_pct, main = "Bonus % vs Years of Service",
     xlab = "Years of Service", ylab = "Bonus %", pch = 19, alpha = 0.5, col = "steelblue")
cor_by <- cor(df$Bonus_pct, df$Years_of_Service, use = "complete.obs")
text(par("usr")[1] + 0.05 * diff(par("usr")[1:2]),
     par("usr")[4] - 0.05 * diff(par("usr")[3:4]),
     paste("Correlation:", round(cor_by, 3)), adj = c(0, 1), cex = 1.2,
     bg = "wheat", box.col = "black")
grid()
dev.off()

# 2.2 Numerical vs Categorical
cat("\n2.2. Numerical vs Categorical Analysis\n")

# Salary by Gender
cat("\nSalary by Gender:\n")
print(df %>% group_by(Gender) %>% 
      summarise(Mean = mean(Salary, na.rm = TRUE),
                Median = median(Salary, na.rm = TRUE),
                SD = sd(Salary, na.rm = TRUE),
                Count = n()))

# Salary by Senior Management
cat("\nSalary by Senior Management:\n")
print(df %>% group_by(Senior_Management) %>% 
      summarise(Mean = mean(Salary, na.rm = TRUE),
                Median = median(Salary, na.rm = TRUE),
                SD = sd(Salary, na.rm = TRUE),
                Count = n()))

# Salary by Team (top 10)
cat("\nSalary by Team (Top 10):\n")
top_teams <- names(head(sort(table(df$Team), decreasing = TRUE), 10))
print(df %>% filter(Team %in% top_teams) %>% 
      group_by(Team) %>% 
      summarise(Mean = mean(Salary, na.rm = TRUE),
                Median = median(Salary, na.rm = TRUE),
                SD = sd(Salary, na.rm = TRUE),
                Count = n()))

# Visualizations
png("results/plots/bivariate_numerical_categorical.png", width = 2400, height = 2400, res = 300)
par(mfrow = c(2, 3))

# Salary by Gender - Box plot
boxplot(Salary ~ Gender, data = df, main = "Salary Distribution by Gender",
        xlab = "Gender", ylab = "Salary", col = c("skyblue", "pink", "lightgray"))

# Salary by Gender - Violin plot (using density)
gender_levels <- unique(df$Gender[!is.na(df$Gender)])
for (i in seq_along(gender_levels)) {
  gender_data <- df$Salary[df$Gender == gender_levels[i] & !is.na(df$Salary)]
  if (i == 1) {
    plot(density(gender_data), main = "Salary Distribution by Gender (Density)",
         xlab = "Salary", ylab = "Density", col = "skyblue", lwd = 2)
  } else {
    lines(density(gender_data), col = c("pink", "lightgray")[i-1], lwd = 2)
  }
}
legend("topright", legend = gender_levels, col = c("skyblue", "pink", "lightgray"), lwd = 2)

# Salary by Senior Management - Box plot
boxplot(Salary ~ Senior_Management, data = df, main = "Salary Distribution by Senior Management",
        xlab = "Senior Management", ylab = "Salary", col = c("lightcoral", "lightgreen"))

# Salary by Team (top 10) - Box plot
top_teams_data <- df[df$Team %in% top_teams, ]
boxplot(Salary ~ Team, data = top_teams_data, main = "Salary Distribution by Team (Top 10)",
        xlab = "Team", ylab = "Salary", las = 2, col = "steelblue")

# Bonus % by Gender
boxplot(Bonus_pct ~ Gender, data = df, main = "Bonus % Distribution by Gender",
        xlab = "Gender", ylab = "Bonus %", col = c("skyblue", "pink", "lightgray"))

# Years of Service by Senior Management
boxplot(Years_of_Service ~ Senior_Management, data = df,
        main = "Years of Service by Senior Management",
        xlab = "Senior Management", ylab = "Years of Service",
        col = c("lightcoral", "lightgreen"))
dev.off()

# 2.3 Categorical vs Categorical
cat("\n2.3. Categorical vs Categorical Analysis\n")

# Gender vs Senior Management
cat("\nGender vs Senior Management:\n")
contingency_gender_sm <- table(df$Gender, df$Senior_Management)
print(addmargins(contingency_gender_sm))
write.csv(addmargins(contingency_gender_sm), "results/tables/contingency_gender_senior_management.csv")

# Gender vs Team (top 10)
cat("\nGender vs Team (Top 10):\n")
contingency_gender_team <- table(df[df$Team %in% top_teams, ]$Gender,
                                 df[df$Team %in% top_teams, ]$Team)
print(addmargins(contingency_gender_team))
write.csv(addmargins(contingency_gender_team), "results/tables/contingency_gender_team.csv")

# Visualizations
png("results/plots/bivariate_categorical_categorical.png", width = 2400, height = 1200, res = 300)
par(mfrow = c(1, 2))

# Gender vs Senior Management - Stacked bar chart
contingency_gender_sm_plot <- table(df$Gender, df$Senior_Management)
barplot(contingency_gender_sm_plot, main = "Gender vs Senior Management",
        xlab = "Gender", ylab = "Count", col = c("lightcoral", "lightgreen"),
        legend.text = rownames(contingency_gender_sm_plot), beside = FALSE)

# Gender vs Team (top 5) - Heatmap
contingency_gender_team_plot <- table(df[df$Team %in% top_teams[1:5], ]$Gender,
                                      df[df$Team %in% top_teams[1:5], ]$Team)
heatmap(contingency_gender_team_plot, main = "Gender vs Team (Top 5) - Heatmap",
        xlab = "Team", ylab = "Gender", col = heat.colors(256))
dev.off()

# =============================================================================
# 3. MULTIVARIATE ANALYSIS
# =============================================================================
cat("\n3. MULTIVARIATE ANALYSIS\n")
cat("===============================================================================\n")

# 3.1 Pairwise Relationships
cat("\n3.1. Pairwise Relationships Analysis\n")

# Pair plot for numerical variables
png("results/plots/multivariate_pairplot.png", width = 2400, height = 2400, res = 300)
pairs(df[numerical_cols], main = "Pairwise Relationships", pch = 19, alpha = 0.5)
dev.off()

# 3.2 Multiple Variable Interactions
cat("\n3.2. Multiple Variable Interactions\n")

# Salary by Gender and Senior Management
cat("\nSalary by Gender and Senior Management:\n")
print(df %>% group_by(Gender, Senior_Management) %>% 
      summarise(Mean = mean(Salary, na.rm = TRUE),
                Median = median(Salary, na.rm = TRUE),
                Count = n()))

# Salary by Team and Senior Management (top 5 teams)
cat("\nSalary by Team and Senior Management (Top 5 Teams):\n")
top_5_teams <- names(head(sort(table(df$Team), decreasing = TRUE), 5))
print(df %>% filter(Team %in% top_5_teams) %>% 
      group_by(Team, Senior_Management) %>% 
      summarise(Mean = mean(Salary, na.rm = TRUE),
                Median = median(Salary, na.rm = TRUE),
                Count = n()))

# Visualizations
png("results/plots/multivariate_analysis.png", width = 2400, height = 2400, res = 300)
par(mfrow = c(2, 2))

# Salary by Gender and Senior Management - Grouped bar chart
salary_by_gender_sm <- df %>% group_by(Gender, Senior_Management) %>% 
  summarise(Mean_Salary = mean(Salary, na.rm = TRUE))
barplot(matrix(salary_by_gender_sm$Mean_Salary, nrow = 2, byrow = TRUE),
        main = "Average Salary by Gender and Senior Management",
        xlab = "Gender", ylab = "Average Salary",
        col = c("lightcoral", "lightgreen"),
        legend.text = c("False", "True"), beside = TRUE)

# Salary by Team and Senior Management - Heatmap
salary_by_team_sm <- df %>% filter(Team %in% top_5_teams) %>% 
  group_by(Team, Senior_Management) %>% 
  summarise(Mean_Salary = mean(Salary, na.rm = TRUE)) %>% 
  pivot_wider(names_from = Senior_Management, values_from = Mean_Salary)
rownames(salary_by_team_sm) <- salary_by_team_sm$Team
salary_by_team_sm$Team <- NULL
heatmap(as.matrix(salary_by_team_sm), main = "Average Salary by Team and Senior Management",
        xlab = "Senior Management", ylab = "Team", col = heat.colors(256))

# 3D scatter plot (Salary, Bonus %, Years of Service) colored by Gender
# Create a 2D representation
plot(df$Salary, df$Bonus_pct, col = as.factor(df$Gender),
     main = "Salary vs Bonus % by Gender", xlab = "Salary", ylab = "Bonus %",
     pch = 19, alpha = 0.5)
legend("topright", legend = levels(as.factor(df$Gender)),
       col = 1:length(levels(as.factor(df$Gender))), pch = 19)

# Correlation heatmap with all variables
df_encoded <- df
df_encoded$Gender_encoded <- as.numeric(as.factor(df_encoded$Gender))
df_encoded$Senior_Management_encoded <- as.numeric(df_encoded$Senior_Management)
correlation_all <- cor(df_encoded[c("Salary", "Bonus_pct", "Years_of_Service",
                                     "Gender_encoded", "Senior_Management_encoded")],
                       use = "complete.obs")
corrplot(correlation_all, method = "color", type = "upper", order = "hclust",
         tl.cex = 0.8, tl.col = "black", tl.srt = 45, addCoef.col = "black",
         main = "Correlation Matrix (All Variables)")
dev.off()

# 3.3 Advanced Multivariate Visualizations
cat("\n3.3. Advanced Multivariate Visualizations\n")

# Faceted scatter plots
png("results/plots/multivariate_faceted.png", width = 2400, height = 1200, res = 300)
par(mfrow = c(1, 2))

# Salary vs Bonus % by Gender
gender_levels <- unique(df$Gender[!is.na(df$Gender)])
colors <- c("steelblue", "pink", "lightgray")
for (i in seq_along(gender_levels)) {
  gender_data <- df[df$Gender == gender_levels[i] & !is.na(df$Salary) & !is.na(df$Bonus_pct), ]
  if (i == 1) {
    plot(gender_data$Salary, gender_data$Bonus_pct, main = "Salary vs Bonus % by Gender",
         xlab = "Salary", ylab = "Bonus %", pch = 19, alpha = 0.6, col = colors[i])
  } else {
    points(gender_data$Salary, gender_data$Bonus_pct, pch = 19, alpha = 0.6, col = colors[i])
  }
}
legend("topright", legend = gender_levels, col = colors, pch = 19)
grid()

# Salary vs Bonus % by Senior Management
sm_levels <- c(TRUE, FALSE)
colors_sm <- c("lightgreen", "lightcoral")
for (i in seq_along(sm_levels)) {
  sm_data <- df[df$Senior_Management == sm_levels[i] & !is.na(df$Salary) & !is.na(df$Bonus_pct), ]
  if (i == 1) {
    plot(sm_data$Salary, sm_data$Bonus_pct,
         main = "Salary vs Bonus % by Senior Management",
         xlab = "Salary", ylab = "Bonus %", pch = 19, alpha = 0.6, col = colors_sm[i])
  } else {
    points(sm_data$Salary, sm_data$Bonus_pct, pch = 19, alpha = 0.6, col = colors_sm[i])
  }
}
legend("topright", legend = c("Senior Management", "Non-Senior Management"),
       col = colors_sm, pch = 19)
grid()
dev.off()

cat("\n===============================================================================\n")
cat("UNIVARIATE, BIVARIATE, AND MULTIVARIATE ANALYSIS COMPLETED!\n")
cat("===============================================================================\n")
cat("\nResults saved in:\n")
cat("- Tables: results/tables/\n")
cat("- Plots: results/plots/\n")

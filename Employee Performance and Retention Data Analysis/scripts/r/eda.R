# Exploratory Data Analysis (EDA) for Employee Dataset
# Comprehensive data exploration, cleaning, and visualization

# Load required libraries
library(tidyverse)
library(ggplot2)
library(dplyr)
library(lubridate)
if (requireNamespace("VIM", quietly = TRUE)) {
  library(VIM)
}
library(corrplot)

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
if (!file.exists("data/raw/employees.csv")) {
  stop("Please run this script from the project root directory or ensure data/raw/employees.csv exists")
}

# Create results directories
dir.create("results/plots", recursive = TRUE, showWarnings = FALSE)
dir.create("data/processed", recursive = TRUE, showWarnings = FALSE)

cat("===============================================================================\n")
cat("EXPLORATORY DATA ANALYSIS - EMPLOYEE DATASET\n")
cat("===============================================================================\n")

# =============================================================================
# 1. DATA LOADING
# =============================================================================
cat("\n1. Loading Data...\n")
df <- read.csv("data/raw/employees.csv", stringsAsFactors = FALSE, check.names = FALSE)
cat("Dataset shape:", nrow(df), "rows,", ncol(df), "columns\n")
cat("Columns:", paste(colnames(df), collapse = ", "), "\n")

# =============================================================================
# 2. DATA OVERVIEW
# =============================================================================
cat("\n2. Data Overview...\n")
cat("\nFirst few rows:\n")
print(head(df, 10))
cat("\nData types:\n")
  print(str(df))
cat("\nSummary:\n")
  print(summary(df))
  
# =============================================================================
# 3. MISSING VALUES ANALYSIS
# =============================================================================
cat("\n3. Missing Values Analysis...\n")
missing_data <- colSums(is.na(df))
missing_percent <- (missing_data / nrow(df)) * 100
  missing_df <- data.frame(
  Column = names(missing_data),
  Missing_Count = missing_data,
  Percentage = missing_percent
  )
  missing_df <- missing_df[missing_df$Missing_Count > 0, ]
missing_df <- missing_df[order(-missing_df$Missing_Count), ]
  print(missing_df)
  
# Visualize missing values
png("results/plots/missing_values_heatmap.png", width = 1200, height = 800, res = 300)
if (requireNamespace("VIM", quietly = TRUE)) {
  VIM::aggr(df, col = c('navyblue', 'red'), numbers = TRUE, sortVars = TRUE, 
           labels = names(df), cex.axis = 0.7, gap = 3, ylab = c("Missing data", "Pattern"))
} else {
  # Simple bar plot of missing values
  missing_counts <- colSums(is.na(df))
  barplot(missing_counts[missing_counts > 0], main = "Missing Values by Column",
          xlab = "Column", ylab = "Missing Count", las = 2, col = "steelblue")
}
dev.off()

# =============================================================================
# 4. DATA CLEANING AND PREPROCESSING
# =============================================================================
cat("\n4. Data Cleaning and Preprocessing...\n")
  df_clean <- df
  
# Rename columns for easier handling
colnames(df_clean) <- gsub(" ", "_", colnames(df_clean))
colnames(df_clean) <- gsub("%", "pct", colnames(df_clean))

# Handle missing values in First Name
df_clean$First_Name[is.na(df_clean$First_Name)] <- "Unknown"

# Handle missing values in Gender
cat("Gender distribution before cleaning:\n")
print(table(df_clean$Gender, useNA = "always"))

# Handle missing values in Senior Management (convert to boolean)
df_clean$Senior_Management <- ifelse(df_clean$Senior_Management == "true", TRUE, 
                                     ifelse(df_clean$Senior_Management == "false", FALSE, FALSE))
df_clean$Senior_Management[is.na(df_clean$Senior_Management)] <- FALSE

# Handle missing values in Team
df_clean$Team[is.na(df_clean$Team) | df_clean$Team == ""] <- "Unknown"

# Parse Start Date
df_clean$Start_Date <- as.Date(df_clean$Start_Date, format = "%m/%d/%Y")
df_clean$Start_Year <- year(df_clean$Start_Date)
df_clean$Start_Month <- month(df_clean$Start_Date)
df_clean$Years_of_Service <- as.numeric(difftime(Sys.Date(), df_clean$Start_Date, units = "days")) / 365.25

# Parse Last Login Time (extract hour)
df_clean$Last_Login_Time <- as.POSIXct(df_clean$Last_Login_Time, format = "%I:%M %p")
df_clean$Last_Login_Hour <- hour(df_clean$Last_Login_Time)

# Ensure numeric columns are numeric
df_clean$Salary <- as.numeric(df_clean$Salary)
df_clean$Bonus_pct <- as.numeric(df_clean$Bonus_pct)

# Remove rows with missing critical data (Salary)
df_clean <- df_clean[!is.na(df_clean$Salary), ]

cat("\nDataset shape after cleaning:", nrow(df_clean), "rows,", ncol(df_clean), "columns\n")
cat("Rows removed:", nrow(df) - nrow(df_clean), "\n")

# Save cleaned dataset
write.csv(df_clean, "data/processed/employees_cleaned.csv", row.names = FALSE)
cat("\nCleaned dataset saved to: data/processed/employees_cleaned.csv\n")

# =============================================================================
# 5. NUMERICAL VARIABLE ANALYSIS
# =============================================================================
cat("\n5. Numerical Variable Analysis...\n")
numerical_cols <- c("Salary", "Bonus_pct", "Years_of_Service")

cat("\nDescriptive Statistics:\n")
print(summary(df_clean[numerical_cols]))

# Distribution plots for numerical variables
png("results/plots/numerical_distributions.png", width = 2400, height = 1600, res = 300)
par(mfrow = c(2, 3))
for (col in numerical_cols) {
  # Histogram
  hist(df_clean[[col]], breaks = 50, main = paste("Distribution of", col), 
       xlab = col, col = "steelblue", border = "black")
  # Box plot
  boxplot(df_clean[[col]], main = paste("Box Plot of", col), ylab = col, col = "lightblue")
}
  dev.off()
  
# =============================================================================
# 6. CATEGORICAL VARIABLE ANALYSIS
# =============================================================================
cat("\n6. Categorical Variable Analysis...\n")
categorical_cols <- c("Gender", "Senior_Management", "Team")

for (col in categorical_cols) {
  cat("\n", col, "distribution:\n")
  print(table(df_clean[[col]], useNA = "always"))
  cat("Unique values:", length(unique(df_clean[[col]])), "\n")
}

# Visualize categorical variables
png("results/plots/categorical_distributions.png", width = 2400, height = 800, res = 300)
par(mfrow = c(1, 3))

# Gender distribution
gender_counts <- table(df_clean$Gender)
barplot(gender_counts, main = "Gender Distribution", xlab = "Gender", ylab = "Count", 
        col = c("skyblue", "pink", "lightgray"))

# Senior Management distribution
sm_counts <- table(df_clean$Senior_Management)
barplot(sm_counts, main = "Senior Management Distribution", xlab = "Senior Management", 
        ylab = "Count", col = c("lightcoral", "lightgreen"), names.arg = c("False", "True"))

# Team distribution (top 10)
team_counts <- head(sort(table(df_clean$Team), decreasing = TRUE), 10)
barplot(team_counts, main = "Top 10 Teams by Employee Count", xlab = "Count", 
        ylab = "Team", col = "steelblue", horiz = TRUE, las = 1)
  dev.off()
  
# =============================================================================
# 7. OUTLIER DETECTION
# =============================================================================
cat("\n7. Outlier Detection...\n")
for (col in numerical_cols) {
  Q1 <- quantile(df_clean[[col]], 0.25, na.rm = TRUE)
  Q3 <- quantile(df_clean[[col]], 0.75, na.rm = TRUE)
    IQR <- Q3 - Q1
    lower_bound <- Q1 - 1.5 * IQR
    upper_bound <- Q3 + 1.5 * IQR
  outliers <- sum(df_clean[[col]] < lower_bound | df_clean[[col]] > upper_bound, na.rm = TRUE)
  cat("\n", col, "outliers:", outliers, "(", round(outliers/nrow(df_clean)*100, 2), "%)\n")
  cat("  Lower bound:", round(lower_bound, 2), ", Upper bound:", round(upper_bound, 2), "\n")
}

# Visualize outliers
png("results/plots/outliers_detection.png", width = 2400, height = 800, res = 300)
par(mfrow = c(1, 3))
for (col in numerical_cols) {
  boxplot(df_clean[[col]], main = paste("Outliers in", col), ylab = col, col = "lightcoral")
}
  dev.off()
  
# =============================================================================
# 8. CORRELATION ANALYSIS
# =============================================================================
cat("\n8. Correlation Analysis...\n")
correlation_matrix <- cor(df_clean[numerical_cols], use = "complete.obs")
  cat("\nCorrelation Matrix:\n")
  print(correlation_matrix)

# Visualize correlation matrix
png("results/plots/correlation_matrix.png", width = 1600, height = 1600, res = 300)
corrplot(correlation_matrix, method = "color", type = "upper", order = "hclust", 
         tl.cex = 0.8, tl.col = "black", tl.srt = 45)
dev.off()

# =============================================================================
# 9. RELATIONSHIP ANALYSIS
# =============================================================================
cat("\n9. Relationship Analysis...\n")

# Salary by Gender, Senior Management, and Team
png("results/plots/relationship_analysis.png", width = 2400, height = 2400, res = 300)
par(mfrow = c(2, 2))
  
  # Salary by Gender
boxplot(Salary ~ Gender, data = df_clean, main = "Salary Distribution by Gender", 
        xlab = "Gender", ylab = "Salary", col = c("skyblue", "pink", "lightgray"))

# Salary by Senior Management
boxplot(Salary ~ Senior_Management, data = df_clean, main = "Salary Distribution by Senior Management", 
        xlab = "Senior Management", ylab = "Salary", col = c("lightcoral", "lightgreen"))

# Salary by Team (top 10 teams)
top_teams <- names(head(sort(table(df_clean$Team), decreasing = TRUE), 10))
df_top_teams <- df_clean[df_clean$Team %in% top_teams, ]
boxplot(Salary ~ Team, data = df_top_teams, main = "Salary Distribution by Team (Top 10)", 
        xlab = "Team", ylab = "Salary", las = 2, col = "steelblue")

# Salary vs Bonus %
plot(df_clean$Salary, df_clean$Bonus_pct, main = "Salary vs Bonus %", 
     xlab = "Salary", ylab = "Bonus %", pch = 19, alpha = 0.5, col = "steelblue")
  dev.off()

# =============================================================================
# 10. TIME SERIES ANALYSIS (HIRING TRENDS)
# =============================================================================
cat("\n10. Time Series Analysis (Hiring Trends)...\n")

# Hiring trends by year
hiring_by_year <- df_clean %>%
    group_by(Start_Year) %>%
  summarise(Count = n()) %>%
  filter(!is.na(Start_Year))

png("results/plots/hiring_trends.png", width = 2000, height = 1000, res = 300)
plot(hiring_by_year$Start_Year, hiring_by_year$Count, type = "o", 
     main = "Hiring Trends Over Years", xlab = "Year", ylab = "Number of Employees Hired", 
     pch = 19, col = "steelblue", lwd = 2)
grid()
  dev.off()
  
# Hiring by month
hiring_by_month <- df_clean %>%
  group_by(Start_Month) %>%
  summarise(Count = n()) %>%
  filter(!is.na(Start_Month))

png("results/plots/hiring_by_month.png", width = 2000, height = 1000, res = 300)
barplot(hiring_by_month$Count, names.arg = month.abb[hiring_by_month$Start_Month], 
        main = "Hiring Trends by Month", xlab = "Month", ylab = "Number of Employees Hired", 
        col = "steelblue", border = "black")
grid(nx = NA, ny = NULL)
dev.off()

# =============================================================================
# 11. SUMMARY STATISTICS
# =============================================================================
cat("\n11. Summary Statistics...\n")
cat("\nOverall Statistics:\n")
cat("Total employees:", nrow(df_clean), "\n")
cat("Average salary: $", round(mean(df_clean$Salary, na.rm = TRUE), 2), "\n", sep = "")
cat("Median salary: $", round(median(df_clean$Salary, na.rm = TRUE), 2), "\n", sep = "")
cat("Average bonus %:", round(mean(df_clean$Bonus_pct, na.rm = TRUE), 2), "%\n")
cat("Average years of service:", round(mean(df_clean$Years_of_Service, na.rm = TRUE), 2), "years\n")
cat("Senior management percentage:", round(sum(df_clean$Senior_Management, na.rm = TRUE) / nrow(df_clean) * 100, 2), "%\n")

# Gender statistics
cat("\nGender Statistics:\n")
print(df_clean %>%
  group_by(Gender) %>%
  summarise(Mean_Salary = mean(Salary, na.rm = TRUE),
            Median_Salary = median(Salary, na.rm = TRUE),
            Count = n()))

# Team statistics
cat("\nTop 5 Teams by Average Salary:\n")
print(df_clean %>%
  group_by(Team) %>%
  summarise(Mean_Salary = mean(Salary, na.rm = TRUE)) %>%
  arrange(desc(Mean_Salary)) %>%
  head(5))

cat("\n===============================================================================\n")
  cat("EDA COMPLETED SUCCESSFULLY!\n")
cat("===============================================================================\n")
cat("\nResults saved in:\n")
cat("- Cleaned dataset: data/processed/employees_cleaned.csv\n")
cat("- Plots: results/plots/\n")

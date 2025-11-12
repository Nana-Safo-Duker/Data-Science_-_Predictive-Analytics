# Exploratory Data Analysis (EDA) Script - Position Salaries Dataset
# This script performs comprehensive EDA on the Position Salaries dataset.

# Load necessary libraries
library(tidyverse)
library(ggplot2)
library(dplyr)
library(readr)
library(corrplot)

# Set working directory
project_root <- dirname(dirname(dirname(getwd())))
data_path <- file.path(project_root, "data", "raw", "Position_Salaries.csv")

# Load data
df <- read_csv(data_path)

# Create output directory
output_dir <- file.path(project_root, "results", "figures")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Data Overview
cat("=" , rep("=", 60), "\n", sep = "")
cat("DATA OVERVIEW\n")
cat("=" , rep("=", 60), "\n", sep = "")
cat("\nDataset Shape:", dim(df), "\n")
cat("\nFirst few rows:\n")
print(head(df, 10))
cat("\nDataset Summary:\n")
print(summary(df))
cat("\nMissing Values:\n")
print(colSums(is.na(df)))
cat("\nDuplicate rows:", sum(duplicated(df)), "\n")

# Statistical Summary
cat("\nStatistical Summary:\n")
print(summary(df))

# Visualizations
# 1. Salary Distribution
png(file.path(output_dir, "salary_distribution.png"), width = 1200, height = 600)
p1 <- ggplot(df, aes(x = Salary)) +
  geom_histogram(bins = 10, fill = "skyblue", color = "black", alpha = 0.7) +
  labs(title = "Distribution of Salary", x = "Salary", y = "Frequency") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold"))
print(p1)
dev.off()

# 2. Level vs Salary
png(file.path(output_dir, "level_vs_salary.png"), width = 1200, height = 600)
p2 <- ggplot(df, aes(x = Level, y = Salary)) +
  geom_point(size = 3, color = "coral", alpha = 0.7) +
  geom_line(color = "green", linewidth = 1) +
  labs(title = "Level vs Salary", x = "Level", y = "Salary") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold"))
print(p2)
dev.off()

# 3. Salary by Position
png(file.path(output_dir, "salary_by_position.png"), width = 1400, height = 800)
df_sorted <- df %>% arrange(Salary)
p3 <- ggplot(df_sorted, aes(x = reorder(Position, Salary), y = Salary)) +
  geom_bar(stat = "identity", fill = "steelblue", alpha = 0.8) +
  coord_flip() +
  labs(title = "Salary by Position", x = "Position", y = "Salary ($)") +
  theme_minimal() +
  theme(plot.title = element_text(size = 16, face = "bold")) +
  geom_text(aes(label = paste0("$", format(Salary, big.mark = ","))), 
            hjust = -0.1, size = 3)
print(p3)
dev.off()

# 4. Correlation
correlation <- cor(df$Level, df$Salary)
cat("\nCorrelation (Level vs Salary):", correlation, "\n")

# Save processed data
processed_path <- file.path(project_root, "data", "processed", "processed_data.csv")
dir.create(dirname(processed_path), recursive = TRUE, showWarnings = FALSE)
write_csv(df, processed_path)

cat("\nEDA completed successfully!\n")
cat("Visualizations saved to:", output_dir, "\n")
cat("Processed data saved to:", processed_path, "\n")



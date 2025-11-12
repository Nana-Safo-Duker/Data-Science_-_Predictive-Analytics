# Exploratory Data Analysis Script for Fuel Consumption Dataset
# This script performs comprehensive EDA and saves visualizations.

# Load necessary libraries
library(dplyr)
library(ggplot2)
library(corrplot)
library(VIM)
library(gridExtra)
library(readr)

# Suppress warnings
options(warn = -1)

# Load the dataset
data_path <- file.path("..", "..", "data", "FuelConsumption.csv")
df <- read.csv(data_path, stringsAsFactors = FALSE)

# Clean column names (remove trailing spaces)
colnames(df) <- trimws(colnames(df))

cat("==================================================\n")
cat("EXPLORATORY DATA ANALYSIS\n")
cat("Fuel Consumption Dataset\n")
cat("==================================================\n\n")

cat("Dataset loaded successfully!\n")
cat("Shape:", nrow(df), "rows,", ncol(df), "columns\n")
cat("Columns:", paste(colnames(df), collapse = ", "), "\n\n")

# Data Overview
cat("==================================================\n")
cat("DATA OVERVIEW\n")
cat("==================================================\n\n")
cat("First few rows:\n")
print(head(df, 10))
cat("\nData structure:\n")
str(df)
cat("\nSummary statistics:\n")
print(summary(df))

# Data Quality Assessment
cat("\n==================================================\n")
cat("DATA QUALITY ASSESSMENT\n")
cat("==================================================\n\n")

# Missing values
missing_values <- colSums(is.na(df))
if(sum(missing_values) > 0) {
  cat("Missing Values:\n")
  print(missing_values[missing_values > 0])
} else {
  cat("✓ No missing values found!\n")
}

# Duplicates
cat("\nNumber of duplicate rows:", sum(duplicated(df)), "\n")

# Create output directory
output_dir <- file.path("..", "..", "outputs", "figures")
if(!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Distribution Analysis
cat("\n==================================================\n")
cat("DISTRIBUTION ANALYSIS\n")
cat("==================================================\n\n")

numerical_cols <- c("ENGINE.SIZE", "CYLINDERS", "FUEL.CONSUMPTION", "COEMISSIONS")

# Histograms
p1 <- ggplot(df, aes(x = ENGINE.SIZE)) + 
  geom_histogram(bins = 30, fill = "steelblue", alpha = 0.7, color = "black") +
  labs(title = "Distribution of Engine Size", x = "Engine Size (L)", y = "Frequency") +
  theme_minimal()

p2 <- ggplot(df, aes(x = CYLINDERS)) + 
  geom_histogram(bins = 30, fill = "coral", alpha = 0.7, color = "black") +
  labs(title = "Distribution of Cylinders", x = "Number of Cylinders", y = "Frequency") +
  theme_minimal()

p3 <- ggplot(df, aes(x = FUEL.CONSUMPTION)) + 
  geom_histogram(bins = 30, fill = "green", alpha = 0.7, color = "black") +
  labs(title = "Distribution of Fuel Consumption", x = "Fuel Consumption (L/100km)", y = "Frequency") +
  theme_minimal()

p4 <- ggplot(df, aes(x = COEMISSIONS)) + 
  geom_histogram(bins = 30, fill = "purple", alpha = 0.7, color = "black") +
  labs(title = "Distribution of CO2 Emissions", x = "CO2 Emissions (g/km)", y = "Frequency") +
  theme_minimal()

ggsave(file.path(output_dir, "distribution_analysis_r.png"), 
       grid.arrange(p1, p2, p3, p4, ncol = 2), 
       width = 15, height = 10, dpi = 300)
cat("✓ Distribution plots saved!\n")

# Correlation Analysis
cat("\n==================================================\n")
cat("CORRELATION ANALYSIS\n")
cat("==================================================\n\n")

numerical_data <- df[, numerical_cols]
correlation_matrix <- cor(numerical_data, use = "complete.obs")

png(file.path(output_dir, "correlation_matrix_r.png"), width = 800, height = 800, res = 300)
corrplot(correlation_matrix, method = "color", type = "upper", 
         order = "hclust", tl.cex = 0.8, tl.col = "black",
         addCoef.col = "black", number.cex = 0.7)
dev.off()
cat("✓ Correlation matrix saved!\n")

cat("\nCorrelation Matrix:\n")
print(round(correlation_matrix, 3))

# Categorical Variable Analysis
cat("\n==================================================\n")
cat("CATEGORICAL VARIABLE ANALYSIS\n")
cat("==================================================\n\n")

# Top 15 Vehicle Makes
top_makes <- df %>%
  count(MAKE, sort = TRUE) %>%
  head(15)

p1 <- ggplot(top_makes, aes(x = reorder(MAKE, n), y = n)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 15 Vehicle Makes", x = "Make", y = "Count") +
  theme_minimal()

# Vehicle Class Distribution
vehicle_class <- df %>%
  count(VEHICLE.CLASS, sort = TRUE)

p2 <- ggplot(vehicle_class, aes(x = reorder(VEHICLE.CLASS, n), y = n)) +
  geom_bar(stat = "identity", fill = "coral") +
  coord_flip() +
  labs(title = "Vehicle Class Distribution", x = "Vehicle Class", y = "Count") +
  theme_minimal()

# Fuel Type Distribution
fuel <- df %>%
  count(FUEL, sort = TRUE)

p3 <- ggplot(fuel, aes(x = FUEL, y = n)) +
  geom_bar(stat = "identity", fill = "green", alpha = 0.7) +
  labs(title = "Fuel Type Distribution", x = "Fuel Type", y = "Count") +
  theme_minimal()

# Cylinders Distribution
cylinders <- df %>%
  count(CYLINDERS, sort = TRUE)

p4 <- ggplot(cylinders, aes(x = factor(CYLINDERS), y = n)) +
  geom_bar(stat = "identity", fill = "purple", alpha = 0.7) +
  labs(title = "Cylinders Distribution", x = "Number of Cylinders", y = "Count") +
  theme_minimal()

ggsave(file.path(output_dir, "categorical_analysis_r.png"), 
       grid.arrange(p1, p2, p3, p4, ncol = 2), 
       width = 20, height = 12, dpi = 300)
cat("✓ Categorical analysis plots saved!\n")

# Temporal Trend Analysis
cat("\n==================================================\n")
cat("TEMPORAL TREND ANALYSIS\n")
cat("==================================================\n\n")

yearly_stats <- df %>%
  group_by(Year) %>%
  summarise(
    mean_fuel = mean(FUEL.CONSUMPTION, na.rm = TRUE),
    mean_co2 = mean(COEMISSIONS, na.rm = TRUE),
    mean_engine = mean(ENGINE.SIZE, na.rm = TRUE),
    mean_cylinders = mean(CYLINDERS, na.rm = TRUE)
  )

cat("Yearly Statistics:\n")
print(yearly_stats)

# Plot temporal trends
p1 <- ggplot(yearly_stats, aes(x = Year, y = mean_fuel)) +
  geom_line(color = "blue", size = 1) +
  geom_point(color = "blue", size = 2) +
  labs(title = "Average Fuel Consumption Over Years", 
       x = "Year", y = "Average Fuel Consumption (L/100km)") +
  theme_minimal()

p2 <- ggplot(yearly_stats, aes(x = Year, y = mean_co2)) +
  geom_line(color = "red", size = 1) +
  geom_point(color = "red", size = 2) +
  labs(title = "Average CO2 Emissions Over Years", 
       x = "Year", y = "Average CO2 Emissions (g/km)") +
  theme_minimal()

p3 <- ggplot(yearly_stats, aes(x = Year, y = mean_engine)) +
  geom_line(color = "green", size = 1) +
  geom_point(color = "green", size = 2) +
  labs(title = "Average Engine Size Over Years", 
       x = "Year", y = "Average Engine Size (L)") +
  theme_minimal()

p4 <- ggplot(yearly_stats, aes(x = Year, y = mean_cylinders)) +
  geom_line(color = "orange", size = 1) +
  geom_point(color = "orange", size = 2) +
  labs(title = "Average Number of Cylinders Over Years", 
       x = "Year", y = "Average Number of Cylinders") +
  theme_minimal()

ggsave(file.path(output_dir, "yearly_trends_r.png"), 
       grid.arrange(p1, p2, p3, p4, ncol = 2), 
       width = 16, height = 10, dpi = 300)
cat("✓ Temporal trend plots saved!\n")

# Summary Insights
cat("\n==================================================\n")
cat("SUMMARY INSIGHTS\n")
cat("==================================================\n\n")
cat("1. Dataset contains", nrow(df), "records with", ncol(df), "features\n")
cat("2. Time period:", min(df$Year), "-", max(df$Year), "\n")
cat("3. Number of unique makes:", length(unique(df$MAKE)), "\n")
cat("4. Number of unique models:", length(unique(df$MODEL)), "\n")
cat("5. Average fuel consumption:", round(mean(df$FUEL.CONSUMPTION, na.rm = TRUE), 2), "L/100km\n")
cat("6. Average CO2 emissions:", round(mean(df$COEMISSIONS, na.rm = TRUE), 2), "g/km\n")
cat("7. Strongest correlation: Fuel Consumption vs CO2 Emissions =", 
    round(cor(df$FUEL.CONSUMPTION, df$COEMISSIONS, use = "complete.obs"), 3), "\n")

cat("\n==================================================\n")
cat("EDA COMPLETE!\n")
cat("==================================================\n")

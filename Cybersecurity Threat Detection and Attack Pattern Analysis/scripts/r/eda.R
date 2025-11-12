# Exploratory Data Analysis for Cybersecurity Attacks Dataset (R)

# Load required libraries
library(data.table)
library(dplyr)
library(tidyr)
library(ggplot2)
library(plotly)
library(corrplot)
library(VIM)
library(naniar)

# Set working directory (adjust as needed)
setwd("../../")

# Load dataset
df <- read.csv("data/Cybersecurity_attacks.csv", stringsAsFactors = FALSE)

# Clean column names
colnames(df) <- trimws(colnames(df))

# Remove the '.' column if it exists
if ("." %in% colnames(df)) {
  df <- df[, !colnames(df) %in% "."]
}

# Parse Time column
if ("Time" %in% colnames(df)) {
  parse_time <- function(time_str) {
    if (is.na(time_str)) {
      return(list(start = NA, end = NA))
    }
    time_str <- as.character(time_str)
    if (grepl("-", time_str)) {
      parts <- strsplit(time_str, "-")[[1]]
      return(list(start = as.numeric(parts[1]), end = as.numeric(parts[2])))
    } else {
      time_val <- as.numeric(time_str)
      return(list(start = time_val, end = time_val))
    }
  }
  
  time_parsed <- lapply(df$Time, parse_time)
  df$Time_Start <- sapply(time_parsed, function(x) x$start)
  df$Time_End <- sapply(time_parsed, function(x) x$end)
  df$Time_Duration <- df$Time_End - df$Time_Start
  df$Datetime_Start <- as.POSIXct(df$Time_Start, origin = "1970-01-01")
  df$Hour <- as.numeric(format(df$Datetime_Start, "%H"))
  df$DayOfWeek <- weekdays(df$Datetime_Start)
  df$Month <- as.numeric(format(df$Datetime_Start, "%m"))
}

# Display dataset information
cat("Dataset Shape:", nrow(df), "rows,", ncol(df), "columns\n")
cat("Columns:", paste(colnames(df), collapse = ", "), "\n")

# Missing values analysis
missing_values <- colSums(is.na(df))
missing_percentage <- (missing_values / nrow(df)) * 100
missing_df <- data.frame(
  Column = names(missing_values),
  Missing_Count = missing_values,
  Missing_Percentage = missing_percentage
)
missing_df <- missing_df[missing_df$Missing_Count > 0, ]
cat("\nMissing Values:\n")
print(missing_df)

# Summary statistics
cat("\nSummary Statistics:\n")
summary(df)

# Attack category distribution
if ("Attack.category" %in% colnames(df)) {
  attack_category_counts <- table(df$Attack.category)
  cat("\nAttack Category Distribution:\n")
  print(sort(attack_category_counts, decreasing = TRUE))
  
  # Visualization
  png("visualizations/attack_category_distribution_R.png", width = 1200, height = 800, res = 300)
  barplot(sort(attack_category_counts, decreasing = TRUE)[1:10], 
          main = "Top 10 Attack Categories",
          xlab = "Attack Category",
          ylab = "Count",
          las = 2)
  dev.off()
}

# Protocol distribution
if ("Protocol" %in% colnames(df)) {
  protocol_counts <- table(df$Protocol)
  cat("\nProtocol Distribution:\n")
  print(sort(protocol_counts, decreasing = TRUE)[1:10])
}

cat("\nEDA Complete!\n")




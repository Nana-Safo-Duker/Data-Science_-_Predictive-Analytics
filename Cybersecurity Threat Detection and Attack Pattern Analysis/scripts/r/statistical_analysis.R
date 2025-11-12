# Statistical Analysis for Cybersecurity Attacks Dataset (R)

# Load required libraries
library(data.table)
library(dplyr)
library(ggplot2)
library(corrplot)
library(car)
library(psych)

# Load dataset
df <- read.csv("data/Cybersecurity_attacks.csv", stringsAsFactors = FALSE)
colnames(df) <- trimws(colnames(df))

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

# Descriptive Statistics
cat("Descriptive Statistics:\n")
summary(df)

# Correlation analysis
if ("Source.Port" %in% colnames(df) && "Destination.Port" %in% colnames(df)) {
  cor_result <- cor(df$Source.Port, df$Destination.Port, use = "complete.obs")
  cat("\nCorrelation between Source Port and Destination Port:", cor_result, "\n")
}

# Chi-square test
if ("Attack.category" %in% colnames(df) && "Protocol" %in% colnames(df)) {
  contingency_table <- table(df$Attack.category, df$Protocol)
  chi_test <- chisq.test(contingency_table)
  cat("\nChi-square test:\n")
  print(chi_test)
}

# ANOVA test
if ("Attack.category" %in% colnames(df) && "Destination.Port" %in% colnames(df)) {
  aov_result <- aov(Destination.Port ~ Attack.category, data = df)
  cat("\nANOVA test:\n")
  print(summary(aov_result))
}

cat("\nStatistical Analysis Complete!\n")




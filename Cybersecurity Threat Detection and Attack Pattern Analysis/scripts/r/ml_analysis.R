# Machine Learning Analysis for Cybersecurity Attacks Dataset (R)

# Load required libraries
library(data.table)
library(dplyr)
library(caret)
library(randomForest)
library(e1071)
library(rpart)
library(rpart.plot)
library(xgboost)
library(glmnet)

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

# Prepare features
features <- c("Source.Port", "Destination.Port", "Hour", "Month")
if ("Time_Duration" %in% colnames(df)) {
  features <- c(features, "Time_Duration")
}

# Encode categorical variables
if ("Protocol" %in% colnames(df)) {
  df$Protocol_encoded <- as.numeric(as.factor(df$Protocol))
  features <- c(features, "Protocol_encoded")
}

# Target variable
if ("Attack.category" %in% colnames(df)) {
  df$Attack_category_factor <- as.factor(df$Attack.category)
  target <- "Attack_category_factor"
} else {
  target <- NULL
}

# Remove rows with missing values
df_clean <- df[complete.cases(df[, features]), ]

# Split data
set.seed(42)
train_index <- createDataPartition(df_clean[[target]], p = 0.8, list = FALSE)
train_data <- df_clean[train_index, ]
test_data <- df_clean[-train_index, ]

# Train Random Forest
if (target %in% colnames(train_data)) {
  formula <- as.formula(paste(target, "~", paste(features, collapse = "+")))
  
  # Random Forest
  rf_model <- randomForest(formula, data = train_data, ntree = 100, importance = TRUE)
  rf_pred <- predict(rf_model, test_data)
  rf_accuracy <- mean(rf_pred == test_data[[target]])
  cat("Random Forest Accuracy:", rf_accuracy, "\n")
  
  # Decision Tree
  dt_model <- rpart(formula, data = train_data, method = "class")
  dt_pred <- predict(dt_model, test_data, type = "class")
  dt_accuracy <- mean(dt_pred == test_data[[target]])
  cat("Decision Tree Accuracy:", dt_accuracy, "\n")
  
  # Naive Bayes
  nb_model <- naiveBayes(formula, data = train_data)
  nb_pred <- predict(nb_model, test_data)
  nb_accuracy <- mean(nb_pred == test_data[[target]])
  cat("Naive Bayes Accuracy:", nb_accuracy, "\n")
}

cat("\nMachine Learning Analysis Complete!\n")




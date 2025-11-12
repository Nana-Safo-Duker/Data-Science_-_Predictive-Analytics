"""
Script to populate R notebooks with basic content
"""
import json
import os

def create_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source if isinstance(source, list) else [line + "\n" for line in source.split("\n")]
    }

def create_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source if isinstance(source, list) else [line + "\n" for line in source.split("\n")]
    }

# R notebooks content
r_notebooks = {
    "notebooks/r/01_EDA_Cybersecurity_Attacks.ipynb": {
        "cells": [
            create_markdown_cell("# Exploratory Data Analysis - Cybersecurity Attacks Dataset (R)\n\n## Overview\nThis notebook provides comprehensive exploratory data analysis of the cybersecurity attacks dataset using R."),
            create_code_cell("""# Load required libraries
library(data.table)
library(dplyr)
library(tidyr)
library(ggplot2)
library(plotly)
library(corrplot)
library(VIM)
library(naniar)

# Set working directory
setwd("../../")"""),
            create_markdown_cell("## 1. Data Loading"),
            create_code_cell("""# Load dataset
df <- read.csv("data/Cybersecurity_attacks.csv", stringsAsFactors = FALSE)

# Clean column names
colnames(df) <- trimws(colnames(df))

# Remove the '.' column if it exists
if ("." %in% colnames(df)) {
  df <- df[, !colnames(df) %in% "."]
}

# Display dataset information
cat("Dataset Shape:", nrow(df), "rows,", ncol(df), "columns\\n")
head(df)"""),
            create_markdown_cell("## 2. Data Cleaning and Preprocessing"),
            create_code_cell("""# Parse Time column
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

cat("Time column parsed successfully!\\n")"""),
            create_markdown_cell("## 3. Missing Values Analysis"),
            create_code_cell("""# Missing values analysis
missing_values <- colSums(is.na(df))
missing_percentage <- (missing_values / nrow(df)) * 100
missing_df <- data.frame(
  Column = names(missing_values),
  Missing_Count = missing_values,
  Missing_Percentage = missing_percentage
)
missing_df <- missing_df[missing_df$Missing_Count > 0, ]
print(missing_df)"""),
            create_markdown_cell("## 4. Categorical Variables Analysis"),
            create_code_cell("""# Attack category distribution
if ("Attack.category" %in% colnames(df)) {
  attack_category_counts <- table(df$Attack.category)
  print(sort(attack_category_counts, decreasing = TRUE)[1:10])
  
  # Visualization
  barplot(sort(attack_category_counts, decreasing = TRUE)[1:10], 
          main = "Top 10 Attack Categories",
          xlab = "Attack Category",
          ylab = "Count",
          las = 2)
}"""),
            create_markdown_cell("## 5. Numerical Variables Analysis"),
            create_code_cell("""# Summary statistics
if ("Source.Port" %in% colnames(df) && "Destination.Port" %in% colnames(df)) {
  summary(df$Source.Port)
  summary(df$Destination.Port)
  
  # Histograms
  hist(df$Source.Port, main = "Source Port Distribution", xlab = "Source Port")
  hist(df$Destination.Port, main = "Destination Port Distribution", xlab = "Destination Port")
}"""),
        ],
        "metadata": {
            "kernelspec": {"display_name": "IRkernel", "language": "R", "name": "ir"},
            "language_info": {"name": "R", "version": "4.0.0"}
        }
    },
    "notebooks/r/02_Statistical_Analysis.ipynb": {
        "cells": [
            create_markdown_cell("# Statistical Analysis - Cybersecurity Attacks Dataset (R)\n\n## Overview\nThis notebook provides comprehensive statistical analysis including descriptive statistics, inferential statistics, and hypothesis testing."),
            create_code_cell("""# Load required libraries
library(data.table)
library(dplyr)
library(ggplot2)
library(corrplot)
library(car)
library(psych)

# Load and prepare data (same as EDA notebook)
# [Include data loading code from EDA notebook]"""),
            create_markdown_cell("## 1. Descriptive Statistics"),
            create_code_cell("""# Descriptive statistics
summary(df)

# For numerical variables
if ("Source.Port" %in% colnames(df)) {
  describe(df$Source.Port)
  describe(df$Destination.Port)
}"""),
            create_markdown_cell("## 2. Hypothesis Testing"),
            create_code_cell("""# Chi-square test
if ("Attack.category" %in% colnames(df) && "Protocol" %in% colnames(df)) {
  contingency_table <- table(df$Attack.category, df$Protocol)
  chi_test <- chisq.test(contingency_table)
  print(chi_test)
}

# ANOVA test
if ("Attack.category" %in% colnames(df) && "Destination.Port" %in% colnames(df)) {
  aov_result <- aov(Destination.Port ~ Attack.category, data = df)
  print(summary(aov_result))
}"""),
            create_markdown_cell("## 3. Correlation Analysis"),
            create_code_cell("""# Correlation matrix
numerical_cols <- c("Source.Port", "Destination.Port", "Time_Duration", "Hour")
numerical_cols <- numerical_cols[numerical_cols %in% colnames(df)]

if (length(numerical_cols) > 1) {
  cor_matrix <- cor(df[, numerical_cols], use = "complete.obs")
  corrplot(cor_matrix, method = "color", type = "upper", order = "hclust")
}"""),
        ],
        "metadata": {
            "kernelspec": {"display_name": "IRkernel", "language": "R", "name": "ir"},
            "language_info": {"name": "R", "version": "4.0.0"}
        }
    },
    "notebooks/r/03_Univariate_Bivariate_Multivariate_Analysis.ipynb": {
        "cells": [
            create_markdown_cell("# Univariate, Bivariate, and Multivariate Analysis (R)\n\n## Overview\nThis notebook provides comprehensive analysis of individual variables, relationships between variables, and patterns across multiple variables."),
            create_code_cell("""# Load required libraries
library(data.table)
library(dplyr)
library(ggplot2)
library(corrplot)

# Load and prepare data
# [Include data loading code from EDA notebook]"""),
            create_markdown_cell("## 1. Univariate Analysis"),
            create_code_cell("""# Univariate analysis for numerical variables
if ("Source.Port" %in% colnames(df)) {
  # Histogram
  hist(df$Source.Port, main = "Source Port Distribution", xlab = "Source Port")
  
  # Box plot
  boxplot(df$Source.Port, main = "Source Port Box Plot")
  
  # Summary statistics
  summary(df$Source.Port)
}"""),
            create_markdown_cell("## 2. Bivariate Analysis"),
            create_code_cell("""# Scatter plot
if ("Source.Port" %in% colnames(df) && "Destination.Port" %in% colnames(df)) {
  plot(df$Source.Port, df$Destination.Port, 
       main = "Source Port vs Destination Port",
       xlab = "Source Port",
       ylab = "Destination Port")
  
  # Correlation
  cor_result <- cor(df$Source.Port, df$Destination.Port, use = "complete.obs")
  cat("Correlation:", cor_result, "\\n")
}

# Box plot by category
if ("Attack.category" %in% colnames(df) && "Destination.Port" %in% colnames(df)) {
  boxplot(Destination.Port ~ Attack.category, data = df,
          main = "Destination Port by Attack Category",
          xlab = "Attack Category",
          ylab = "Destination Port")
}"""),
            create_markdown_cell("## 3. Multivariate Analysis"),
            create_code_cell("""# Correlation matrix
numerical_cols <- c("Source.Port", "Destination.Port", "Time_Duration", "Hour")
numerical_cols <- numerical_cols[numerical_cols %in% colnames(df)]

if (length(numerical_cols) > 1) {
  cor_matrix <- cor(df[, numerical_cols], use = "complete.obs")
  corrplot(cor_matrix, method = "color", type = "upper", order = "hclust",
           tl.cex = 0.8, tl.col = "black")
}"""),
        ],
        "metadata": {
            "kernelspec": {"display_name": "IRkernel", "language": "R", "name": "ir"},
            "language_info": {"name": "R", "version": "4.0.0"}
        }
    },
    "notebooks/r/04_ML_Analysis.ipynb": {
        "cells": [
            create_markdown_cell("# Machine Learning Analysis - Cybersecurity Attacks Dataset (R)\n\n## Overview\nThis notebook implements machine learning algorithms to classify cybersecurity attacks."),
            create_code_cell("""# Load required libraries
library(data.table)
library(dplyr)
library(caret)
library(randomForest)
library(e1071)
library(rpart)
library(rpart.plot)
library(xgboost)

# Load and prepare data
# [Include data loading code from EDA notebook]"""),
            create_markdown_cell("## 1. Feature Engineering"),
            create_code_cell("""# Prepare features
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
}

# Remove rows with missing values
df_clean <- df[complete.cases(df[, features]), ]"""),
            create_markdown_cell("## 2. Model Training"),
            create_code_cell("""# Split data
set.seed(42)
train_index <- createDataPartition(df_clean[[target]], p = 0.8, list = FALSE)
train_data <- df_clean[train_index, ]
test_data <- df_clean[-train_index, ]

# Train Random Forest
formula <- as.formula(paste(target, "~", paste(features, collapse = "+")))
rf_model <- randomForest(formula, data = train_data, ntree = 100, importance = TRUE)
rf_pred <- predict(rf_model, test_data)
rf_accuracy <- mean(rf_pred == test_data[[target]])
cat("Random Forest Accuracy:", rf_accuracy, "\\n")

# Train Decision Tree
dt_model <- rpart(formula, data = train_data, method = "class")
dt_pred <- predict(dt_model, test_data, type = "class")
dt_accuracy <- mean(dt_pred == test_data[[target]])
cat("Decision Tree Accuracy:", dt_accuracy, "\\n")"""),
            create_markdown_cell("## 3. Model Evaluation"),
            create_code_cell("""# Confusion matrix
confusionMatrix(rf_pred, test_data[[target]])

# Feature importance
importance(rf_model)
varImpPlot(rf_model)"""),
        ],
        "metadata": {
            "kernelspec": {"display_name": "IRkernel", "language": "R", "name": "ir"},
            "language_info": {"name": "R", "version": "4.0.0"}
        }
    }
}

# Generate R notebooks
for notebook_path, notebook_data in r_notebooks.items():
    os.makedirs(os.path.dirname(notebook_path), exist_ok=True)
    notebook = {
        "cells": notebook_data["cells"],
        "metadata": notebook_data["metadata"],
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"Generated: {notebook_path}")

print("\nAll R notebooks generated successfully!")




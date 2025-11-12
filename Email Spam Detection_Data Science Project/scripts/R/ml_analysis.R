# Machine Learning Analysis for Email Spam Detection
# R Script

# Load required libraries
library(tidyverse)
library(caret)
library(tm)
library(e1071)
library(randomForest)
library(xgboost)
library(naivebayes)
library(pROC)
library(ggplot2)

# Load and prepare data
df <- read.csv("../../data/emails_spam_clean.csv", stringsAsFactors = FALSE)

# Clean text function
clean_text <- function(text) {
  text <- tolower(text)
  text <- gsub("http\\S+|www\\S+|https\\S+", "", text, perl = TRUE)
  text <- gsub("\\S+@\\S+", "", text)
  text <- gsub("[^a-zA-Z\\s]", "", text)
  text <- gsub("\\s+", " ", text)
  text <- trimws(text)
  return(text)
}

# Clean text
df$cleaned_text <- sapply(df$text, clean_text)
df <- df[df$cleaned_text != "", ]

# Create corpus and document-term matrix
cat("Creating document-term matrix...\n")
corpus <- Corpus(VectorSource(df$cleaned_text))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stripWhitespace)

# Create DTM
dtm <- DocumentTermMatrix(corpus, control = list(
  bounds = list(global = c(2, Inf)),
  weighting = weightTfIdf
))

# Remove sparse terms
dtm <- removeSparseTerms(dtm, 0.95)

# Convert to matrix
X <- as.matrix(dtm)
y <- as.factor(df$spam)

# Split data
set.seed(42)
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[trainIndex, ]
X_test <- X[-trainIndex, ]
y_train <- y[trainIndex]
y_test <- y[-trainIndex]

# Helper function for string concatenation
`%+%` <- function(a, b) paste0(a, b)

# Train and evaluate model function
train_and_evaluate <- function(model, model_name, X_train, X_test, y_train, y_test) {
  cat("\n", paste(rep("=", 60), collapse = ""), "\n", sep = "")
  cat("Training", model_name, "\n")
  cat(paste(rep("=", 60), collapse = ""), "\n", sep = "")
  
  # Train model
  cat("Training model...\n")
  trained_model <- train(X_train, y_train, method = model, trControl = trainControl(method = "cv", number = 5))
  
  # Predictions
  y_pred <- predict(trained_model, X_test)
  y_pred_proba <- predict(trained_model, X_test, type = "prob")[, 2]
  
  # Metrics
  cm <- confusionMatrix(y_pred, y_test)
  accuracy <- cm$overall["Accuracy"]
  precision <- cm$byClass["Precision"]
  recall <- cm$byClass["Recall"]
  f1 <- cm$byClass["F1"]
  
  cat("\n", model_name, "Results:\n", sep = "")
  cat("  Accuracy:", accuracy, "\n")
  cat("  Precision:", precision, "\n")
  cat("  Recall:", recall, "\n")
  cat("  F1-Score:", f1, "\n")
  
  # ROC-AUC
  roc_auc <- auc(roc(y_test, as.numeric(y_pred_proba)))
  cat("  ROC-AUC:", roc_auc, "\n")
  
  # Confusion matrix
  cat("\nConfusion Matrix:\n")
  print(cm$table)
  
  # Classification report
  cat("\nClassification Report:\n")
  print(cm)
  
  return(list(
    model = trained_model,
    model_name = model_name,
    accuracy = accuracy,
    precision = precision,
    recall = recall,
    f1 = f1,
    roc_auc = roc_auc,
    confusion_matrix = cm$table,
    y_test = y_test,
    y_pred = y_pred,
    y_pred_proba = y_pred_proba
  ))
}

# Train models
results <- list()

# Naive Bayes
cat("\nTraining Naive Bayes...\n")
nb_result <- train_and_evaluate("naive_bayes", "Naive Bayes", X_train, X_test, y_train, y_test)
results[[length(results) + 1]] <- nb_result

# SVM
cat("\nTraining SVM...\n")
svm_result <- train_and_evaluate("svmLinear", "SVM", X_train, X_test, y_train, y_test)
results[[length(results) + 1]] <- svm_result

# Random Forest
cat("\nTraining Random Forest...\n")
rf_result <- train_and_evaluate("rf", "Random Forest", X_train, X_test, y_train, y_test)
results[[length(results) + 1]] <- rf_result

# Logistic Regression
cat("\nTraining Logistic Regression...\n")
lr_result <- train_and_evaluate("glm", "Logistic Regression", X_train, X_test, y_train, y_test)
results[[length(results) + 1]] <- lr_result

# XGBoost
cat("\nTraining XGBoost...\n")
xgb_result <- train_and_evaluate("xgbTree", "XGBoost", X_train, X_test, y_train, y_test)
results[[length(results) + 1]] <- xgb_result

# Compare models
cat("\n", paste(rep("=", 60), collapse = ""), "\n", sep = "")
cat("MODEL COMPARISON\n")
cat(paste(rep("=", 60), collapse = ""), "\n", sep = "")

comparison <- data.frame(
  Model = sapply(results, function(x) x$model_name),
  Accuracy = sapply(results, function(x) as.numeric(x$accuracy)),
  Precision = sapply(results, function(x) as.numeric(x$precision)),
  Recall = sapply(results, function(x) as.numeric(x$recall)),
  F1_Score = sapply(results, function(x) as.numeric(x$f1)),
  ROC_AUC = sapply(results, function(x) as.numeric(x$roc_auc))
)

print(comparison)

# Visualize comparison
png("../../output/figures/model_comparison_R.png", width = 1600, height = 1200, res = 300)
par(mfrow = c(2, 2))

metrics <- c("Accuracy", "Precision", "Recall", "F1_Score")
for (i in 1:4) {
  metric <- metrics[i]
  barplot(comparison[[metric]], names.arg = comparison$Model, 
          main = paste(metric, "Comparison"), ylab = metric, las = 2, col = "steelblue")
}

dev.off()

# Find best model
best_idx <- which.max(comparison$F1_Score)
best_model <- comparison$Model[best_idx]
cat("\nBest Model (by F1-Score):", best_model, "\n")
cat("  F1-Score:", comparison$F1_Score[best_idx], "\n")
cat("  Accuracy:", comparison$Accuracy[best_idx], "\n")

# Save best model
saveRDS(results[[best_idx]]$model, paste0("../../models/best_model_", gsub(" ", "_", best_model), ".rds"))
cat("\nBest model saved!\n")

cat("\n", paste(rep("=", 60), collapse = ""), "\n", sep = "")
cat("Machine Learning Analysis Complete!\n")
cat(paste(rep("=", 60), collapse = ""), "\n", sep = "")


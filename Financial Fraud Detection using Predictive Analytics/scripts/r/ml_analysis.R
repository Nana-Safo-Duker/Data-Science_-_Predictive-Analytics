# Machine Learning Analysis Script for Fraud Detection (R)
#
# This script implements machine learning models for fraud detection:
# 1. Data preprocessing and feature engineering
# 2. Model training (Logistic Regression, Random Forest, XGBoost)
# 3. Model evaluation and validation
# 4. Feature importance analysis

# Load libraries
library(tidyverse)
library(caret)
library(randomForest)
library(xgboost)
library(pROC)
library(ROCR)
library(ggplot2)

# Set options
options(warn = -1)
set.seed(42)

# Set up paths
project_root <- dirname(dirname(dirname(getwd())))
data_path <- file.path(project_root, "data", "fraud_data.csv")
output_dir <- file.path(project_root, "outputs", "figures")
model_dir <- file.path(project_root, "outputs", "models")

# Create output directories if they don't exist
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)

# Load data
cat("Loading data...\n")
df <- read.csv(data_path, stringsAsFactors = FALSE)
cat("Data loaded:", dim(df), "\n")
cat("Fraud rate:", mean(df$isFraud), "\n")
cat("Target distribution:\n")
print(table(df$isFraud))

# ============================================================================
# 1. DATA PREPROCESSING
# ============================================================================
cat("\n", rep("=", 80), "\n", sep = "")
cat("DATA PREPROCESSING\n")
cat(rep("=", 80), "\n", sep = "")

# Select features for modeling
key_features <- c("TransactionAmt", "card1", "card2", "card3", "card5", 
                  "addr1", "addr2", "dist1", "dist2")
key_features <- key_features[key_features %in% colnames(df)]

# Add some C and D features if available
c_features <- grep("^C[0-9]+$", colnames(df), value = TRUE)[1:10]
d_features <- grep("^D[0-9]+$", colnames(df), value = TRUE)[1:10]

features <- c(key_features, c_features, d_features)
features <- features[features %in% colnames(df)]

cat("Selected", length(features), "features for modeling\n")

# Prepare data
X <- df[features]
y <- df$isFraud

# Handle missing values
for(col in features) {
  if(any(is.na(X[[col]]))) {
    X[[col]][is.na(X[[col]])] <- median(X[[col]], na.rm = TRUE)
  }
}

cat("X shape:", dim(X), "\n")
cat("y shape:", length(y), "\n")
cat("Missing values in X:", sum(is.na(X)), "\n")

# Split data into train and test sets
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[trainIndex, ]
X_test <- X[-trainIndex, ]
y_train <- y[trainIndex]
y_test <- y[-trainIndex]

cat("\nTrain set:", dim(X_train), "Fraud rate:", mean(y_train), "\n")
cat("Test set:", dim(X_test), "Fraud rate:", mean(y_test), "\n")

# ============================================================================
# 2. MODEL TRAINING
# ============================================================================
cat("\n", rep("=", 80), "\n", sep = "")
cat("MODEL TRAINING\n")
cat(rep("=", 80), "\n", sep = "")

models <- list()
predictions <- list()
predictions_proba <- list()

# Model 1: Logistic Regression
cat("\nTraining Logistic Regression...\n")
train_data <- cbind(X_train, isFraud = y_train)
lr_model <- glm(isFraud ~ ., data = train_data, family = binomial)
lr_pred_proba <- predict(lr_model, newdata = X_test, type = "response")
lr_pred <- ifelse(lr_pred_proba > 0.5, 1, 0)

models[["Logistic Regression"]] <- lr_model
predictions[["Logistic Regression"]] <- lr_pred
predictions_proba[["Logistic Regression"]] <- lr_pred_proba

cat("Logistic Regression trained!\n")
cat("AUC-ROC:", auc(roc(y_test, lr_pred_proba)), "\n")
cat("Accuracy:", mean(lr_pred == y_test), "\n")

# Model 2: Random Forest
cat("\nTraining Random Forest...\n")
train_data <- cbind(X_train, isFraud = as.factor(y_train))
rf_model <- randomForest(isFraud ~ ., data = train_data,
                         ntree = 100, mtry = sqrt(ncol(X_train)),
                         classwt = c(1, sum(y_train == 0) / sum(y_train == 1)))
rf_pred_proba <- predict(rf_model, newdata = X_test, type = "prob")[, 2]
rf_pred <- predict(rf_model, newdata = X_test)

models[["Random Forest"]] <- rf_model
predictions[["Random Forest"]] <- as.numeric(rf_pred) - 1
predictions_proba[["Random Forest"]] <- rf_pred_proba

cat("Random Forest trained!\n")
cat("AUC-ROC:", auc(roc(y_test, rf_pred_proba)), "\n")
cat("Accuracy:", mean((as.numeric(rf_pred) - 1) == y_test), "\n")

# Model 3: XGBoost
cat("\nTraining XGBoost...\n")
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
dtest <- xgb.DMatrix(data = as.matrix(X_test), label = y_test)

xgb_model <- xgboost(data = dtrain, nrounds = 100, max_depth = 6, 
                     eta = 0.1, objective = "binary:logistic", 
                     eval_metric = "auc", verbose = 0,
                     scale_pos_weight = sum(y_train == 0) / sum(y_train == 1))
xgb_pred_proba <- predict(xgb_model, dtest)
xgb_pred <- ifelse(xgb_pred_proba > 0.5, 1, 0)

models[["XGBoost"]] <- xgb_model
predictions[["XGBoost"]] <- xgb_pred
predictions_proba[["XGBoost"]] <- xgb_pred_proba

cat("XGBoost trained!\n")
cat("AUC-ROC:", auc(roc(y_test, xgb_pred_proba)), "\n")
cat("Accuracy:", mean(xgb_pred == y_test), "\n")

# ============================================================================
# 3. MODEL EVALUATION
# ============================================================================
cat("\n", rep("=", 80), "\n", sep = "")
cat("MODEL EVALUATION\n")
cat(rep("=", 80), "\n", sep = "")

# ROC curves
roc_curves <- lapply(predictions_proba, function(pred) roc(y_test, pred))

# Plot ROC curves
png(file.path(output_dir, "roc_curves_r.png"), width = 1200, height = 1000, res = 300)
plot(roc_curves[[1]], main = "ROC Curve Comparison", col = 2, lwd = 2)
for(i in 2:length(roc_curves)) {
  lines(roc_curves[[i]], col = i + 1, lwd = 2)
}
legend("bottomright", legend = names(roc_curves), 
       col = 2:(length(roc_curves) + 1), lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")
dev.off()
cat("\nROC curves saved to", file.path(output_dir, "roc_curves_r.png"), "\n")

# Model comparison
comparison <- data.frame(
  Model = names(predictions_proba),
  AUC_ROC = sapply(roc_curves, function(r) as.numeric(auc(r))),
  Accuracy = sapply(1:length(predictions), function(i) {
    mean(predictions[[i]] == y_test)
  })
)

cat("\nModel Comparison:\n")
cat(rep("=", 80), "\n", sep = "")
print(comparison[order(comparison$AUC_ROC, decreasing = TRUE), ])

# ============================================================================
# 4. FEATURE IMPORTANCE ANALYSIS
# ============================================================================
cat("\n", rep("=", 80), "\n", sep = "")
cat("FEATURE IMPORTANCE ANALYSIS\n")
cat(rep("=", 80), "\n", sep = "")

# Random Forest feature importance
if("Random Forest" %in% names(models)) {
  rf_importance <- data.frame(
    feature = features,
    importance = importance(models[["Random Forest"]])
  )
  rf_importance <- rf_importance[order(rf_importance$importance, decreasing = TRUE), ]
  rf_importance_top <- head(rf_importance, 20)
  
  png(file.path(output_dir, "feature_importance_rf_r.png"), 
      width = 1200, height = 800, res = 300)
  barplot(rf_importance_top$importance, names.arg = rf_importance_top$feature,
          main = "Random Forest - Top 20 Feature Importance",
          xlab = "Importance", ylab = "Feature", horiz = TRUE, las = 1)
  dev.off()
  cat("\nFeature importance plot saved to", 
      file.path(output_dir, "feature_importance_rf_r.png"), "\n")
  
  cat("\nTop 10 Features (Random Forest):\n")
  print(head(rf_importance, 10))
}

# XGBoost feature importance
if("XGBoost" %in% names(models)) {
  xgb_importance <- xgb.importance(feature_names = features, model = models[["XGBoost"]])
  xgb_importance_top <- head(xgb_importance, 20)
  
  png(file.path(output_dir, "feature_importance_xgb_r.png"), 
      width = 1200, height = 800, res = 300)
  xgb.plot.importance(xgb_importance_top, main = "XGBoost - Top 20 Feature Importance")
  dev.off()
  cat("\nFeature importance plot saved to", 
      file.path(output_dir, "feature_importance_xgb_r.png"), "\n")
  
  cat("\nTop 10 Features (XGBoost):\n")
  print(head(xgb_importance, 10))
}

# ============================================================================
# 5. SAVE BEST MODEL
# ============================================================================
cat("\n", rep("=", 80), "\n", sep = "")
cat("SAVING BEST MODEL\n")
cat(rep("=", 80), "\n", sep = "")

# Find best model
best_model_name <- comparison$Model[which.max(comparison$AUC_ROC)]
best_model <- models[[best_model_name]]

# Save model
saveRDS(best_model, file.path(model_dir, "best_model_r.rds"))
cat("\nBest Model:", best_model_name, "\n")
cat("Best AUC-ROC:", max(comparison$AUC_ROC), "\n")
cat("Model saved to", file.path(model_dir, "best_model_r.rds"), "\n")

# Summary
cat("\n", rep("=", 80), "\n", sep = "")
cat("MACHINE LEARNING ANALYSIS SUMMARY\n")
cat(rep("=", 80), "\n", sep = "")
cat("\nBest Model:", best_model_name, "\n")
cat("Best AUC-ROC:", max(comparison$AUC_ROC), "\n")
cat("\nModels trained:", length(models), "\n")
cat("Features used:", length(features), "\n")
cat("\nAnalysis complete!\n")
cat(rep("=", 80), "\n", sep = "")


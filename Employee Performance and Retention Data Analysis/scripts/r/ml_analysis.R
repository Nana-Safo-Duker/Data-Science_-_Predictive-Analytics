# Machine Learning Analysis for Employee Dataset
# Predictive modeling using various algorithms for salary and bonus prediction

# Load required libraries
library(tidyverse)
library(caret)
library(randomForest)
library(xgboost)
library(e1071)
library(corrplot)
library(ggplot2)

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
dir.create("results/models", recursive = TRUE, showWarnings = FALSE)
dir.create("results/plots", recursive = TRUE, showWarnings = FALSE)
dir.create("results/tables", recursive = TRUE, showWarnings = FALSE)

cat("===============================================================================\n")
cat("MACHINE LEARNING ANALYSIS - EMPLOYEE DATASET\n")
cat("===============================================================================\n")

# Load cleaned dataset
df <- read.csv("data/processed/employees_cleaned.csv", stringsAsFactors = FALSE)

# =============================================================================
# 1. DATA PREPROCESSING FOR ML
# =============================================================================
cat("\n1. DATA PREPROCESSING FOR ML\n")
cat("===============================================================================\n")

# Create a copy for ML
df_ml <- df

# Feature engineering
df_ml$Start_Year <- as.numeric(format(as.Date(df_ml$Start_Date), "%Y"))
df_ml$Start_Month <- as.numeric(format(as.Date(df_ml$Start_Date), "%m"))

# Handle missing values
df_ml$Gender[is.na(df_ml$Gender)] <- "Unknown"
df_ml$Team[is.na(df_ml$Team) | df_ml$Team == ""] <- "Unknown"
df_ml$Senior_Management[is.na(df_ml$Senior_Management)] <- FALSE

# Encode categorical variables
df_ml$Gender_encoded <- as.numeric(as.factor(df_ml$Gender))
df_ml$Team_encoded <- as.numeric(as.factor(df_ml$Team))
df_ml$Senior_Management_encoded <- as.numeric(df_ml$Senior_Management)

# Select features for modeling
feature_columns <- c("Gender_encoded", "Team_encoded", "Senior_Management_encoded",
                     "Bonus_pct", "Years_of_Service", "Start_Year", "Start_Month")

# Remove rows with missing target or features
df_ml <- df_ml[complete.cases(df_ml[, c("Salary", feature_columns)]), ]

cat("Dataset shape for ML:", nrow(df_ml), "rows,", ncol(df_ml), "columns\n")
cat("Features used:", paste(feature_columns, collapse = ", "), "\n")

# =============================================================================
# 2. SALARY PREDICTION
# =============================================================================
cat("\n2. SALARY PREDICTION\n")
cat("===============================================================================\n")

# Prepare data for salary prediction
X_salary <- df_ml[, feature_columns]
y_salary <- df_ml$Salary

# Split data
set.seed(42)
train_index <- createDataPartition(y_salary, p = 0.8, list = FALSE)
X_train_salary <- X_salary[train_index, ]
X_test_salary <- X_salary[-train_index, ]
y_train_salary <- y_salary[train_index]
y_test_salary <- y_salary[-train_index]

cat("Training set size:", nrow(X_train_salary), "\n")
cat("Test set size:", nrow(X_test_salary), "\n")

# =============================================================================
# 2.1 LINEAR REGRESSION
# =============================================================================
cat("\n2.1. Linear Regression\n")

lr_model <- lm(y_train_salary ~ ., data = data.frame(X_train_salary, y_train_salary))
y_pred_lr <- predict(lr_model, newdata = data.frame(X_test_salary))

lr_r2 <- cor(y_test_salary, y_pred_lr)^2
lr_rmse <- sqrt(mean((y_test_salary - y_pred_lr)^2))
lr_mae <- mean(abs(y_test_salary - y_pred_lr))

cat("R² Score:", round(lr_r2, 4), "\n")
cat("RMSE: $", round(lr_rmse, 2), "\n", sep = "")
cat("MAE: $", round(lr_mae, 2), "\n", sep = "")

# Cross-validation
lr_cv <- train(x = X_train_salary, y = y_train_salary, method = "lm",
               trControl = trainControl(method = "cv", number = 5))
cat("Cross-validation R²:", round(mean(lr_cv$resample$Rsquared), 4), "\n")

# =============================================================================
# 2.2 RIDGE REGRESSION
# =============================================================================
cat("\n2.2. Ridge Regression\n")

# Scale features for Ridge
preProc <- preProcess(X_train_salary, method = c("center", "scale"))
X_train_salary_scaled <- predict(preProc, X_train_salary)
X_test_salary_scaled <- predict(preProc, X_test_salary)

# Ridge regression with cross-validation
ridge_model <- train(x = X_train_salary_scaled, y = y_train_salary,
                     method = "ridge",
                     trControl = trainControl(method = "cv", number = 5),
                     tuneGrid = expand.grid(lambda = seq(0.1, 10, by = 0.1)))
y_pred_ridge <- predict(ridge_model, newdata = X_test_salary_scaled)

ridge_r2 <- cor(y_test_salary, y_pred_ridge)^2
ridge_rmse <- sqrt(mean((y_test_salary - y_pred_ridge)^2))
ridge_mae <- mean(abs(y_test_salary - y_pred_ridge))

cat("R² Score:", round(ridge_r2, 4), "\n")
cat("RMSE: $", round(ridge_rmse, 2), "\n", sep = "")
cat("MAE: $", round(ridge_mae, 2), "\n", sep = "")
cat("Best lambda:", ridge_model$bestTune$lambda, "\n")

# =============================================================================
# 2.3 RANDOM FOREST
# =============================================================================
cat("\n2.3. Random Forest\n")

set.seed(42)
rf_model <- randomForest(x = X_train_salary, y = y_train_salary,
                         ntree = 100, mtry = sqrt(ncol(X_train_salary)),
                         importance = TRUE)
y_pred_rf <- predict(rf_model, newdata = X_test_salary)

rf_r2 <- cor(y_test_salary, y_pred_rf)^2
rf_rmse <- sqrt(mean((y_test_salary - y_pred_rf)^2))
rf_mae <- mean(abs(y_test_salary - y_pred_rf))

cat("R² Score:", round(rf_r2, 4), "\n")
cat("RMSE: $", round(rf_rmse, 2), "\n", sep = "")
cat("MAE: $", round(rf_mae, 2), "\n", sep = "")

# Feature importance
rf_feature_importance <- data.frame(
  Feature = feature_columns,
  Importance = importance(rf_model)[, 1]
)
rf_feature_importance <- rf_feature_importance[order(-rf_feature_importance$Importance), ]

cat("\nFeature Importance (Random Forest):\n")
print(rf_feature_importance)

# =============================================================================
# 2.4 GRADIENT BOOSTING
# =============================================================================
cat("\n2.4. Gradient Boosting\n")

gb_model <- train(x = X_train_salary, y = y_train_salary,
                  method = "gbm",
                  trControl = trainControl(method = "cv", number = 5),
                  tuneGrid = expand.grid(n.trees = 100, interaction.depth = 3,
                                         shrinkage = 0.1, n.minobsinnode = 10),
                  verbose = FALSE)
y_pred_gb <- predict(gb_model, newdata = X_test_salary)

gb_r2 <- cor(y_test_salary, y_pred_gb)^2
gb_rmse <- sqrt(mean((y_test_salary - y_pred_gb)^2))
gb_mae <- mean(abs(y_test_salary - y_pred_gb))

cat("R² Score:", round(gb_r2, 4), "\n")
cat("RMSE: $", round(gb_rmse, 2), "\n", sep = "")
cat("MAE: $", round(gb_mae, 2), "\n", sep = "")

# Feature importance
gb_feature_importance <- data.frame(
  Feature = feature_columns,
  Importance = varImp(gb_model)$importance[, 1]
)
gb_feature_importance <- gb_feature_importance[order(-gb_feature_importance$Importance), ]

cat("\nFeature Importance (Gradient Boosting):\n")
print(gb_feature_importance)

# =============================================================================
# 2.5 XGBOOST
# =============================================================================
cat("\n2.5. XGBoost\n")

# Prepare data for XGBoost
dtrain <- xgb.DMatrix(data = as.matrix(X_train_salary), label = y_train_salary)
dtest <- xgb.DMatrix(data = as.matrix(X_test_salary), label = y_test_salary)

# Train XGBoost model
xgb_model <- xgb.train(data = dtrain,
                       nrounds = 100,
                       objective = "reg:squarederror",
                       max_depth = 6,
                       eta = 0.1,
                       nthread = 2,
                       verbose = 0)
y_pred_xgb <- predict(xgb_model, newdata = dtest)

xgb_r2 <- cor(y_test_salary, y_pred_xgb)^2
xgb_rmse <- sqrt(mean((y_test_salary - y_pred_xgb)^2))
xgb_mae <- mean(abs(y_test_salary - y_pred_xgb))

cat("R² Score:", round(xgb_r2, 4), "\n")
cat("RMSE: $", round(xgb_rmse, 2), "\n", sep = "")
cat("MAE: $", round(xgb_mae, 2), "\n", sep = "")

# Feature importance
xgb_feature_importance <- xgb.importance(feature_names = feature_columns, model = xgb_model)
xgb_feature_importance <- xgb_feature_importance[order(-xgb_feature_importance$Gain), ]

cat("\nFeature Importance (XGBoost):\n")
print(xgb_feature_importance)

# =============================================================================
# 3. MODEL COMPARISON
# =============================================================================
cat("\n3. MODEL COMPARISON\n")
cat("===============================================================================\n")

# Create comparison DataFrame
model_comparison <- data.frame(
  Model = c("Linear Regression", "Ridge Regression", "Random Forest",
            "Gradient Boosting", "XGBoost"),
  R2_Score = c(lr_r2, ridge_r2, rf_r2, gb_r2, xgb_r2),
  RMSE = c(lr_rmse, ridge_rmse, rf_rmse, gb_rmse, xgb_rmse),
  MAE = c(lr_mae, ridge_mae, rf_mae, gb_mae, xgb_mae)
)
model_comparison <- model_comparison[order(-model_comparison$R2_Score), ]

cat("\nModel Comparison (Salary Prediction):\n")
print(model_comparison)
write.csv(model_comparison, "results/tables/model_comparison_salary.csv", row.names = FALSE)

# Visualize model comparison
png("results/plots/model_comparison_salary.png", width = 2400, height = 800, res = 300)
par(mfrow = c(1, 3))

# R² Score comparison
barplot(model_comparison$R2_Score, names.arg = model_comparison$Model,
        main = "Model Comparison: R² Score", xlab = "Model", ylab = "R² Score",
        col = "steelblue", las = 2)

# RMSE comparison
barplot(model_comparison$RMSE, names.arg = model_comparison$Model,
        main = "Model Comparison: RMSE", xlab = "Model", ylab = "RMSE ($)",
        col = "lightcoral", las = 2)

# MAE comparison
barplot(model_comparison$MAE, names.arg = model_comparison$Model,
        main = "Model Comparison: MAE", xlab = "Model", ylab = "MAE ($)",
        col = "lightgreen", las = 2)
dev.off()

# =============================================================================
# 4. FEATURE IMPORTANCE VISUALIZATION
# =============================================================================
cat("\n4. FEATURE IMPORTANCE VISUALIZATION\n")

# Plot feature importance for tree-based models
png("results/plots/feature_importance.png", width = 2400, height = 2400, res = 300)
par(mfrow = c(2, 2))

# Random Forest
barplot(rf_feature_importance$Importance, names.arg = rf_feature_importance$Feature,
        main = "Feature Importance: Random Forest", xlab = "Feature",
        ylab = "Importance", col = "steelblue", las = 2)

# Gradient Boosting
barplot(gb_feature_importance$Importance, names.arg = gb_feature_importance$Feature,
        main = "Feature Importance: Gradient Boosting", xlab = "Feature",
        ylab = "Importance", col = "lightcoral", las = 2)

# XGBoost
barplot(xgb_feature_importance$Gain, names.arg = xgb_feature_importance$Feature,
        main = "Feature Importance: XGBoost", xlab = "Feature",
        ylab = "Gain", col = "lightgreen", las = 2)

# Combined feature importance
all_features <- unique(c(rf_feature_importance$Feature, gb_feature_importance$Feature,
                        xgb_feature_importance$Feature))
combined_importance <- data.frame(
  Feature = all_features,
  RF = sapply(all_features, function(x) {
    if (x %in% rf_feature_importance$Feature) {
      rf_feature_importance$Importance[rf_feature_importance$Feature == x]
    } else {
      0
    }
  }),
  GB = sapply(all_features, function(x) {
    if (x %in% gb_feature_importance$Feature) {
      gb_feature_importance$Importance[gb_feature_importance$Feature == x]
    } else {
      0
    }
  }),
  XGB = sapply(all_features, function(x) {
    if (x %in% xgb_feature_importance$Feature) {
      xgb_feature_importance$Gain[xgb_feature_importance$Feature == x]
    } else {
      0
    }
  })
)
combined_importance$Average <- rowMeans(combined_importance[, 2:4])
combined_importance <- combined_importance[order(-combined_importance$Average), ]

barplot(combined_importance$Average, names.arg = combined_importance$Feature,
        main = "Average Feature Importance (All Models)", xlab = "Feature",
        ylab = "Average Importance", col = "gold", las = 2)
dev.off()

# =============================================================================
# 5. PREDICTION VISUALIZATIONS
# =============================================================================
cat("\n5. PREDICTION VISUALIZATIONS\n")

# Get best model
best_model_name <- model_comparison$Model[1]
best_models <- list(
  "Linear Regression" = list(model = lr_model, predictions = y_pred_lr),
  "Ridge Regression" = list(model = ridge_model, predictions = y_pred_ridge),
  "Random Forest" = list(model = rf_model, predictions = y_pred_rf),
  "Gradient Boosting" = list(model = gb_model, predictions = y_pred_gb),
  "XGBoost" = list(model = xgb_model, predictions = y_pred_xgb)
)

best_predictions <- best_models[[best_model_name]]$predictions

# Actual vs Predicted
png("results/plots/prediction_visualizations.png", width = 2400, height = 2400, res = 300)
par(mfrow = c(2, 2))

# Scatter plot: Actual vs Predicted
plot(y_test_salary, best_predictions, main = paste("Actual vs Predicted:", best_model_name),
     xlab = "Actual Salary", ylab = "Predicted Salary", pch = 19, alpha = 0.5)
abline(0, 1, col = "red", lwd = 2, lty = 2)
grid()

# Residual plot
residuals <- y_test_salary - best_predictions
plot(best_predictions, residuals, main = paste("Residual Plot:", best_model_name),
     xlab = "Predicted Salary", ylab = "Residuals", pch = 19, alpha = 0.5)
abline(h = 0, col = "red", lwd = 2, lty = 2)
grid()

# Distribution of residuals
hist(residuals, main = paste("Distribution of Residuals:", best_model_name),
     xlab = "Residuals", ylab = "Frequency", col = "steelblue", border = "black")
abline(v = 0, col = "red", lwd = 2, lty = 2)
grid()

# Q-Q plot of residuals
qqnorm(residuals, main = paste("Q-Q Plot of Residuals:", best_model_name))
qqline(residuals, col = "red", lwd = 2)
grid()
dev.off()

# =============================================================================
# 6. SAVE MODELS
# =============================================================================
cat("\n6. SAVE MODELS\n")

# Save best model
saveRDS(best_models[[best_model_name]]$model, file = "results/models/best_model_salary.rds")
saveRDS(preProc, file = "results/models/preprocessor_salary.rds")

cat("Best model (", best_model_name, ") saved to: results/models/best_model_salary.rds\n", sep = "")

# =============================================================================
# 7. BONUS % PREDICTION (OPTIONAL)
# =============================================================================
cat("\n7. BONUS % PREDICTION\n")

# Prepare data for bonus prediction (excluding Bonus_pct from features)
feature_columns_bonus <- feature_columns[feature_columns != "Bonus_pct"]
X_bonus <- df_ml[, feature_columns_bonus]
y_bonus <- df_ml$Bonus_pct

# Remove rows with missing target
complete_cases <- complete.cases(X_bonus, y_bonus)
X_bonus <- X_bonus[complete_cases, ]
y_bonus <- y_bonus[complete_cases]

# Split data
set.seed(42)
train_index_bonus <- createDataPartition(y_bonus, p = 0.8, list = FALSE)
X_train_bonus <- X_bonus[train_index_bonus, ]
X_test_bonus <- X_bonus[-train_index_bonus, ]
y_train_bonus <- y_bonus[train_index_bonus]
y_test_bonus <- y_bonus[-train_index_bonus]

# Train XGBoost for bonus prediction
dtrain_bonus <- xgb.DMatrix(data = as.matrix(X_train_bonus), label = y_train_bonus)
dtest_bonus <- xgb.DMatrix(data = as.matrix(X_test_bonus), label = y_test_bonus)

xgb_bonus_model <- xgb.train(data = dtrain_bonus,
                             nrounds = 100,
                             objective = "reg:squarederror",
                             max_depth = 6,
                             eta = 0.1,
                             nthread = 2,
                             verbose = 0)
y_pred_bonus <- predict(xgb_bonus_model, newdata = dtest_bonus)

bonus_r2 <- cor(y_test_bonus, y_pred_bonus)^2
bonus_rmse <- sqrt(mean((y_test_bonus - y_pred_bonus)^2))
bonus_mae <- mean(abs(y_test_bonus - y_pred_bonus))

cat("R² Score:", round(bonus_r2, 4), "\n")
cat("RMSE:", round(bonus_rmse, 4), "%\n", sep = "")
cat("MAE:", round(bonus_mae, 4), "%\n", sep = "")

# Save bonus model
saveRDS(xgb_bonus_model, file = "results/models/xgb_model_bonus.rds")

cat("\n===============================================================================\n")
cat("MACHINE LEARNING ANALYSIS COMPLETED SUCCESSFULLY!\n")
cat("===============================================================================\n")
cat("\nBest Model for Salary Prediction:", best_model_name, "\n")
cat("R² Score:", round(model_comparison$R2_Score[1], 4), "\n")
cat("RMSE: $", round(model_comparison$RMSE[1], 2), "\n", sep = "")
cat("MAE: $", round(model_comparison$MAE[1], 2), "\n", sep = "")
cat("\nResults saved in:\n")
cat("- Models: results/models/\n")
cat("- Tables: results/tables/\n")
cat("- Plots: results/plots/\n")

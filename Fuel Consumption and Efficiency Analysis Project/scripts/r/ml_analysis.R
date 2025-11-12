# Machine Learning Analysis Script
# Implements various ML algorithms for predicting Fuel Consumption and CO2 Emissions

# Load necessary libraries
library(caret)
library(randomForest)
library(e1071)
library(glmnet)
library(xgboost)

# Suppress warnings
options(warn = -1)

# Load the dataset
data_path <- file.path("..", "..", "data", "FuelConsumption.csv")
df <- read.csv(data_path, stringsAsFactors = FALSE)

# Clean column names
colnames(df) <- trimws(colnames(df))

cat("==================================================\n")
cat("MACHINE LEARNING ANALYSIS\n")
cat("==================================================\n\n")

# Prepare data for modeling
# Encode categorical variables
df$MAKE_encoded <- as.numeric(as.factor(df$MAKE))
df$VEHICLE.CLASS_encoded <- as.numeric(as.factor(df$VEHICLE.CLASS))
df$TRANSMISSION_encoded <- as.numeric(as.factor(df$TRANSMISSION))
df$FUEL_encoded <- as.numeric(as.factor(df$FUEL))

# Select features
features <- c("Year", "ENGINE.SIZE", "CYLINDERS", "MAKE_encoded", 
              "VEHICLE.CLASS_encoded", "TRANSMISSION_encoded", "FUEL_encoded")
X <- df[, features]
y_fuel <- df$FUEL.CONSUMPTION
y_co2 <- df$COEMISSIONS

cat("Features selected:", paste(features, collapse = ", "), "\n")
cat("X shape:", nrow(X), "rows,", ncol(X), "columns\n")
cat("y_fuel shape:", length(y_fuel), "\n")
cat("y_co2 shape:", length(y_co2), "\n\n")

# Split data for fuel consumption
set.seed(42)
trainIndex_fuel <- createDataPartition(y_fuel, p = 0.8, list = FALSE)
X_train_fuel <- X[trainIndex_fuel, ]
X_test_fuel <- X[-trainIndex_fuel, ]
y_fuel_train <- y_fuel[trainIndex_fuel]
y_fuel_test <- y_fuel[-trainIndex_fuel]

# Split data for CO2 emissions
set.seed(42)
trainIndex_co2 <- createDataPartition(y_co2, p = 0.8, list = FALSE)
X_train_co2 <- X[trainIndex_co2, ]
X_test_co2 <- X[-trainIndex_co2, ]
y_co2_train <- y_co2[trainIndex_co2]
y_co2_test <- y_co2[-trainIndex_co2]

cat("Data split completed!\n")
cat("Training set size (fuel):", nrow(X_train_fuel), "\n")
cat("Test set size (fuel):", nrow(X_test_fuel), "\n")
cat("Training set size (CO2):", nrow(X_train_co2), "\n")
cat("Test set size (CO2):", nrow(X_test_co2), "\n\n")

# Train models for Fuel Consumption
cat("--- Fuel Consumption Prediction ---\n\n")

# Linear Regression
lr_fuel <- train(X_train_fuel, y_fuel_train, method = "lm", 
                 trControl = trainControl(method = "cv", number = 5))
y_fuel_pred_lr <- predict(lr_fuel, X_test_fuel)
lr_r2 <- R2(y_fuel_pred_lr, y_fuel_test)
lr_rmse <- RMSE(y_fuel_pred_lr, y_fuel_test)
lr_mae <- MAE(y_fuel_pred_lr, y_fuel_test)

cat("Linear Regression:\n")
cat("  R2 Score:", round(lr_r2, 4), "\n")
cat("  RMSE:", round(lr_rmse, 4), "\n")
cat("  MAE:", round(lr_mae, 4), "\n\n")

# Random Forest
rf_fuel <- randomForest(X_train_fuel, y_fuel_train, ntree = 100, 
                        mtry = sqrt(ncol(X_train_fuel)), random_state = 42)
y_fuel_pred_rf <- predict(rf_fuel, X_test_fuel)
rf_r2 <- R2(y_fuel_pred_rf, y_fuel_test)
rf_rmse <- RMSE(y_fuel_pred_rf, y_fuel_test)
rf_mae <- MAE(y_fuel_pred_rf, y_fuel_test)

cat("Random Forest:\n")
cat("  R2 Score:", round(rf_r2, 4), "\n")
cat("  RMSE:", round(rf_rmse, 4), "\n")
cat("  MAE:", round(rf_mae, 4), "\n\n")

# Feature importance
feature_importance_fuel <- data.frame(
  feature = features,
  importance = importance(rf_fuel)[, 1]
)
feature_importance_fuel <- feature_importance_fuel[order(-feature_importance_fuel$importance), ]
cat("Feature Importance (Fuel Consumption):\n")
print(feature_importance_fuel)

# Train models for CO2 Emissions
cat("\n--- CO2 Emissions Prediction ---\n\n")

# Linear Regression
lr_co2 <- train(X_train_co2, y_co2_train, method = "lm", 
                trControl = trainControl(method = "cv", number = 5))
y_co2_pred_lr <- predict(lr_co2, X_test_co2)
lr_co2_r2 <- R2(y_co2_pred_lr, y_co2_test)
lr_co2_rmse <- RMSE(y_co2_pred_lr, y_co2_test)
lr_co2_mae <- MAE(y_co2_pred_lr, y_co2_test)

cat("Linear Regression:\n")
cat("  R2 Score:", round(lr_co2_r2, 4), "\n")
cat("  RMSE:", round(lr_co2_rmse, 4), "\n")
cat("  MAE:", round(lr_co2_mae, 4), "\n\n")

# Random Forest
rf_co2 <- randomForest(X_train_co2, y_co2_train, ntree = 100, 
                       mtry = sqrt(ncol(X_train_co2)), random_state = 42)
y_co2_pred_rf <- predict(rf_co2, X_test_co2)
rf_co2_r2 <- R2(y_co2_pred_rf, y_co2_test)
rf_co2_rmse <- RMSE(y_co2_pred_rf, y_co2_test)
rf_co2_mae <- MAE(y_co2_pred_rf, y_co2_test)

cat("Random Forest:\n")
cat("  R2 Score:", round(rf_co2_r2, 4), "\n")
cat("  RMSE:", round(rf_co2_rmse, 4), "\n")
cat("  MAE:", round(rf_co2_mae, 4), "\n\n")

# Feature importance
feature_importance_co2 <- data.frame(
  feature = features,
  importance = importance(rf_co2)[, 1]
)
feature_importance_co2 <- feature_importance_co2[order(-feature_importance_co2$importance), ]
cat("Feature Importance (CO2 Emissions):\n")
print(feature_importance_co2)

# Save models
model_dir <- file.path("..", "..", "outputs", "models")
if(!dir.exists(model_dir)) {
  dir.create(model_dir, recursive = TRUE)
}

saveRDS(rf_fuel, file.path(model_dir, "random_forest_fuel.rds"))
saveRDS(rf_co2, file.path(model_dir, "random_forest_co2.rds"))
saveRDS(lr_fuel, file.path(model_dir, "linear_regression_fuel.rds"))
saveRDS(lr_co2, file.path(model_dir, "linear_regression_co2.rds"))

cat("\n✓ Models saved!\n")

cat("\n==================================================\n")
cat("ML ANALYSIS COMPLETE!\n")
cat("==================================================\n")

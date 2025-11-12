# Machine Learning Analysis Script
# Consumer Purchase Prediction Project

# Load necessary libraries
library(dplyr)
library(caret)
library(randomForest)
library(e1071)
library(rpart)
library(rpart.plot)
library(pROC)
library(ROCR)

# Function to find project root by looking for data directory
find_project_root <- function() {
  current_dir <- getwd()
  max_levels <- 10
  project_marker <- file.path("Consumer Purchase Prediction", "Consumer Purchase Prediction", "data", "Advertisement.csv")
  
  for (i in 1:max_levels) {
    if (file.exists(file.path(current_dir, project_marker))) {
      return(current_dir)
    }
    if (file.exists(file.path(current_dir, "data", "Advertisement.csv"))) {
      return(current_dir)
    }
    if (basename(current_dir) == "Consumer Purchase Prediction") {
      if (file.exists(file.path(current_dir, "Consumer Purchase Prediction", "data", "Advertisement.csv"))) {
        return(current_dir)
      }
      if (file.exists(file.path(current_dir, "data", "Advertisement.csv"))) {
        return(current_dir)
      }
    }
    parent_dir <- dirname(current_dir)
    if (parent_dir == current_dir) break
    current_dir <- parent_dir
  }
  return(NULL)
}

# Set working directory to project root
project_root <- find_project_root()
if (!is.null(project_root)) {
  setwd(project_root)
  cat("Working directory set to:", getwd(), "\n")
} else {
  cat("Warning: Could not find project root. Using current directory:", getwd(), "\n")
}

# Load the dataset - try multiple possible paths
data_paths <- c(
  file.path("Consumer Purchase Prediction", "Consumer Purchase Prediction", "data", "Advertisement.csv"),
  file.path("data", "Advertisement.csv"),
  "Advertisement.csv"
)

data_path <- NULL
for (path in data_paths) {
  if (file.exists(path)) {
    data_path <- path
    break
  }
}

if (is.null(data_path)) {
  stop(paste("Cannot find Advertisement.csv. Searched in:\n",
             paste("  -", data_paths, collapse = "\n"),
             "\nCurrent working directory:", getwd()))
}

df <- read.csv(data_path, stringsAsFactors = TRUE)

cat("Dataset loaded successfully from:", data_path, "\n")
cat("MACHINE LEARNING ANALYSIS\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

# Preprocessing
df$Gender <- as.factor(df$Gender)
df$Purchased <- as.factor(df$Purchased)

# Prepare features and target
X <- df[, c("Gender", "Age", "EstimatedSalary")]
y <- df$Purchased

# Split the data
set.seed(42)
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[trainIndex, ]
X_test <- X[-trainIndex, ]
y_train <- y[trainIndex]
y_test <- y[-trainIndex]

cat("Training set size:", nrow(X_train), "\n")
cat("Test set size:", nrow(X_test), "\n\n")

# Create training data frame
train_data <- data.frame(X_train, Purchased = y_train)
test_data <- data.frame(X_test, Purchased = y_test)

# 1. Logistic Regression
cat("1. LOGISTIC REGRESSION\n")
cat(paste(rep("-", 50), collapse=""), "\n")
logistic_model <- train(Purchased ~ ., data = train_data, method = "glm", 
                        family = "binomial", trControl = trainControl(method = "cv", number = 5))
logistic_pred <- predict(logistic_model, test_data)
logistic_pred_proba <- predict(logistic_model, test_data, type = "prob")[, 2]
logistic_cm <- confusionMatrix(logistic_pred, y_test)
cat("Accuracy:", logistic_cm$overall["Accuracy"], "\n")
cat("Precision:", logistic_cm$byClass["Precision"], "\n")
cat("Recall:", logistic_cm$byClass["Recall"], "\n")
cat("F1 Score:", logistic_cm$byClass["F1"], "\n\n")

# 2. Random Forest
cat("2. RANDOM FOREST\n")
cat(paste(rep("-", 50), collapse=""), "\n")
rf_model <- train(Purchased ~ ., data = train_data, method = "rf", 
                  ntree = 100, trControl = trainControl(method = "cv", number = 5))
rf_pred <- predict(rf_model, test_data)
rf_pred_proba <- predict(rf_model, test_data, type = "prob")[, 2]
rf_cm <- confusionMatrix(rf_pred, y_test)
cat("Accuracy:", rf_cm$overall["Accuracy"], "\n")
cat("Precision:", rf_cm$byClass["Precision"], "\n")
cat("Recall:", rf_cm$byClass["Recall"], "\n")
cat("F1 Score:", rf_cm$byClass["F1"], "\n\n")

# 3. SVM
cat("3. SUPPORT VECTOR MACHINE\n")
cat(paste(rep("-", 50), collapse=""), "\n")
svm_model <- train(Purchased ~ ., data = train_data, method = "svmRadial", 
                   trControl = trainControl(method = "cv", number = 5))
svm_pred <- predict(svm_model, test_data)
svm_pred_proba <- predict(svm_model, test_data, type = "prob")[, 2]
svm_cm <- confusionMatrix(svm_pred, y_test)
cat("Accuracy:", svm_cm$overall["Accuracy"], "\n")
cat("Precision:", svm_cm$byClass["Precision"], "\n")
cat("Recall:", svm_cm$byClass["Recall"], "\n")
cat("F1 Score:", svm_cm$byClass["F1"], "\n\n")

# 4. Naive Bayes
cat("4. NAIVE BAYES\n")
cat(paste(rep("-", 50), collapse=""), "\n")
nb_model <- train(Purchased ~ ., data = train_data, method = "nb", 
                  trControl = trainControl(method = "cv", number = 5))
nb_pred <- predict(nb_model, test_data)
nb_pred_proba <- predict(nb_model, test_data, type = "prob")[, 2]
nb_cm <- confusionMatrix(nb_pred, y_test)
cat("Accuracy:", nb_cm$overall["Accuracy"], "\n")
cat("Precision:", nb_cm$byClass["Precision"], "\n")
cat("Recall:", nb_cm$byClass["Recall"], "\n")
cat("F1 Score:", nb_cm$byClass["F1"], "\n\n")

# 5. Decision Tree
cat("5. DECISION TREE\n")
cat(paste(rep("-", 50), collapse=""), "\n")
dt_model <- train(Purchased ~ ., data = train_data, method = "rpart", 
                  trControl = trainControl(method = "cv", number = 5))
dt_pred <- predict(dt_model, test_data)
dt_pred_proba <- predict(dt_model, test_data, type = "prob")[, 2]
dt_cm <- confusionMatrix(dt_pred, y_test)
cat("Accuracy:", dt_cm$overall["Accuracy"], "\n")
cat("Precision:", dt_cm$byClass["Precision"], "\n")
cat("Recall:", dt_cm$byClass["Recall"], "\n")
cat("F1 Score:", dt_cm$byClass["F1"], "\n\n")

# Model Comparison
cat("MODEL COMPARISON\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

results <- data.frame(
  Model = c("Logistic Regression", "Random Forest", "SVM", "Naive Bayes", "Decision Tree"),
  Accuracy = c(logistic_cm$overall["Accuracy"], rf_cm$overall["Accuracy"], 
               svm_cm$overall["Accuracy"], nb_cm$overall["Accuracy"], dt_cm$overall["Accuracy"]),
  Precision = c(logistic_cm$byClass["Precision"], rf_cm$byClass["Precision"], 
                svm_cm$byClass["Precision"], nb_cm$byClass["Precision"], dt_cm$byClass["Precision"]),
  Recall = c(logistic_cm$byClass["Recall"], rf_cm$byClass["Recall"], 
             svm_cm$byClass["Recall"], nb_cm$byClass["Recall"], dt_cm$byClass["Recall"]),
  F1 = c(logistic_cm$byClass["F1"], rf_cm$byClass["F1"], 
         svm_cm$byClass["F1"], nb_cm$byClass["F1"], dt_cm$byClass["F1"])
)

results <- results[order(-results$Accuracy), ]
print(results)

# Find best model
best_model_idx <- which.max(results$Accuracy)
best_model_name <- results$Model[best_model_idx]

cat("\nBEST MODEL:", best_model_name, "\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("Accuracy:", results$Accuracy[best_model_idx], "\n")
cat("Precision:", results$Precision[best_model_idx], "\n")
cat("Recall:", results$Recall[best_model_idx], "\n")
cat("F1 Score:", results$F1[best_model_idx], "\n")

# Get output directory
output_paths <- c(
  file.path("Consumer Purchase Prediction", "Consumer Purchase Prediction", "output"),
  "output"
)

output_dir <- NULL
for (path in output_paths) {
  if (dir.exists(path)) {
    output_dir <- path
    break
  }
}

if (is.null(output_dir)) {
  output_dir <- output_paths[1]
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
}

# ROC Curves
png(file.path(output_dir, "roc_curves_r.png"), width = 1000, height = 800, res = 300)
roc_logistic <- roc(y_test, logistic_pred_proba)
roc_rf <- roc(y_test, rf_pred_proba)
roc_svm <- roc(y_test, svm_pred_proba)
roc_nb <- roc(y_test, nb_pred_proba)
roc_dt <- roc(y_test, dt_pred_proba)

plot(roc_logistic, col = "blue", main = "ROC Curves for All Models")
lines(roc_rf, col = "red")
lines(roc_svm, col = "green")
lines(roc_nb, col = "purple")
lines(roc_dt, col = "orange")
legend("bottomright", 
       legend = c(paste("Logistic Regression (AUC =", round(auc(roc_logistic), 3), ")"),
                  paste("Random Forest (AUC =", round(auc(roc_rf), 3), ")"),
                  paste("SVM (AUC =", round(auc(roc_svm), 3), ")"),
                  paste("Naive Bayes (AUC =", round(auc(roc_nb), 3), ")"),
                  paste("Decision Tree (AUC =", round(auc(roc_dt), 3), ")")),
       col = c("blue", "red", "green", "purple", "orange"), lty = 1)
dev.off()

cat("\n", paste(rep("=", 50), collapse=""), "\n")
cat("ML ANALYSIS COMPLETED SUCCESSFULLY!\n")
cat(paste(rep("=", 50), collapse=""), "\n")

# Load necessary libraries
library(tidyverse)
library(caret)
library(randomForest)
library(corrplot)

# Load the diamonds dataset
diamonds_df <- read.csv("")

# Correlation matrix
correlation_matrix <- cor(diamonds_df[, c("carat", "cut", "color", "clarity", "depth", "table", "price")])
print("Correlation Matrix:")
print(correlation_matrix)

# Visualize the correlation matrix
corrplot(correlation_matrix, method = "color")

# Feature engineering (select relevant variables)
diamonds_selected <- diamonds_df[, c("carat", "cut", "color", "clarity", "depth", "table", "price")]

# Split the dataset into training and testing sets (70/30 ratio)
set.seed(42)
train_index <- createDataPartition(diamonds_selected$price, p = 0.7, list = FALSE)
train_data <- diamonds_selected[train_index, ]
test_data <- diamonds_selected[-train_index, ]

# Linear Regression
linear_model <- lm(price ~ ., data = train_data)

# Evaluate Linear Regression model
linear_predictions_train <- predict(linear_model, newdata = train_data)
linear_mse_train <- mean((train_data$price - linear_predictions_train)^2)
cat("\nLinear Regression - Mean Squared Error (Train):", linear_mse_train, "\n")

linear_predictions_test <- predict(linear_model, newdata = test_data)
linear_mse_test <- mean((test_data$price - linear_predictions_test)^2)
cat("Linear Regression - Mean Squared Error (Test):", linear_mse_test, "\n")

# Calculate Mean Absolute Error for both train and test sets for linear regression
linear_mae_train <- mean(abs(train_data$price - linear_predictions_train))
cat("\nLinear Regression - Mean Absolute Error (Train):", linear_mae_train, "\n")

linear_mae_test <- mean(abs(test_data$price - linear_predictions_test))
cat("Linear Regression - Mean Absolute Error (Test):", linear_mae_test, "\n")

# Calculate R-squared for both train and test sets linear regression
linear_r_squared_train <- summary(linear_model)$r.squared
cat("\nLinear Regression - R-squared (Train):", linear_r_squared_train, "\n")

linear_r_squared_test <- cor(test_data$price, linear_predictions_test)^2
cat("Linear Regression - R-squared (Test):", linear_r_squared_test, "\n")

# Decision Tree Regression
tree_model <- train(price ~ ., data = train_data, method = "rpart")

# Evaluate Decision Tree Regression model
tree_predictions_train <- predict(tree_model, newdata = train_data)
tree_mse_train <- mean((train_data$price - tree_predictions_train)^2)
cat("\nDecision Tree Regression - Mean Squared Error (Train):", tree_mse_train, "\n")

tree_predictions_test <- predict(tree_model, newdata = test_data)
tree_mse_test <- mean((test_data$price - tree_predictions_test)^2)
cat("Decision Tree Regression - Mean Squared Error (Test):", tree_mse_test, "\n")

# Random Forest Regression
rf_model <- randomForest(price ~ ., data = train_data)

# Evaluate Random Forest Regression model
rf_predictions_train <- predict(rf_model, newdata = train_data)
rf_mse_train <- mean((train_data$price - rf_predictions_train)^2)
cat("\nRandom Forest Regression - Mean Squared Error (Train):", rf_mse_train, "\n")

rf_predictions_test <- predict(rf_model, newdata = test_data)
rf_mse_test <- mean((test_data$price - rf_predictions_test)^2)
cat("Random Forest Regression - Mean Squared Error (Test):", rf_mse_test, "\n")

# Testing with an instance
instance <- data.frame(carat = 0.3, cut = "Ideal", color = "E", clarity = "SI2", depth = 60, table = 55)
linear_pred <- predict(linear_model, newdata = instance)
tree_pred <- predict(tree_model, newdata = instance)
rf_pred <- predict(rf_model, newdata = instance)

cat("\nPredicted Price with Linear Regression:", linear_pred, "\n")
cat("Predicted Price with Decision Tree Regression:", tree_pred, "\n")
cat("Predicted Price with Random Forest Regression:", rf_pred, "\n")

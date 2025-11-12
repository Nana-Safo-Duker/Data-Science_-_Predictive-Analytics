# Univariate, Bivariate, and Multivariate Analysis Script

# Load necessary libraries
library(dplyr)
library(ggplot2)
library(corrplot)
library(GGally)
library(gridExtra)

# Suppress warnings
options(warn = -1)

# Load the dataset
data_path <- file.path("..", "..", "data", "FuelConsumption.csv")
df <- read.csv(data_path, stringsAsFactors = FALSE)

# Clean column names
colnames(df) <- trimws(colnames(df))

cat("==================================================\n")
cat("UNIVARIATE, BIVARIATE, AND MULTIVARIATE ANALYSIS\n")
cat("==================================================\n\n")

# Create output directory
output_dir <- file.path("..", "..", "outputs", "figures")
if(!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

numerical_cols <- c("ENGINE.SIZE", "CYLINDERS", "FUEL.CONSUMPTION", "COEMISSIONS")

# Univariate Analysis
cat("==================================================\n")
cat("UNIVARIATE ANALYSIS\n")
cat("==================================================\n\n")

# Univariate statistics
cat("Univariate Statistics:\n\n")
for(col in numerical_cols) {
  cat(col, ":\n")
  cat("  Mean:", round(mean(df[[col]], na.rm = TRUE), 2), "\n")
  cat("  Median:", round(median(df[[col]], na.rm = TRUE), 2), "\n")
  cat("  Std:", round(sd(df[[col]], na.rm = TRUE), 2), "\n")
  cat("  Variance:", round(var(df[[col]], na.rm = TRUE), 2), "\n")
  cat("  Skewness:", round(e1071::skewness(df[[col]], na.rm = TRUE), 2), "\n")
  cat("  Kurtosis:", round(e1071::kurtosis(df[[col]], na.rm = TRUE), 2), "\n")
  Q1 <- quantile(df[[col]], 0.25, na.rm = TRUE)
  Q3 <- quantile(df[[col]], 0.75, na.rm = TRUE)
  cat("  Q1:", round(Q1, 2), ", Q3:", round(Q3, 2), "\n")
  cat("  IQR:", round(Q3 - Q1, 2), "\n\n")
}

# Univariate visualizations
p1 <- ggplot(df, aes(x = ENGINE.SIZE)) + 
  geom_histogram(bins = 30, fill = "steelblue", alpha = 0.7, color = "black") +
  geom_density(aes(y = ..density.. * (max(df$ENGINE.SIZE, na.rm = TRUE) - min(df$ENGINE.SIZE, na.rm = TRUE)) * 30), 
               color = "red", size = 1) +
  labs(title = "Distribution of Engine Size", x = "Engine Size (L)", y = "Density") +
  theme_minimal()

p2 <- ggplot(df, aes(x = CYLINDERS)) + 
  geom_histogram(bins = 30, fill = "coral", alpha = 0.7, color = "black") +
  geom_density(aes(y = ..density.. * (max(df$CYLINDERS, na.rm = TRUE) - min(df$CYLINDERS, na.rm = TRUE)) * 30), 
               color = "red", size = 1) +
  labs(title = "Distribution of Cylinders", x = "Number of Cylinders", y = "Density") +
  theme_minimal()

p3 <- ggplot(df, aes(x = FUEL.CONSUMPTION)) + 
  geom_histogram(bins = 30, fill = "green", alpha = 0.7, color = "black") +
  geom_density(aes(y = ..density.. * (max(df$FUEL.CONSUMPTION, na.rm = TRUE) - min(df$FUEL.CONSUMPTION, na.rm = TRUE)) * 30), 
               color = "red", size = 1) +
  labs(title = "Distribution of Fuel Consumption", x = "Fuel Consumption (L/100km)", y = "Density") +
  theme_minimal()

p4 <- ggplot(df, aes(x = COEMISSIONS)) + 
  geom_histogram(bins = 30, fill = "purple", alpha = 0.7, color = "black") +
  geom_density(aes(y = ..density.. * (max(df$COEMISSIONS, na.rm = TRUE) - min(df$COEMISSIONS, na.rm = TRUE)) * 30), 
               color = "red", size = 1) +
  labs(title = "Distribution of CO2 Emissions", x = "CO2 Emissions (g/km)", y = "Density") +
  theme_minimal()

ggsave(file.path(output_dir, "univariate_analysis_r.png"), 
       grid.arrange(p1, p2, p3, p4, ncol = 2), 
       width = 16, height = 12, dpi = 300)
cat("✓ Univariate analysis plots saved!\n")

# Bivariate Analysis
cat("\n==================================================\n")
cat("BIVARIATE ANALYSIS\n")
cat("==================================================\n\n")

# Correlation coefficients
cat("Bivariate Correlation Analysis:\n\n")
target <- "COEMISSIONS"
for(col in c("ENGINE.SIZE", "CYLINDERS", "FUEL.CONSUMPTION")) {
  corr <- cor(df[[col]], df[[target]], use = "complete.obs")
  cat(col, " vs ", target, ": r =", round(corr, 4), "\n")
}

cat("\nFuel Consumption vs Other Variables:\n")
for(col in c("ENGINE.SIZE", "CYLINDERS", "COEMISSIONS")) {
  corr <- cor(df$FUEL.CONSUMPTION, df[[col]], use = "complete.obs")
  cat("  FUEL.CONSUMPTION vs ", col, ": r =", round(corr, 4), "\n")
}

# Scatter plots
p1 <- ggplot(df, aes(x = ENGINE.SIZE, y = FUEL.CONSUMPTION)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  labs(title = "Engine Size vs Fuel Consumption", 
       x = "Engine Size (L)", y = "Fuel Consumption (L/100km)") +
  theme_minimal()

p2 <- ggplot(df, aes(x = CYLINDERS, y = FUEL.CONSUMPTION)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  labs(title = "Cylinders vs Fuel Consumption", 
       x = "Number of Cylinders", y = "Fuel Consumption (L/100km)") +
  theme_minimal()

p3 <- ggplot(df, aes(x = FUEL.CONSUMPTION, y = COEMISSIONS)) +
  geom_point(alpha = 0.5, color = "green") +
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  labs(title = "Fuel Consumption vs CO2 Emissions", 
       x = "Fuel Consumption (L/100km)", y = "CO2 Emissions (g/km)") +
  theme_minimal()

p4 <- ggplot(df, aes(x = ENGINE.SIZE, y = COEMISSIONS)) +
  geom_point(alpha = 0.5, color = "orange") +
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  labs(title = "Engine Size vs CO2 Emissions", 
       x = "Engine Size (L)", y = "CO2 Emissions (g/km)") +
  theme_minimal()

ggsave(file.path(output_dir, "bivariate_analysis_r.png"), 
       grid.arrange(p1, p2, p3, p4, ncol = 2), 
       width = 16, height = 12, dpi = 300)
cat("✓ Bivariate analysis plots saved!\n")

# Multivariate Analysis
cat("\n==================================================\n")
cat("MULTIVARIATE ANALYSIS\n")
cat("==================================================\n\n")

# Pair plot
numerical_data <- df[, numerical_cols]
png(file.path(output_dir, "multivariate_pairplot_r.png"), 
    width = 2000, height = 2000, res = 300)
pairs(numerical_data, pch = 19, cex = 0.5, 
      lower.panel = function(x, y, ...) {
        points(x, y, ...)
        abline(lm(y ~ x), col = "red")
      },
      upper.panel = function(x, y, ...) {
        cor_val <- cor(x, y, use = "complete.obs")
        text(mean(x, na.rm = TRUE), mean(y, na.rm = TRUE), 
             round(cor_val, 2), cex = 1.5)
      })
dev.off()
cat("✓ Multivariate pair plot saved!\n")

# Correlation heatmap
correlation_matrix <- cor(numerical_data, use = "complete.obs")
png(file.path(output_dir, "multivariate_correlation_r.png"), 
    width = 1000, height = 1000, res = 300)
corrplot(correlation_matrix, method = "color", type = "upper", 
         order = "hclust", tl.cex = 0.8, tl.col = "black",
         addCoef.col = "black", number.cex = 0.7,
         title = "Multivariate Correlation Matrix", mar = c(0,0,1,0))
dev.off()
cat("✓ Multivariate correlation matrix saved!\n")

# Multivariate analysis by groups
top_classes <- df %>%
  count(VEHICLE.CLASS, sort = TRUE) %>%
  head(5) %>%
  pull(VEHICLE.CLASS)

df_filtered <- df[df$VEHICLE.CLASS %in% top_classes, ]

p1 <- ggplot(df, aes(x = factor(CYLINDERS), y = FUEL.CONSUMPTION, fill = FUEL)) +
  geom_boxplot() +
  labs(title = "Fuel Consumption by Cylinders and Fuel Type", 
       x = "Number of Cylinders", y = "Fuel Consumption (L/100km)") +
  theme_minimal() +
  theme(legend.position = "bottom")

p2 <- ggplot(df_filtered, aes(x = VEHICLE.CLASS, y = COEMISSIONS)) +
  geom_boxplot(fill = "coral", alpha = 0.7) +
  labs(title = "CO2 Emissions by Vehicle Class (Top 5)", 
       x = "Vehicle Class", y = "CO2 Emissions (g/km)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(file.path(output_dir, "multivariate_grouped_r.png"), 
       grid.arrange(p1, p2, ncol = 2), 
       width = 18, height = 8, dpi = 300)
cat("✓ Multivariate grouped analysis plots saved!\n")

cat("\n==================================================\n")
cat("ANALYSIS COMPLETE!\n")
cat("==================================================\n")

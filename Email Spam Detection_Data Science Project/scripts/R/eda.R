# Exploratory Data Analysis (EDA) Script for Email Spam Detection
# R Script

# Load required libraries
library(tidyverse)
library(ggplot2)
library(wordcloud)
library(tm)
library(stringr)
library(RColorBrewer)

# Set working directory (adjust as needed)
# setwd("path/to/project")

# Create output directory
dir.create("../../output/figures", recursive = TRUE, showWarnings = FALSE)

# Load data
df <- read.csv("../../data/emails_spam_clean.csv", stringsAsFactors = FALSE)

# Basic information
cat("Dataset Shape:", dim(df), "\n")
cat("Columns:", names(df), "\n")
cat("\nFirst few rows:\n")
print(head(df))

# Dataset info
cat("\nDataset Info:\n")
str(df)
summary(df)

# Missing values
cat("\nMissing Values:\n")
print(colSums(is.na(df)))

# Target variable analysis
cat("\nSpam Distribution:\n")
print(table(df$spam))
cat("\nSpam Percentage:\n")
print(prop.table(table(df$spam)) * 100)

# Visualize target distribution
png("../../output/figures/target_distribution_R.png", width = 1200, height = 600, res = 300)
par(mfrow = c(1, 2))

# Bar plot
barplot(table(df$spam), main = "Spam vs Ham Distribution (Count)",
        xlab = "Spam (1) vs Ham (0)", ylab = "Count",
        col = c("skyblue", "salmon"), names.arg = c("Ham", "Spam"))

# Pie chart
pie(table(df$spam), main = "Spam vs Ham Distribution (Percentage)",
    labels = paste0(c("Ham", "Spam"), "\n", round(prop.table(table(df$spam)) * 100, 1), "%"),
    col = c("skyblue", "salmon"))

dev.off()

# Text statistics
df$text_length <- nchar(df$text)
df$word_count <- str_count(df$text, "\\S+")
df$sentence_count <- str_count(df$text, "[.!?]+")
df$avg_word_length <- sapply(strsplit(df$text, "\\s+"), function(x) {
  if(length(x) > 0) mean(nchar(x)) else 0
})

cat("\nText Statistics:\n")
print(summary(df[c("text_length", "word_count", "sentence_count", "avg_word_length")]))

cat("\nText Statistics by Class:\n")
print(df %>% 
  group_by(spam) %>% 
  summarise(
    mean_length = mean(text_length),
    mean_words = mean(word_count),
    mean_sentences = mean(sentence_count),
    mean_avg_word = mean(avg_word_length),
    .groups = 'drop'
  ))

# Visualizations
png("../../output/figures/text_statistics_R.png", width = 1600, height = 1200, res = 300)
par(mfrow = c(2, 2))

# Text length distribution
hist(df$text_length[df$spam == 0], breaks = 50, col = rgb(0.5, 0.7, 1, 0.5),
     main = "Text Length Distribution", xlab = "Text Length (characters)",
     ylab = "Frequency")
hist(df$text_length[df$spam == 1], breaks = 50, col = rgb(1, 0.5, 0.5, 0.5), add = TRUE)
legend("topright", legend = c("Ham", "Spam"), fill = c(rgb(0.5, 0.7, 1, 0.5),
                                                       rgb(1, 0.5, 0.5, 0.5)))

# Word count distribution
hist(df$word_count[df$spam == 0], breaks = 50, col = rgb(0.5, 0.7, 1, 0.5),
     main = "Word Count Distribution", xlab = "Word Count", ylab = "Frequency")
hist(df$word_count[df$spam == 1], breaks = 50, col = rgb(1, 0.5, 0.5, 0.5), add = TRUE)

# Box plots
boxplot(text_length ~ spam, data = df, main = "Text Length by Spam/Ham",
        xlab = "Spam (1) vs Ham (0)", ylab = "Text Length",
        col = c("skyblue", "salmon"), names = c("Ham", "Spam"))

boxplot(word_count ~ spam, data = df, main = "Word Count by Spam/Ham",
        xlab = "Spam (1) vs Ham (0)", ylab = "Word Count",
        col = c("skyblue", "salmon"), names = c("Ham", "Spam"))

dev.off()

# Text cleaning function
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

# Word frequency analysis
spam_words <- unlist(strsplit(paste(df$cleaned_text[df$spam == 1], collapse = " "), "\\s+"))
ham_words <- unlist(strsplit(paste(df$cleaned_text[df$spam == 0], collapse = " "), "\\s+"))

spam_word_freq <- table(spam_words)
ham_word_freq <- table(ham_words)

top_spam_words <- head(sort(spam_word_freq, decreasing = TRUE), 20)
top_ham_words <- head(sort(ham_word_freq, decreasing = TRUE), 20)

cat("\nTop 20 Spam Words:\n")
print(top_spam_words)

cat("\nTop 20 Ham Words:\n")
print(top_ham_words)

# Word clouds
png("../../output/figures/wordclouds_R.png", width = 1600, height = 800, res = 300)
par(mfrow = c(1, 2))

wordcloud(names(spam_word_freq), spam_word_freq, max.words = 100,
          colors = brewer.pal(8, "Dark2"), main = "Spam Emails")

wordcloud(names(ham_word_freq), ham_word_freq, max.words = 100,
          colors = brewer.pal(8, "Set2"), main = "Ham Emails")

dev.off()

# Top words bar plot
png("../../output/figures/top_words_R.png", width = 1600, height = 800, res = 300)
par(mfrow = c(1, 2))

barplot(rev(top_spam_words), horiz = TRUE, las = 1, main = "Top 20 Words in Spam Emails",
        xlab = "Frequency", col = "salmon")

barplot(rev(top_ham_words), horiz = TRUE, las = 1, main = "Top 20 Words in Ham Emails",
        xlab = "Frequency", col = "skyblue")

dev.off()

# Character analysis
df$uppercase_count <- str_count(df$text, "[A-Z]")
df$digit_count <- str_count(df$text, "\\d")
df$special_char_count <- str_count(df$text, "[^a-zA-Z0-9\\s]")
df$exclamation_count <- str_count(df$text, "!")
df$question_count <- str_count(df$text, "\\?")

cat("\nCharacter Statistics by Class:\n")
print(df %>% 
  group_by(spam) %>% 
  summarise(
    mean_uppercase = mean(uppercase_count),
    mean_digit = mean(digit_count),
    mean_special = mean(special_char_count),
    mean_exclamation = mean(exclamation_count),
    mean_question = mean(question_count),
    .groups = 'drop'
  ))

# Save processed data
write.csv(df, "../../data/emails_spam_processed_R.csv", row.names = FALSE)

cat("\nEDA Complete! Processed data saved.\n")



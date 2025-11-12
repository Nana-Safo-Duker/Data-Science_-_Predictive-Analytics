"""
Exploratory Data Analysis (EDA) Script for Email Spam Detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
from wordcloud import WordCloud
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Create output directory
os.makedirs('../../output/figures', exist_ok=True)

def load_data():
    """Load the cleaned dataset"""
    df = pd.read_csv('../../data/emails_spam_clean.csv')
    print(f"Dataset Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df

def basic_info(df):
    """Display basic dataset information"""
    print("\n" + "="*50)
    print("Dataset Info:")
    print(df.info())
    print("\n" + "="*50)
    print("Dataset Description:")
    print(df.describe())
    
    # Missing values
    missing_values = df.isnull().sum()
    print("\nMissing Values:")
    print(missing_values)
    
    if missing_values.sum() > 0:
        plt.figure(figsize=(8, 4))
        missing_values.plot(kind='bar')
        plt.title('Missing Values by Column')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('../../output/figures/missing_values.png')
        plt.close()

def target_analysis(df):
    """Analyze target variable distribution"""
    print("\n" + "="*50)
    print("Target Variable Analysis:")
    print("Spam Distribution:")
    print(df['spam'].value_counts())
    print("\nSpam Percentage:")
    print(df['spam'].value_counts(normalize=True) * 100)
    
    # Visualize target distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    df['spam'].value_counts().plot(kind='bar', ax=axes[0], color=['skyblue', 'salmon'])
    axes[0].set_title('Spam vs Ham Distribution (Count)')
    axes[0].set_xlabel('Spam (1) vs Ham (0)')
    axes[0].set_ylabel('Count')
    axes[0].set_xticklabels(['Ham', 'Spam'], rotation=0)
    
    # Pie chart
    df['spam'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%', 
                                    colors=['skyblue', 'salmon'], startangle=90)
    axes[1].set_title('Spam vs Ham Distribution (Percentage)')
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    plt.savefig('../../output/figures/target_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def text_statistics(df):
    """Calculate and visualize text statistics"""
    # Calculate text statistics
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    df['sentence_count'] = df['text'].str.count(r'[.!?]+')
    df['avg_word_length'] = df['text'].str.split().apply(
        lambda x: np.mean([len(word) for word in x]) if x else 0
    )
    
    print("\n" + "="*50)
    print("Text Statistics:")
    print(df[['text_length', 'word_count', 'sentence_count', 'avg_word_length']].describe())
    
    print("\nText Statistics by Class:")
    print(df.groupby('spam')[['text_length', 'word_count', 'sentence_count', 'avg_word_length']].describe())
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Text length distribution
    axes[0, 0].hist(df[df['spam']==0]['text_length'], bins=50, alpha=0.7, label='Ham', color='skyblue')
    axes[0, 0].hist(df[df['spam']==1]['text_length'], bins=50, alpha=0.7, label='Spam', color='salmon')
    axes[0, 0].set_title('Text Length Distribution')
    axes[0, 0].set_xlabel('Text Length (characters)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    # Word count distribution
    axes[0, 1].hist(df[df['spam']==0]['word_count'], bins=50, alpha=0.7, label='Ham', color='skyblue')
    axes[0, 1].hist(df[df['spam']==1]['word_count'], bins=50, alpha=0.7, label='Spam', color='salmon')
    axes[0, 1].set_title('Word Count Distribution')
    axes[0, 1].set_xlabel('Word Count')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # Box plot for text length
    df.boxplot(column='text_length', by='spam', ax=axes[1, 0])
    axes[1, 0].set_title('Text Length by Spam/Ham')
    axes[1, 0].set_xlabel('Spam (1) vs Ham (0)')
    axes[1, 0].set_ylabel('Text Length')
    
    # Box plot for word count
    df.boxplot(column='word_count', by='spam', ax=axes[1, 1])
    axes[1, 1].set_title('Word Count by Spam/Ham')
    axes[1, 1].set_xlabel('Spam (1) vs Ham (0)')
    axes[1, 1].set_ylabel('Word Count')
    
    plt.tight_layout()
    plt.savefig('../../output/figures/text_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return df

def clean_text(text):
    """Clean and preprocess text"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def word_frequency_analysis(df):
    """Analyze word frequencies"""
    # Apply cleaning
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Get all words from spam and ham emails
    spam_words = ' '.join(df[df['spam']==1]['cleaned_text']).split()
    ham_words = ' '.join(df[df['spam']==0]['cleaned_text']).split()
    
    # Count word frequencies
    spam_word_freq = Counter(spam_words)
    ham_word_freq = Counter(ham_words)
    
    # Get top words
    top_spam_words = spam_word_freq.most_common(20)
    top_ham_words = ham_word_freq.most_common(20)
    
    print("\n" + "="*50)
    print("Top 20 Spam Words:")
    for word, count in top_spam_words:
        print(f"{word}: {count}")
    
    print("\nTop 20 Ham Words:")
    for word, count in top_ham_words:
        print(f"{word}: {count}")
    
    # Create word clouds
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Spam word cloud
    spam_text = ' '.join(df[df['spam']==1]['cleaned_text'])
    spam_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(spam_text)
    axes[0].imshow(spam_wordcloud, interpolation='bilinear')
    axes[0].set_title('Word Cloud - Spam Emails', fontsize=16)
    axes[0].axis('off')
    
    # Ham word cloud
    ham_text = ' '.join(df[df['spam']==0]['cleaned_text'])
    ham_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(ham_text)
    axes[1].imshow(ham_wordcloud, interpolation='bilinear')
    axes[1].set_title('Word Cloud - Ham Emails', fontsize=16)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('../../output/figures/wordclouds.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot top words
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Top spam words
    words, counts = zip(*top_spam_words)
    axes[0].barh(range(len(words)), counts, color='salmon')
    axes[0].set_yticks(range(len(words)))
    axes[0].set_yticklabels(words)
    axes[0].set_title('Top 20 Words in Spam Emails')
    axes[0].set_xlabel('Frequency')
    axes[0].invert_yaxis()
    
    # Top ham words
    words, counts = zip(*top_ham_words)
    axes[1].barh(range(len(words)), counts, color='skyblue')
    axes[1].set_yticks(range(len(words)))
    axes[1].set_yticklabels(words)
    axes[1].set_title('Top 20 Words in Ham Emails')
    axes[1].set_xlabel('Frequency')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('../../output/figures/top_words.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return df

def character_analysis(df):
    """Analyze character patterns"""
    df['uppercase_count'] = df['text'].str.findall(r'[A-Z]').str.len()
    df['digit_count'] = df['text'].str.findall(r'\d').str.len()
    df['special_char_count'] = df['text'].str.findall(r'[^a-zA-Z0-9\s]').str.len()
    df['exclamation_count'] = df['text'].str.count('!')
    df['question_count'] = df['text'].str.count('?')
    
    print("\n" + "="*50)
    print("Character Statistics by Class:")
    print(df.groupby('spam')[['uppercase_count', 'digit_count', 'special_char_count', 
                              'exclamation_count', 'question_count']].mean())
    
    return df

def main():
    """Main function to run EDA"""
    print("="*50)
    print("Exploratory Data Analysis - Email Spam Detection")
    print("="*50)
    
    # Load data
    df = load_data()
    
    # Basic information
    basic_info(df)
    
    # Target analysis
    target_analysis(df)
    
    # Text statistics
    df = text_statistics(df)
    
    # Word frequency analysis
    df = word_frequency_analysis(df)
    
    # Character analysis
    df = character_analysis(df)
    
    # Save processed data
    df.to_csv('../../data/emails_spam_processed.csv', index=False)
    print("\n" + "="*50)
    print("EDA Complete! Processed data saved.")
    print("="*50)

if __name__ == "__main__":
    main()



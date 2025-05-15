import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
from collections import Counter
import re
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Function to clean text
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

# Function to load and preprocess data
def load_data():
    print("Loading CSV file...")
    df = pd.read_csv('soft_labeled_full_dataset.csv')
    
    # Clean text data
    df['cleaned_text'] = df['post_text'].apply(clean_text)
    
    # Add text length features
    df['text_length'] = df['post_text'].str.len()
    df['word_count'] = df['post_text'].str.split().str.len()
    
    # Add sentiment analysis
    df['sentiment'] = df['post_text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    
    return df

# Function for basic statistics
def basic_statistics(df):
    print("\n=== Basic Statistics ===")
    print(f"Total number of records: {len(df)}")
    
    # Display basic info about the dataset
    print("\nDataset Info:")
    print(df.info())
    
    # Display basic statistics for numeric columns
    print("\nNumeric Columns Statistics:")
    print(df.describe())
    
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Display value counts for categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\nValue counts for {col}:")
        print(df[col].value_counts())

# Function for label distribution analysis
def analyze_label_distribution(df):
    print("\n=== Label Distribution Analysis ===")
    
    # Bar Chart
    plt.figure(figsize=(10, 6))
    label_counts = df['final_label'].value_counts()
    sns.barplot(x=label_counts.index, y=label_counts.values, palette='Set2')
    plt.title('Label Distribution (Bar Chart)')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('label_distribution_bar.png')
    plt.close()
    
    # Pie Chart
    plt.figure(figsize=(8, 8))
    plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', colors=sns.color_palette('Set2'))
    plt.title('Label Distribution (Pie Chart)')
    plt.tight_layout()
    plt.savefig('label_distribution_pie.png')
    plt.close()

    # Print the actual counts and percentages
    total = len(df)
    print("\nLabel Distribution:")
    for label, count in label_counts.items():
        percentage = (count / total) * 100
        print(f"{label}: {count:,} instances ({percentage:.2f}%)")
    print(f"\nTotal instances: {total:,}")

# Function for text length analysis
def analyze_text_length(df):
    print("\n=== Text Length Analysis ===")
    
    # Box Plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='final_label', y='text_length', data=df, palette='Set2')
    plt.title('Text Length Distribution by Label')
    plt.xlabel('Label')
    plt.ylabel('Text Length (characters)')
    plt.tight_layout()
    plt.savefig('text_length_boxplot.png')
    plt.close()

# Function to generate word clouds
def generate_wordclouds(df):
    print("\n=== Word Cloud Generation ===")
    
    # Generate word cloud for each label
    for label in df['final_label'].unique():
        text = ' '.join(df[df['final_label'] == label]['cleaned_text'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(16, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for {label} Posts')
        plt.savefig(f'wordcloud_{label}.png')
        plt.close()

# Function to analyze annotator agreement
def analyze_annotator_agreement(df):
    print("\n=== Annotator Agreement Analysis ===")
    
    # Create agreement matrix
    labels = ['normal', 'offensive', 'hatespeech']
    agreement_matrix = np.zeros((3, 3))
    
    for i, label1 in enumerate(labels):
        for j, label2 in enumerate(labels):
            agreement_matrix[i, j] = len(df[
                (df['annotator_1_label'] == label1) & 
                (df['annotator_2_label'] == label2)
            ])
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(agreement_matrix, 
                annot=True, 
                fmt='.0f',
                xticklabels=labels,
                yticklabels=labels,
                cmap='YlOrRd')
    plt.title('Annotator Agreement Heatmap')
    plt.xlabel('Annotator 2 Label')
    plt.ylabel('Annotator 1 Label')
    plt.tight_layout()
    plt.savefig('annotator_agreement_heatmap.png')
    plt.close()
    
    # Calculate agreement statistics
    total = len(df)
    perfect_agreement = len(df[
        (df['annotator_1_label'] == df['annotator_2_label']) & 
        (df['annotator_2_label'] == df['annotator_3_label'])
    ])
    print(f"\nPerfect agreement (all 3 annotators): {perfect_agreement/total*100:.2f}%")
    
    two_agreement = len(df[
        (df['annotator_1_label'] == df['annotator_2_label']) | 
        (df['annotator_2_label'] == df['annotator_3_label']) |
        (df['annotator_1_label'] == df['annotator_3_label'])
    ])
    print(f"Two annotators agree: {two_agreement/total*100:.2f}%")

# Function to plot sentiment distribution
def plot_sentiment_distribution(df):
    print("\n=== Sentiment Analysis ===")
    
    # Histogram for sentiment polarity
    plt.figure(figsize=(10, 6))
    sns.histplot(df['sentiment'], bins=30, kde=True, color='blue')
    plt.title('Sentiment Polarity Distribution')
    plt.xlabel('Sentiment Polarity')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('sentiment_distribution_histogram.png')
    plt.close()
    
    # Boxplot by labels
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='final_label', y='sentiment', data=df, palette='Set2')
    plt.title('Sentiment Polarity by Label')
    plt.xlabel('Label')
    plt.ylabel('Sentiment Polarity')
    plt.tight_layout()
    plt.savefig('sentiment_distribution_boxplot.png')
    plt.close()

# Function to plot correlation heatmap
def plot_correlation_heatmap(df):
    print("\n=== Correlation Heatmap ===")
    
    # Compute correlations
    correlation_matrix = df[['text_length', 'word_count', 'sentiment']].corr()
    
    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()

# Function to plot most frequent words
def plot_most_frequent_words(df):
    print("\n=== Most Frequent Words ===")
    
    stop_words = set(stopwords.words('english'))
    
    for label in df['final_label'].unique():
        text = ' '.join(df[df['final_label'] == label]['cleaned_text'])
        words = [word for word in word_tokenize(text) if word not in stop_words]
        word_counts = Counter(words)
        most_common_words = dict(word_counts.most_common(10))
        
        # Bar chart for most common words
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(most_common_words.values()), y=list(most_common_words.keys()), palette='Set2')
        plt.title(f'Most Frequent Words for {label} Posts')
        plt.xlabel('Frequency')
        plt.ylabel('Words')
        plt.tight_layout()
        plt.savefig(f'most_frequent_words_{label}.png')
        plt.close()

# Function to plot n-grams
def plot_ngrams(df, n=2):
    print("\n=== N-Gram Analysis ===")
    
    for label in df['final_label'].unique():
        text = ' '.join(df[df['final_label'] == label]['cleaned_text'])
        vectorizer = CountVectorizer(ngram_range=(n, n), stop_words='english').fit([text])
        ngrams = vectorizer.get_feature_names_out()
        ngram_counts = vectorizer.transform([text]).toarray().flatten()
        ngram_freq = dict(zip(ngrams, ngram_counts))
        most_common_ngrams = dict(sorted(ngram_freq.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # Bar chart for most common n-grams
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(most_common_ngrams.values()), y=list(most_common_ngrams.keys()), palette='Set2')
        plt.title(f'Most Common {n}-Grams for {label} Posts')
        plt.xlabel('Frequency')
        plt.ylabel(f'{n}-Grams')
        plt.tight_layout()
        plt.savefig(f'{n}grams_{label}.png')
        plt.close()

# Function to plot text length vs sentiment
def plot_text_length_vs_sentiment(df):
    print("\n=== Text Length vs Sentiment ===")
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='text_length', y='sentiment', hue='final_label', data=df, palette='Set2')
    plt.title('Text Length vs Sentiment Polarity')
    plt.xlabel('Text Length')
    plt.ylabel('Sentiment Polarity')
    plt.legend(title='Label')
    plt.tight_layout()
    plt.savefig('text_length_vs_sentiment.png')
    plt.close()

# Main function to execute the EDA pipeline
def main():
    # Set style for better visualizations
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = [12, 6]
    
    # Load and clean data
    print("Loading and cleaning data...")
    df = load_data()
    
    # Perform analysis
    basic_statistics(df)
    analyze_label_distribution(df)
    analyze_text_length(df)
    generate_wordclouds(df)
    analyze_annotator_agreement(df)
    plot_sentiment_distribution(df)
    plot_correlation_heatmap(df)
    plot_most_frequent_words(df)
    plot_ngrams(df, n=2)  # Bigrams
    plot_ngrams(df, n=3)  # Trigrams
    plot_text_length_vs_sentiment(df)
    
    print("\nEnhanced EDA Report completed! Check the generated visualizations.")

if __name__ == "__main__":
    main()
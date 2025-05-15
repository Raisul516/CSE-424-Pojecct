import pandas as pd

# Load your dataset
df = pd.read_csv("soft_labeled_full_dataset.csv")

# Drop rows where post_text is missing
df = df.dropna(subset=["post_text"])

# Number of data points
print("Number of data points:", len(df))

# Add text length and word count
df["text_length"] = df["post_text"].apply(len)
df["word_count"] = df["post_text"].apply(lambda x: len(x.split()))

# --- Text Length ---
print("\n--- Text Length Stats ---")
print("Mean:", df["text_length"].mean())
print("Median:", df["text_length"].median())
print("Mode:", df["text_length"].mode()[0])
print("Range:", df["text_length"].min(), "-", df["text_length"].max())

# --- Word Count ---
print("\n--- Word Count Stats ---")
print("Mean:", df["word_count"].mean())
print("Median:", df["word_count"].median())
print("Mode:", df["word_count"].mode()[0])
print("Range:", df["word_count"].min(), "-", df["word_count"].max())

# --- Label Stats ---
print("\n--- Final Label Distribution ---")
print(df["final_label"].value_counts())
print("Most common label (Mode):", df["final_label"].mode()[0])

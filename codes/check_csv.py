import pandas as pd

# Load and examine the dataset
df = pd.read_csv("soft_labeled_full_dataset.csv")

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("\nFirst 3 rows:")
print(df.head(3))

# Check distribution of probabilities
print("\nAverage probabilities:")
if 'normal_prob' in df.columns:
    print(f"Normal: {df['normal_prob'].mean():.4f}")
if 'offensive_prob' in df.columns:
    print(f"Offensive: {df['offensive_prob'].mean():.4f}")
if 'hatespeech_prob' in df.columns:
    print(f"Hatespeech: {df['hatespeech_prob'].mean():.4f}")

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum()) 
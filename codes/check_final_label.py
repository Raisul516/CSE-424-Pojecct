import pandas as pd

# Load the dataset
df = pd.read_csv('soft_labeled_full_dataset.csv')

print("=== Final Label Distribution ===")
final_label_counts = df['final_label'].value_counts()
total = len(df)

print("\nFinal Label counts and percentages:")
for label, count in final_label_counts.items():
    percentage = (count / total) * 100
    print(f"{label}: {count:,} instances ({percentage:.2f}%)")

print(f"\nTotal instances: {total:,}")

# Compare with 'label' column
print("\n=== Comparison with 'label' column ===")
label_counts = df['label'].value_counts()

print("\nLabel column counts and percentages:")
for label, count in label_counts.items():
    percentage = (count / total) * 100
    print(f"{label}: {count:,} instances ({percentage:.2f}%)")

# Check if they match
print("\n=== Checking for differences ===")
differences = (df['label'] != df['final_label']).sum()
if differences > 0:
    print(f"Found {differences:,} differences between 'label' and 'final_label' columns")
    print("\nSample of rows where labels differ:")
    print(df[df['label'] != df['final_label']][['post_id', 'label', 'final_label']].head())

print("\n=== Dataset Information ===")
print(f"Total rows: {len(df)}")
print("\nColumns in dataset:")
for col in df.columns:
    print(f"- {col}")

print("\n=== Sample Data ===")
print("\nFirst 3 rows of relevant columns:")
columns_to_show = ['post_id', 'label', 'final_label'] if 'final_label' in df.columns else ['post_id', 'label']
print(df[columns_to_show].head(3)) 
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('soft_labeled_full_dataset.csv')

# Calculate label distribution
label_counts = df['label'].value_counts()
total = len(df)

# Print counts and percentages
print("\nLabel Distribution:")
for label, count in label_counts.items():
    percentage = (count / total) * 100
    print(f"{label}: {count:,} instances ({percentage:.2f}%)")

print(f"\nTotal instances: {total:,}")

# Create pie chart
plt.figure(figsize=(10, 8))
plt.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%')
plt.title('Label Distribution')
plt.savefig('label_distribution_pie.png')
plt.close() 
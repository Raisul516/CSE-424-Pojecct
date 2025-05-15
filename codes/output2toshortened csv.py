import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

# Step 1: Load the dataset
df = pd.read_csv("output2.csv")

# Step 2: Create the 'label' column using majority vote from annotators
def majority_label(row):
    labels = [row['annotator_1_label'], row['annotator_2_label'], row['annotator_3_label']]
    return Counter(labels).most_common(1)[0][0]

df['label'] = df.apply(majority_label, axis=1)

# Step 3: Perform stratified sampling to get 5,000 rows while maintaining label proportions
df_sampled, _ = train_test_split(
    df,
    train_size=5000,
    stratify=df['label'],
    random_state=42
)

# Step 4: Save the sampled dataframe to a new CSV file
df_sampled.to_csv("sampled_5000_stratified.csv", index=False)

# Optional: Verify the class distribution in the sampled data
print(df_sampled['label'].value_counts())
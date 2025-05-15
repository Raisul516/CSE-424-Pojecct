import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 1: Load the Dataset
df = pd.read_csv("soft_labeled_full_dataset.csv")  # Ensure this file is in the same directory

# Print column names to identify the correct label column
print("Column names in the dataset:", df.columns.tolist())

# Check first few rows to understand data structure
print("First few rows of the dataset:")
print(df.head())

# Use the correct label column (either 'label' or one of the annotator labels)
if 'label' in df.columns:
    label_column = 'final_label'
elif 'annotator_1_label' in df.columns:
    label_column = 'annotator_1_label'
else:
    # Find first column with 'label' in its name
    label_columns = [col for col in df.columns if 'label' in col.lower()]
    label_column = label_columns[0] if label_columns else None
    if not label_column:
        raise ValueError("Could not find a suitable label column in the dataset")

print(f"Using '{label_column}' as the label column")

# Ensure 'text' column exists or create it
if 'text' not in df.columns:
    if 'post_text' in df.columns:
        df = df.rename(columns={"post_text": "text"})  # Renaming for clarity
    else:
        # Find first column with 'text' in its name
        text_columns = [col for col in df.columns if 'text' in col.lower()]
        text_column = text_columns[0] if text_columns else None
        if text_column:
            df = df.rename(columns={text_column: "text"})
        else:
            raise ValueError("Could not find a suitable text column in the dataset")

# Keep only relevant columns for text classification
df = df[["text", label_column]].copy()  
df = df.rename(columns={label_column: "label"})

# Check the contents of 'label' column to verify data
print("\nFirst 5 labels:")
print(df['label'].head())
print(f"Label column data type: {df['label'].dtype}")
print(f"Unique label values: {df['label'].unique()}")

# Map labels to integers
label_mapping = {"normal": 0, "offensive": 1, "hatespeech": 2}
df["label_numeric"] = df["label"].map(label_mapping)

# Check if there are any NaN values after mapping
null_count = df['label_numeric'].isnull().sum()
print(f"Number of null values after mapping: {null_count}")

if null_count > 0:
    print("Sample of rows with unmapped labels:")
    print(df[df['label_numeric'].isnull()]['label'].value_counts().head())
    # Handle unmapped values
    df = df.dropna(subset=["label_numeric"])
    print(f"Dropped {null_count} rows with unmapped labels")

# Use the numeric labels for classification
df["label"] = df["label_numeric"]
df = df.drop("label_numeric", axis=1)

# Step 2: Split the Dataset into Train and Test
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# Step 3: Vectorize the Text Using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 4: Train a Logistic Regression Model
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_tfidf, y_train)

# Step 5: Evaluate the Model
y_pred = model.predict(X_test_tfidf)
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=list(label_mapping.keys())))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Print accuracy as a percentage
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2%}")

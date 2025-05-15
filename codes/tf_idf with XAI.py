import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import eli5
from eli5.sklearn import explain_weights_sklearn
from eli5.sklearn import explain_prediction_sklearn
from eli5.lime import TextExplainer
from eli5.lime.samplers import MaskingTextSampler
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the Dataset
df = pd.read_csv("soft_labeled_full_dataset.csv")  # Ensure this file is in the same directory

# Print column names and preview
print("Column names in the dataset:", df.columns.tolist())
print("First few rows of the dataset:")
print(df.head())

# Identify label column
if 'label' in df.columns:
    label_column = 'final_label'
elif 'annotator_1_label' in df.columns:
    label_column = 'annotator_1_label'
else:
    label_columns = [col for col in df.columns if 'label' in col.lower()]
    label_column = label_columns[0] if label_columns else None
    if not label_column:
        raise ValueError("Could not find a suitable label column in the dataset")

print(f"Using '{label_column}' as the label column")

# Ensure text column
if 'text' not in df.columns:
    if 'post_text' in df.columns:
        df = df.rename(columns={"post_text": "text"})
    else:
        text_columns = [col for col in df.columns if 'text' in col.lower()]
        text_column = text_columns[0] if text_columns else None
        if text_column:
            df = df.rename(columns={text_column: "text"})
        else:
            raise ValueError("Could not find a suitable text column")

# Keep only text and label
df = df[["text", label_column]].copy()
df = df.rename(columns={label_column: "label"})

# Map string labels to numeric
label_mapping = {"normal": 0, "offensive": 1, "hatespeech": 2}
df["label_numeric"] = df["label"].map(label_mapping)

# Drop unmapped labels
null_count = df['label_numeric'].isnull().sum()
print(f"Number of null values after mapping: {null_count}")
if null_count > 0:
    print("Sample of rows with unmapped labels:")
    print(df[df['label_numeric'].isnull()]['label'].value_counts().head())
    df = df.dropna(subset=["label_numeric"])
    print(f"Dropped {null_count} rows with unmapped labels")

# Finalize label column
df["label"] = df["label_numeric"].astype(int)
df = df.drop("label_numeric", axis=1)

# Step 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# Step 3: TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 4: Train Logistic Regression with balanced class weights
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_tfidf, y_train)

# Step 5: Evaluation
y_pred = model.predict(X_test_tfidf)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=list(label_mapping.keys())))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2%}")

def explain_model_globally():
    """Generate and save global model explanation"""
    print("\nGenerating global model explanation...")
    global_explanation = explain_weights_sklearn(model, vec=vectorizer, top=30)
    global_html = eli5.format_as_html(global_explanation)
    
    with open("global_explanation.html", "w", encoding="utf-8") as f:
        f.write(global_html)
    print("✓ Global explanation saved to 'global_explanation.html'")
    
    # Also generate feature importance plot
    feature_names = vectorizer.get_feature_names_out()
    importances = model.coef_[0]
    n_top_features = 20
    
    indices = np.argsort(np.abs(importances))[-n_top_features:]
    top_features = feature_names[indices]
    top_importances = importances[indices]
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_importances, y=top_features)
    plt.title('Top Feature Importance')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    print("✓ Feature importance plot saved to 'feature_importance.png'")

def explain_prediction_detailed(text, index=None):
    """Generate both eli5 and LIME explanations for a single prediction"""
    print(f"\nExplaining prediction for text{f' at index {index}' if index is not None else ''}:")
    print(f"Text: {text[:200]}...")
    
    # Get eli5 explanation
    vec_text = vectorizer.transform([text])
    eli5_explanation = explain_prediction_sklearn(model, text, vec=vectorizer)
    eli5_html = eli5.format_as_html(eli5_explanation)
    
    # Get LIME explanation
    te = TextExplainer(random_state=42, sampler=MaskingTextSampler(random_state=42))
    
    def predict_proba(texts):
        vec_texts = vectorizer.transform(texts)
        return model.predict_proba(vec_texts)
    
    te.fit(text, predict_proba)
    lime_explanation = te.explain_prediction()
    lime_html = eli5.format_as_html(lime_explanation)
    
    # Save both explanations
    filename_prefix = f"prediction_explanation_{index}" if index is not None else "prediction_explanation"
    
    with open(f"{filename_prefix}_eli5.html", "w", encoding="utf-8") as f:
        f.write(eli5_html)
    
    with open(f"{filename_prefix}_lime.html", "w", encoding="utf-8") as f:
        f.write(lime_html)
    
    print(f"✓ Explanations saved to '{filename_prefix}_eli5.html' and '{filename_prefix}_lime.html'")

# Generate explanations
explain_model_globally()

# Explain a few sample predictions
print("\nGenerating explanations for sample predictions...")
for i, text in enumerate(X_test[:3]):
    explain_prediction_detailed(text, i)

print("\n✅ All done! Check the generated HTML files and feature_importance.png in your folder.")

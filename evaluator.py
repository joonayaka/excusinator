import pandas as pd
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# --- Load CSV file ---
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError("❌ No CSV file found in the current directory.")
csv_path = csv_files[0]

df = pd.read_csv(csv_path)
if 'text' not in df.columns or 'label' not in df.columns:
    raise ValueError("❌ CSV must contain 'text' and 'label' columns.")

# --- Load model and vectorizer ---
try:
    model = joblib.load("model_weights.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except FileNotFoundError:
    raise FileNotFoundError("❌ model_weights.pkl or vectorizer.pkl not found. Run training first.")

# --- Encode labels ---
le = LabelEncoder()
y = le.fit_transform(df['label'])
X = vectorizer.transform(df['text'])

# --- Train/Test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Predict ---
y_pred = model.predict(X_test)

# --- Metrics ---
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_)
cm = confusion_matrix(y_test, y_pred)

# --- Output Results ---
print(f"\n Model Evaluation Summary")
print(f"---------------------------")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n")
print(report)

# --- Show confusion matrix ---
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

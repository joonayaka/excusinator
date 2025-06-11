from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Automatically find the first .csv file in the current directory
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError("No CSV file found in the current directory.")
csv_path = csv_files[0]

# Load the dataset
df = pd.read_csv(csv_path)

# Ensure it has the required columns
if 'text' not in df.columns or 'label' not in df.columns:
    raise ValueError("CSV file must contain 'text' and 'label' columns.")

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df['label'])

# -------------If loading weights, comment out this section--------------------
# TF-IDF vectorization of text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(max_iter=5000, class_weight='balanced')
model.fit(X_train, y_train)

# Save weights
joblib.dump(model, "model_weights.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
# ------------------------------------------------------------------------------

# Load weights (optional)
model = joblib.load("model_weights.pkl")
vectorizer = joblib.load("vectorizer.pkl")
X = vectorizer.fit_transform(df['text'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluate model
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Interactive prediction + feedback loop
while True:
    user_input = input("\nEnter your homework excuse (type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break

    X_new = vectorizer.transform([user_input])
    prediction = model.predict(X_new)
    predicted_label = le.inverse_transform(prediction)[0]

    probs = model.predict_proba(X_new)[0]

    print(f"\nPrediction: {predicted_label}")
    print("Confidence scores:")
    for label, prob in zip(le.classes_, probs):
        print(f"{label:<10}:{prob:>7.4f}")

    # Ask for feedback
    feedback = input("\nIs this prediction correct? (yes/no): ").strip().lower()
    if feedback == 'yes':
        final_label = predicted_label
    else:
        print("Available categories:")
        for i, label in enumerate(le.classes_, 1):
            print(f"  {i}. {label}")
        while True:
            corrected = input("Please enter the correct category name: ").strip()
            if corrected in le.classes_:
                final_label = corrected
                break
            else:
                print("Invalid label. Try again.")

    # Save to CSV
    new_row = pd.DataFrame([[user_input, final_label]], columns=['text', 'label'])
    new_row.to_csv(csv_path, mode='a', index=False, header=False)
    print(f"Saved: \"{user_input}\" as \"{final_label}\" to {csv_path}")


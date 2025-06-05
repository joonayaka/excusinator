from sklearn.linear_model import LogisticRegression as logReg
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\hanna\Downloads\homework_excuses.csv")

# Step 2: Encode string labels into integers
le = LabelEncoder()
y_log = le.fit_transform(df['label'])  # y_log will be your label array

vectorizer = TfidfVectorizer()
X_log = vectorizer.fit_transform(df['text'])  # X_log is your feature matrix

X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log, y_log, test_size=0.2, random_state=42)

start = time.time() #calculate training time
logreg = logReg(max_iter=5000) #create instance of scikits logistic regression
logreg.fit(X_train_log, y_train_log) #train model on trainig set
time_sk_logreg = time.time() - start

#weights_sk_logreg = np.concatenate(([logreg.intercept_[0]], logreg.coef_[0])) #calculate weights
predictions_sk_logreg = logreg.predict(X_test_log) #uses trained model to make predictions
accuracy_sk_logreg = accuracy_score(y_test_log, predictions_sk_logreg) #calculate mse using true values and predictions

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test_log, predictions_sk_logreg, target_names=le.classes_))

# Confusion matrix
cm = confusion_matrix(y_test_log, predictions_sk_logreg)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

while True:
    new_excuse = input("\nEnter your homework excuse (or type 'exit' to quit): ")
    if new_excuse.lower() == "exit":
        break

    # Vectorize and predict
    X_new = vectorizer.transform([new_excuse])
    prediction = logreg.predict(X_new)
    predicted_label = le.inverse_transform(prediction)[0]

    # Show probability scores for all classes
    probs = logreg.predict_proba(X_new)[0]
    print(f"\nPrediction: {predicted_label}")
    print("Confidence scores for each category:")
    for label, prob in zip(le.classes_, probs):
        print(f"  {label}: {prob:.4f}")

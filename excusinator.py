import os
import pandas as pd
import joblib
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load CSV File ---
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError("No CSV file found in this directory.")
csv_path = csv_files[0]

# --- Initial Training ---
def retrain_model():
    global df, model, vectorizer, le

    df = pd.read_csv(csv_path)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must have 'text' and 'label' columns.")

    le = LabelEncoder()
    y = le.fit_transform(df['label'])

    #-------------------Comment out if loading weights------------------------
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['text'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=5000, class_weight='balanced')
    model.fit(X_train, y_train)

    joblib.dump(model, "model_weights.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

    #----------------------------------------------------------------------

    #Load weights
    #Uncomment this section when loading
    #model = joblib.load("model_weights.pkl")
    #vectorizer = joblib.load("vectorizer.pkl")
    #X = vectorizer.fit_transform(df['text'])
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Show confusion matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix After Retraining")
    plt.tight_layout()
    plt.show()

# Train the model initially
retrain_model()

# --- initial interface ---
class ExcuseClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Excusinator")
        self.root.geometry("600x500")
        self.current_text = ""
        self.predicted_label = ""

        self.label_frame = ttk.LabelFrame(root, text="Enter Your Homework Excuse:")
        self.label_frame.pack(fill="x", padx=20, pady=10)

        self.text_input = tk.Text(self.label_frame, height=4)
        self.text_input.pack(fill="x", padx=10, pady=5)

        self.predict_button = ttk.Button(root, text="Predict", command=self.predict)
        self.predict_button.pack(pady=10)

        self.result_label = ttk.Label(root, text="", font=("Arial", 12))
        self.result_label.pack()

        self.confidence_box = tk.Text(root, height=6, state='disabled')
        self.confidence_box.pack(pady=5)

        self.feedback_frame = ttk.Frame(root)
        self.correct_button = ttk.Button(self.feedback_frame, text="✅ Correct", command=self.save_prediction)
        self.incorrect_button = ttk.Button(self.feedback_frame, text="❌ Incorrect", command=self.show_dropdown)

        self.dropdown = ttk.Combobox(root, state='disabled')
        self.submit_correction = ttk.Button(root, text="Submit Correction", command=self.save_correction)
# --- require input ---
    def predict(self):
        self.current_text = self.text_input.get("1.0", "end").strip()
        if not self.current_text:
            messagebox.showwarning("Input Required", "Please enter a homework excuse.")
            return

        X_new = vectorizer.transform([self.current_text]) #transform input using vectorizer
        prediction = model.predict(X_new) #make prediction
        probs = model.predict_proba(X_new)[0] #gets confidence score for each label
        self.predicted_label = le.inverse_transform(prediction)[0] #convert number back to label (string)

        self.result_label.config(text=f"Prediction: {self.predicted_label}")
        self.confidence_box.config(state='normal')
        self.confidence_box.delete("1.0", "end")
        for label, prob in zip(le.classes_, probs):
            self.confidence_box.insert("end", f"{label:<10}: {prob:.4f}\n")
        self.confidence_box.config(state='disabled')

        self.feedback_frame.pack(pady=5)
        self.correct_button.pack(side='left', padx=10)
        self.incorrect_button.pack(side='right', padx=10)
# --- saves the prediction in the database ---
    def save_prediction(self):
        new_row = pd.DataFrame([[self.current_text, self.predicted_label]], columns=['text', 'label'])
        new_row.to_csv(csv_path, mode='a', header=False, index=False)
        retrain_model()
        messagebox.showinfo("Saved", f"Saved as '{self.predicted_label}' and model retrained.")
        self.reset()
# --- dropdown list of labels ---
    def show_dropdown(self):
        self.dropdown.config(state='readonly')
        self.dropdown['values'] = le.classes_.tolist()
        self.dropdown.set("Select correct label")
        self.dropdown.pack(pady=10)
        self.submit_correction.pack()
# --- validate label / prevent invalid or new label input---
    def save_correction(self):
        corrected = self.dropdown.get()
        if corrected not in le.classes_:
            messagebox.showwarning("Invalid", "Please choose a valid label.")
            return
        new_row = pd.DataFrame([[self.current_text, corrected]], columns=['text', 'label'])
        new_row.to_csv(csv_path, mode='a', header=False, index=False)
        retrain_model()
        messagebox.showinfo("Saved", f"Saved as '{corrected}' and model retrained.")
        self.reset()

    def reset(self):
        self.text_input.delete("1.0", "end")
        self.result_label.config(text="")
        self.confidence_box.config(state='normal')
        self.confidence_box.delete("1.0", "end")
        self.confidence_box.config(state='disabled')
        self.feedback_frame.pack_forget()
        self.dropdown.pack_forget()
        self.submit_correction.pack_forget()
        self.current_text = ""
        self.predicted_label = ""

# Launch GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = ExcuseClassifierApp(root)
    root.mainloop()

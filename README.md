# Excusinator
A data science project using logistic regression to classify student's excuses into multiple categories.

This project is a machine learning-based excuse classifier that predicts what kind of excuse a student is giving for not doing homework. It features a clean, interactive GUI and supports real-time learning and retraining.

## Features

 - Classifies excuses into categories (e.g., *funny*, *plausible*, *fake*, etc.)
 - Uses TF-IDF + Logistic Regression for text classification
 - Interactive **Tkinter GUI**
 - Displays prediction confidence for each category
 - Asks the user if the prediction was correct
 - Allows manual correction if incorrect
 - Saves new examples to the dataset
 - **Automatically retrains** the model on each new input
 - Displays a **confusion matrix** after every retraining
 - Extra: save weights for the trained model and vectorizer
 - Extra: load previously saved weights if needed

## Dataset Format

The classifier uses a CSV file with the following format:


text,label
"My dog ate my homework",funny
"I had a family emergency",plausible


- The file **must** be in the same directory as the Python script
- The classifier automatically detects the first `.csv` file

## How It Works

1. Loads the CSV and trains a logistic regression classifier using scikit-learn
2. Builds a TF-IDF matrix from excuse text
3. Provides a GUI for entering excuses
4. Displays predicted category and confidence scores
5. Updates the dataset and retrains the model on each confirmed or corrected example
6. Shows a **confusion matrix heatmap** after retraining

## 🛠 Requirements

Install required Python packages using pip:

```bash
pip install pandas scikit-learn matplotlib seaborn joblib
```

This project uses:

- Python 3.8+
- Tkinter (included in standard Python)
- scikit-learn
- pandas
- seaborn
- matplotlib
- joblib

## Running the Program

```bash
python excusinator.py
```

Make sure your CSV file (e.g., `excuses.csv`) is in the same folder as `excusinator.py`.

## Example Use

1. Enter an excuse like “I overslept because of a time zone change.”
2. Click **Predict**
3. Review the predicted label and confidence scores
4. Click ✅ if the prediction is correct or ❌ to correct it
5. The program appends the new example to the dataset and retrains the model
6. A confusion matrix is shown to visualize current performance

## Confusion Matrix

After each retraining, a heatmap is displayed to show how well the model is performing on different classes using a 20% test split.

## Files Created

 - `model_weights.pkl`: Trained logistic regression model
 - `vectorizer.pkl`: Trained TF-IDF vectorizer
 - Appended `your_dataset.csv`: With new labeled examples

## Notes

 - The program retrains **every time a new example is added**
 - No internet access is required — everything runs locally
 - The GUI is responsive and works on most desktops (Windows/Linux/Mac)

## Future Improvements

 - Save model accuracy over time
 - Track user corrections and visualize model drift
 - Web-based version using Streamlit or Flask

## Author

Runee Zubaida Zahid
Jeong Hanna
Ahmed Ishraq

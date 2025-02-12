import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords')

# Load the data
df = pd.read_csv(r"C:\Users\Admin\Desktop\mini\bbc\bbc_dataset.csv") 

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)  # Remove punctuation and special characters
    text = " ".join([word for word in text.split() if word not in stopwords.words("english")]) 
    return text

# Apply preprocessing
df['Cleaned_Text'] = df['Text'].apply(preprocess_text)

# Create TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['Cleaned_Text'])

# Encode categories
categories = {category: idx for idx, category in enumerate(df['Category'].unique())}
df['Category_Code'] = df['Category'].map(categories)
y = df['Category_Code']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)  # Increase max_iter for stability
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=categories.keys()))

# Function to classify new text
def predict_category(text):
    processed_text = preprocess_text(text)
    text_tfidf = vectorizer.transform([processed_text])
    predicted_label = model.predict(text_tfidf)[0]
    return list(categories.keys())[predicted_label]

# Test the model with an example
sample_text = "The stock market is experiencing high volatility today."
print("Predicted Category:", predict_category(sample_text))


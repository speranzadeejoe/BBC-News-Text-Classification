# BBC-News-Text-Classification
# BBC News Text Classification

## Project Overview
This project implements a text classification model using Natural Language Processing (NLP) and Machine Learning to categorize BBC news articles into different categories. The dataset used is the BBC News dataset.

## Technologies Used
- Python
- Pandas
- NumPy
- NLTK (Natural Language Toolkit)
- Scikit-learn (sklearn)
- Logistic Regression
- TF-IDF (Term Frequency-Inverse Document Frequency) Vectorization

## Dataset
The dataset (`bbc_dataset.csv`) contains two columns:
- `Text`: The news article content.
- `Category`: The category label for the article.

## Installation & Setup
1. Install the required libraries using pip:
   ```bash
   pip install pandas numpy nltk scikit-learn
   ```
2. Download the NLTK stopwords:
   ```python
   import nltk
   nltk.download('stopwords')
   ```
3. Place the `bbc_dataset.csv` file in the correct directory (update the path in the script if needed).

## Preprocessing Steps
- Convert text to lowercase.
- Remove punctuation and special characters.
- Remove stopwords.
- Apply TF-IDF vectorization.
- Encode the category labels numerically.

## Model Training
- The dataset is split into training and testing sets (80% training, 20% testing).
- A Logistic Regression model is trained on the processed data.
- The model is evaluated using accuracy score and classification report.

## Usage
### Running the Script
Execute the script in a Python environment:
```bash
python script.py
```
### Predicting a New Text Category
Use the `predict_category` function to classify new text:
```python
sample_text = "The stock market is experiencing high volatility today."
print("Predicted Category:", predict_category(sample_text))
```

## Output
- The script prints the model's accuracy.
- The classification report shows precision, recall, and F1-score for each category.
- The function `predict_category` predicts the category of a given text.

## Improvements & Future Work
- Use advanced NLP techniques like stemming and lemmatization.
- Experiment with different machine learning models like Naïve Bayes or Neural Networks.
- Tune hyperparameters for better accuracy.

## License
This project is open-source and available for educational purposes.


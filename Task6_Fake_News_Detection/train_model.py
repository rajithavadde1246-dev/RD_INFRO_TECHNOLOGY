import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle

# Load dataset
data = pd.read_csv("news.csv")

# Features and labels
X = data["text"]
y = data["label"]

# TF-IDF
vectorizer = TfidfVectorizer()

X_vectorized = vectorizer.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = MultinomialNB()

model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained successfully!")
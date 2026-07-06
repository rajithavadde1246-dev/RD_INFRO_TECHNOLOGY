import pandas as pd
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download("punkt")
nltk.download("stopwords")

data = pd.read_csv("dataset/sentiment_data.csv")

def clean_text(text):
    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word.isalnum()]
    words = [word for word in words if word not in stopwords.words("english")]
    return " ".join(words)

data["clean_text"] = data["text"].apply(clean_text)

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data["clean_text"])
y = data["sentiment"]

model = LogisticRegression()

model.fit(X, y)

joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model trained successfully.")
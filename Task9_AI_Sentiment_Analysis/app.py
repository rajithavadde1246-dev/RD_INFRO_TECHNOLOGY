from flask import Flask, render_template, request
import joblib

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob

import nltk

nltk.download("punkt")
nltk.download("stopwords")

app = Flask(__name__)

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

stop_words = set(stopwords.words("english"))

def preprocess(text):

    text = str(TextBlob(text).correct())

    words = word_tokenize(text.lower())

    words = [word for word in words if word.isalpha()]

    words = [word for word in words if word not in stop_words]

    return " ".join(words)

@app.route("/", methods=["GET", "POST"])

def home():

    prediction = ""

    if request.method == "POST":

        message = request.form["message"]

        clean = preprocess(message)

        vector = vectorizer.transform([clean])

        prediction = model.predict(vector)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":

    app.run(debug=True)
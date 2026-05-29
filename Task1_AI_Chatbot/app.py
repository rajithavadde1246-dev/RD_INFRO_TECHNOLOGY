from flask import Flask, render_template, request
import nltk
from nltk.tokenize import word_tokenize

# Download tokenizer data (run only once)
nltk.download('punkt')

app = Flask(__name__)

# Chatbot responses
chat_responses = {
    "hi": "Hello! Welcome to our customer support service.",
    "hello": "Hi there! How can I help you today?",
    "price": "Our products are available starting from ₹999.",
    "refund": "Refunds are usually processed within 7 working days.",
    "bye": "Thank you for visiting. Have a great day!"
}

# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Chatbot API
@app.route("/get")
def get_bot_reply():

    user_message = request.args.get("msg")

    if not user_message:
        return "Please enter your message."

    # lowercase conversion
    user_message = user_message.lower()

    # tokenize message
    message_words = word_tokenize(user_message)

    # check words in dictionary
    for word in message_words:
        if word in chat_responses:
            return chat_responses[word]

    return "Sorry, I could not understand your message."


# Run app
if __name__ == "__main__":
    app.run(debug=True)
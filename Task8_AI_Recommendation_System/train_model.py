import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read movie dataset
movies = pd.read_csv("movies.csv")

# Convert genres into TF-IDF vectors
tfidf = TfidfVectorizer()

genre_matrix = tfidf.fit_transform(movies["genre"])

# Calculate similarity between movies
similarity = cosine_similarity(genre_matrix)

# Save everything into a pickle file
data = {
    "movies": movies,
    "similarity": similarity
}

with open("recommender.pkl", "wb") as file:
    pickle.dump(data, file)

print("Recommendation model created successfully!")
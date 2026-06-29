import streamlit as st
import pickle

st.set_page_config(
    page_title="AI Movie Recommendation System",
    layout="centered"
)

st.title("AI Movie Recommendation System")

st.write("Select a movie to get similar movie recommendations.")

with open("recommender.pkl", "rb") as file:
    data = pickle.load(file)

movies = data["movies"]
similarity = data["similarity"]


def recommend(movie_name):

    movie_index = movies[movies["title"] == movie_name].index[0]

    scores = list(enumerate(similarity[movie_index]))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recommended_movies = []

    for movie in scores[1:6]:
        recommended_movies.append(movies.iloc[movie[0]]["title"])

    return recommended_movies


selected_movie = st.selectbox(
    "Choose a Movie",
    movies["title"]
)

if st.button("Recommend Movies"):

    result = recommend(selected_movie)

    st.subheader("Recommended Movies")

    for movie in result:
        st.write(movie)

st.markdown("---")
st.write("Developed using Python, Pandas, NumPy, Scikit-learn and Streamlit")
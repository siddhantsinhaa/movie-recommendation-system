from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__, template_folder="../templates", static_folder="../public")

# -----------------------------------
# Load Dataset with Safe Paths
# -----------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

movies_path = os.path.join(BASE_DIR, "movies.csv")
ratings_path = os.path.join(BASE_DIR, "ratings.csv")

movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)

# Keep required columns
movies = movies[['movieId', 'title', 'genres']]
ratings = ratings[['userId', 'movieId', 'rating']]

# Merge datasets
data = pd.merge(ratings, movies, on="movieId")

# Create movie-user matrix
movie_user_matrix = data.pivot_table(
    index='title',
    columns='userId',
    values='rating'
).fillna(0)

# Compute similarity between movies
movie_similarity = cosine_similarity(movie_user_matrix)
movie_similarity_df = pd.DataFrame(
    movie_similarity,
    index=movie_user_matrix.index,
    columns=movie_user_matrix.index
)

# Sorted movie list for input suggestions
movie_titles = sorted(movie_user_matrix.index.tolist())

# Map movie -> genre
movie_genre_map = dict(zip(movies['title'], movies['genres']))


# -----------------------------------
# Recommendation Function
# -----------------------------------
def recommend_movies(user_inputs, top_n=10):
    scores = {}
    similarity_sums = {}

    for movie, rating in user_inputs.items():
        if movie in movie_similarity_df.columns:
            similar_movies = movie_similarity_df[movie].sort_values(ascending=False)

            for similar_movie, similarity_score in similar_movies.items():
                if similar_movie not in user_inputs:
                    if similar_movie not in scores:
                        scores[similar_movie] = 0
                        similarity_sums[similar_movie] = 0

                    scores[similar_movie] += similarity_score * float(rating)
                    similarity_sums[similar_movie] += similarity_score

    results = []
    for movie in scores:
        if similarity_sums[movie] == 0:
            predicted_rating = 0
        else:
            predicted_rating = scores[movie] / similarity_sums[movie]

        predicted_rating = round(min(predicted_rating, 5), 2)

        results.append({
            "title": movie,
            "predicted_rating": predicted_rating,
            "genre": movie_genre_map.get(movie, "Unknown")
        })

    results = sorted(results, key=lambda x: x['predicted_rating'], reverse=True)

    return results[:top_n]


# -----------------------------------
# Routes
# -----------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_inputs = {}
        selected_movies = []
        invalid_movies = []

        for i in range(1, 6):
            movie = request.form.get(f"movie{i}")
            rating = request.form.get(f"rating{i}")

            if movie and rating and movie.strip() != "":
                movie = movie.strip()

                if movie not in movie_titles:
                    invalid_movies.append(movie)
                else:
                    selected_movies.append(movie)
                    user_inputs[movie] = float(rating)

        if len(user_inputs) == 0:
            return render_template(
                "index.html",
                movies=movie_titles,
                error="Please select at least one valid movie and rating."
            )

        if len(invalid_movies) > 0:
            return render_template(
                "index.html",
                movies=movie_titles,
                error=f"These movies are not in dataset: {', '.join(invalid_movies)}"
            )

        if len(selected_movies) != len(set(selected_movies)):
            return render_template(
                "index.html",
                movies=movie_titles,
                error="Please do not select the same movie more than once."
            )

        recommendations = recommend_movies(user_inputs)

        return render_template(
            "result.html",
            recommendations=recommendations,
            user_inputs=user_inputs
        )

    return render_template("index.html", movies=movie_titles)


# IMPORTANT FOR VERCEL
app = app
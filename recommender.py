import os
import zipfile
import urllib.request

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def ensure_movielens_data(data_dir="ml-latest-small"):
    movies_path = os.path.join(data_dir, "movies.csv")
    ratings_path = os.path.join(data_dir, "ratings.csv")

    if os.path.exists(movies_path) and os.path.exists(ratings_path):
        return

    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    zip_path = "ml-latest-small.zip"

    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(".")

    try:
        os.remove(zip_path)
    except OSError:
        pass

def load_movies(path="ml-latest-small/movies.csv"):
    ensure_movielens_data()
    movies = pd.read_csv(path)

    # Extract year from titles like "Toy Story (1995)"
    # Some movies may not have a year; we'll set those to NA
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)").astype("float")

    return movies

def load_ratings(path="ml-latest-small/ratings.csv"):
    ensure_movielens_data()
    return pd.read_csv(path)


def build_rating_stats(movies: pd.DataFrame, ratings: pd.DataFrame):
    # avg rating + how many ratings each movie got
    stats = (
        ratings.groupby("movieId")["rating"]
        .agg(avg_rating="mean", rating_count="count")
        .reset_index()
    )

    movies = movies.merge(stats, on="movieId", how="left")

    # fill movies that have no ratings in this dataset
    movies["avg_rating"] = movies["avg_rating"].fillna(0)
    movies["rating_count"] = movies["rating_count"].fillna(0).astype(int)

    return movies

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

def build_recommender(movies: pd.DataFrame):
    # Turn "Action|Crime|Drama" into "Action Crime Drama"
    corpus = movies["genres"].fillna("").str.replace("|", " ", regex=False)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)  # sparse matrix (memory efficient)

    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(X)

    return X, nn


def recommend(query: str, movies: pd.DataFrame, sim_matrix, k=10):
    matches = movies[movies["title"].str.contains(query, case=False, na=False)]

    if matches.empty:
        return None, f"No movie found matching: {query!r}"

    # If multiple matches, ask the user to choose
    if len(matches) > 1:
        print("\nI found multiple matches. Which one did you mean?\n")
        for i, (row_idx, row) in enumerate(matches.head(10).iterrows(), start=1):
            print(f"{i}) {row['title']}")

        choice = input("\nEnter a number (1-10), or press Enter for #1: ").strip()
        if choice == "":
            idx = matches.index[0]
        else:
            try:
                n = int(choice)
                if not (1 <= n <= min(10, len(matches))):
                    return None, "Invalid choice number."
                idx = matches.head(10).index[n - 1]
            except ValueError:
                return None, "Please enter a valid number."
    else:
        idx = matches.index[0]

    chosen = movies.loc[idx, "title"]
    chosen_year = movies.loc[idx, "year"]

    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    YEAR_WINDOW = 5
    POOL_SIZE = 200

    top_pool = scores[1 : POOL_SIZE + 1]

    # Map: movie index -> similarity score
    sim_lookup = {i: s for i, s in top_pool}

    rec_indices = [i for i, _ in top_pool]
    candidates = movies.loc[rec_indices].copy()

    candidates["similarity"] = candidates.index.map(sim_lookup)
    candidates["similarity"] = candidates["similarity"].fillna(0)

    # Popularity normalization (0 to 1 inside this candidate set)
    max_count = candidates["rating_count"].max()
    if max_count == 0:
        candidates["popularity"] = 0
    else:
        candidates["popularity"] = candidates["rating_count"] / max_count

    # Combined score:
    # - similarity matters most
    # - avg_rating helps quality
    # - popularity helps avoid obscure picks with 1 rating
    candidates["score"] = (
        0.70 * candidates["similarity"]
        + 0.20 * (candidates["avg_rating"] / 5.0)
        + 0.10 * candidates["popularity"]
    )

    candidates = candidates.sort_values("score", ascending=False)

    recs = candidates[["title", "genres", "year", "avg_rating", "rating_count", "score"]].head(k).reset_index(drop=True)
    return chosen, recs

if __name__ == "__main__":
    movies = load_movies()
    ratings = load_ratings()
    movies = build_rating_stats(movies, ratings)

    sim = build_similarity_matrix(movies)
    print("Movie Recommender (try: Toy Story, Batman, Heat, Shrek)\n")
    q = input("Enter a movie you like: ").strip()

    chosen, out = recommend(q, movies, sim, k=10)

    if chosen is None:
        print(out)
    else:
        print(f"\nBecause you liked: {chosen}\n")
        print(out.to_string(index=False))
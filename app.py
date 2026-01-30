import streamlit as st
import pandas as pd

from recommender import (
    load_movies,
    load_ratings,
    build_rating_stats,
    build_recommender,
)

# ---------- Page ----------
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.title("🎬 Movie Recommendation Engine")
st.markdown(
    """
**Find movies similar in genre and era — then rank by quality using MovieLens ratings.**  
Built in Python with a content-based recommender (genre similarity) + a scoring layer (ratings + popularity).
"""
)

with st.expander("How it works (simple)"):
    st.write(
        "- You pick a movie.\n"
        "- We compute similarity using genres.\n"
        "- We optionally filter to a year window around your movie.\n"
        "- Then we rank candidates using a weighted score:\n"
        "  - similarity (most important)\n"
        "  - average rating (quality)\n"
        "  - rating count (popularity)\n"
    )

# ---------- Data ----------
@st.cache_data
def load_data():
    movies = load_movies()
    ratings = load_ratings()
    movies = build_rating_stats(movies, ratings)
    X, nn = build_recommender(movies)
    return movies, X, nn

movies, X, nn = load_data()


def get_matches(q: str) -> pd.DataFrame:
    return movies[movies["title"].str.contains(q, case=False, na=False)]


def recommend_from_index(seed_idx: int, k: int, year_window: int, min_ratings: int, sort_by: str):
    chosen_title = movies.loc[seed_idx, "title"]
    chosen_year = movies.loc[seed_idx, "year"]

    POOL_SIZE = 400

    distances, indices = nn.kneighbors(X[seed_idx], n_neighbors=POOL_SIZE + 1)
    distances = distances.flatten()
    indices = indices.flatten()

    # Skip the first one (it's the movie itself)
    candidate_indices = indices[1:]
    candidate_sims = 1 - distances[1:]  # cosine similarity = 1 - cosine distance

    sim_lookup = {int(i): float(s) for i, s in zip(candidate_indices, candidate_sims)}
    candidates = movies.loc[candidate_indices].copy()

    # Filter by year window (if we have year)
    if pd.notna(chosen_year) and year_window > 0:
        year_filtered = candidates[
            (candidates["year"].notna()) &
            (candidates["year"] >= chosen_year - year_window) &
            (candidates["year"] <= chosen_year + year_window)
        ]
        if len(year_filtered) >= k:
            candidates = year_filtered

    # Attach similarity
    candidates["similarity"] = candidates.index.map(sim_lookup).fillna(0)

    # Popularity filter
    if min_ratings > 0:
        pop_filtered = candidates[candidates["rating_count"] >= min_ratings]
        if len(pop_filtered) >= k:
            candidates = pop_filtered

    # Normalize popularity inside candidate set
    max_count = candidates["rating_count"].max() if len(candidates) else 0
    candidates["popularity"] = 0 if max_count == 0 else (candidates["rating_count"] / max_count)

    # Combined score
    candidates["score"] = (
        0.70 * candidates["similarity"]
        + 0.20 * (candidates["avg_rating"] / 5.0)
        + 0.10 * candidates["popularity"]
    )

    # Sorting
    if sort_by == "avg_rating":
        candidates = candidates.sort_values("avg_rating", ascending=False)
    elif sort_by == "rating_count":
        candidates = candidates.sort_values("rating_count", ascending=False)
    elif sort_by == "year":
        candidates = candidates.sort_values("year", ascending=False)
    else:
        candidates = candidates.sort_values("score", ascending=False)

    recs = candidates[["title", "genres", "year", "avg_rating", "rating_count", "score"]].head(k).reset_index(drop=True)
    return chosen_title, recs


# ---------- Controls (on page, no sidebar) ----------
st.subheader("Controls")
c1, c2, c3, c4 = st.columns(4)

with c1:
    query = st.text_input(
        "Search any movie title",
        placeholder="Type anything: Inception, Interstellar, Godfather, Spider-Man...",
    )

with c2:
    year_window = st.slider("Year window (± years)", 0, 20, 5, 1)

with c3:
    k = st.slider("Recommendations", 5, 25, 10, 1)

with c4:
    min_ratings = st.slider("Min ratings", 0, 200, 25, 5)

sort_by = st.selectbox(
    "Sort results by",
    ["score (recommended)", "avg_rating", "rating_count", "year"],
    index=0
)

# ---------- Layout ----------
left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("1) Choose a movie")

    # If query is empty, allow browsing full list
    if query.strip():
        matches = get_matches(query.strip())
        st.caption(f"Found {len(matches):,} matches in MovieLens.")
        if matches.empty:
            st.warning("No matches found in this dataset.")
            st.stop()
        options = matches.head(50)["title"].tolist()
    else:
        st.caption(f"Browse all titles ({len(movies):,} movies). Tip: type in the box above to filter.")
        options = movies["title"].sort_values().tolist()

    chosen = st.selectbox("Select a movie", options)
    seed_idx = movies.index[movies["title"] == chosen][0]

    go = st.button("Recommend 🎯", width="stretch")

with right:
    st.subheader("2) Recommendations")

    if not go:
        st.caption("Click **Recommend** to generate results.")
    else:
        chosen_title, recs = recommend_from_index(
            seed_idx=seed_idx,
            k=k,
            year_window=year_window,
            min_ratings=min_ratings,
            sort_by=sort_by
        )

        st.success(f"Because you liked: **{chosen_title}**")

        st.markdown("### Top Picks")
        top3 = recs.head(3)
        for i, row in top3.iterrows():
            year_str = f"{int(row['year'])}" if pd.notna(row["year"]) else "—"
            st.markdown(
                f"**{i+1}. {row['title']}** ({year_str})  \n"
                f"⭐ {row['avg_rating']:.2f}  •  {int(row['rating_count'])} ratings  •  score {row['score']:.3f}"
            )

        st.markdown("### Full List")
        st.dataframe(
            recs.style.format(
                {
                    "year": "{:.0f}",
                    "avg_rating": "{:.2f}",
                    "rating_count": "{:.0f}",
                    "score": "{:.3f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

st.markdown("---")
st.caption("Built by Jakie * Written by Chat")
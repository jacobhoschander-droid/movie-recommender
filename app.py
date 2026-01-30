import streamlit as st
import pandas as pd

from recommender import (
    load_movies,
    load_ratings,
    build_rating_stats,
    build_similarity_matrix,
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
    sim = build_similarity_matrix(movies)
    return movies, sim

movies, sim = load_data()

def get_matches(q: str) -> pd.DataFrame:
    return movies[movies["title"].str.contains(q, case=False, na=False)]

def recommend_from_index(seed_idx: int, k: int, year_window: int, min_ratings: int, sort_by: str):
    chosen_title = movies.loc[seed_idx, "title"]
    chosen_year = movies.loc[seed_idx, "year"]

    scores = list(enumerate(sim[seed_idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    POOL_SIZE = 400
    top_pool = scores[1 : POOL_SIZE + 1]
    sim_lookup = {i: s for i, s in top_pool}

    rec_indices = [i for i, _ in top_pool]
    candidates = movies.loc[rec_indices].copy()

    # Filter by year window (if we have year)
    if pd.notna(chosen_year) and year_window > 0:
        candidates = candidates[
            (candidates["year"].notna()) &
            (candidates["year"] >= chosen_year - year_window) &
            (candidates["year"] <= chosen_year + year_window)
        ]

    # If year filter removes too many, fall back
    if len(candidates) < k:
        candidates = movies.loc[rec_indices].copy()

    # Attach similarity (after filtering)
    candidates["similarity"] = candidates.index.map(sim_lookup).fillna(0)

    # Popularity filter
    if min_ratings > 0:
        candidates = candidates[candidates["rating_count"] >= min_ratings]

    # If that removes too many, loosen it automatically
    if len(candidates) < k:
        candidates = movies.loc[rec_indices].copy()
        candidates["similarity"] = candidates.index.map(sim_lookup).fillna(0)

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


# ---------- Sidebar ----------
st.sidebar.header("Controls")

browse_all = st.sidebar.checkbox("Browse all titles (no search needed)", value=False)

if browse_all:
    query = ""
else:
    query = st.sidebar.text_input(
        "Search any movie title",
        placeholder="Type anything: Inception, Interstellar, Godfather, Spider-Man...",
    )

year_window = st.sidebar.slider("Year window (± years)", 0, 20, 5, 1)
k = st.sidebar.slider("Number of recommendations", 5, 25, 10, 1)

min_ratings = st.sidebar.slider("Minimum ratings (N)", 0, 200, 25, 5)

sort_by = st.sidebar.selectbox(
    "Sort results by",
    ["score (recommended)", "avg_rating", "rating_count", "year"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: If search returns many matches, use the dropdown search to narrow fast.")


# ---------- Layout ----------
left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("1) Choose a movie")

    if browse_all:
        st.caption(f"Browsing all MovieLens titles ({len(movies):,} movies).")
        chosen = st.selectbox("Select a title (type to search)", movies["title"].sort_values().tolist())
    else:
        if not query.strip():
            st.info("Type a movie title in the sidebar search, or enable **Browse all titles**.")
            st.stop()

        matches = get_matches(query.strip())
        st.caption(f"Found {len(matches):,} matches in MovieLens.")

        if matches.empty:
            st.warning("No matches found in this dataset. Try a different spelling/title.")
            st.stop()

        chosen = st.selectbox("Select the exact title", matches.head(50)["title"].tolist())

    seed_idx = movies.index[movies["title"] == chosen][0]
    go = st.button("Recommend 🎯", use_container_width=True)

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
            use_container_width=True,
            hide_index=True,
        )

st.markdown("---")
st.caption("Built by Jakie")
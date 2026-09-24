import os
import pickle
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "clustering_model.pkl"))

_model_bundle: Optional[Dict[str, Any]] = None

def load_model_bundle() -> Optional[Dict[str, Any]]:
    """Loads and caches the model bundle from disk."""
    global _model_bundle
    if _model_bundle is not None:
        return _model_bundle

    if not os.path.exists(MODEL_PATH):
        logger.warning(f"Model file not found at {MODEL_PATH}.")
        return None

    try:
        logger.info(f"Loading clustering model bundle from {MODEL_PATH}...")
        with open(MODEL_PATH, "rb") as f:
            _model_bundle = pickle.load(f)
        logger.info("Clustering model bundle successfully loaded.")
        return _model_bundle
    except Exception as e:
        logger.error(f"Failed to load model bundle from {MODEL_PATH}: {e}")
        return None


def is_model_loaded() -> bool:
    """Returns True if the model bundle is loaded and ready for inference."""
    bundle = load_model_bundle()
    return bundle is not None and "model" in bundle and "df" in bundle and "X" in bundle


def get_model_metadata() -> Dict[str, Any]:
    """Returns metadata about the loaded dataset and model."""
    bundle = load_model_bundle()
    if bundle is None:
        return {"loaded": False, "error": "Model bundle not loaded"}
    
    df = bundle.get("df")
    model = bundle.get("model")
    return {
        "loaded": True,
        "total_movies": len(df) if df is not None else 0,
        "n_clusters": getattr(model, "n_clusters", None),
        "columns": df.columns.tolist() if df is not None else [],
    }


def format_poster(path: Any) -> str:
    if isinstance(path, str) and path.startswith("/"):
        return f"https://image.tmdb.org/t/p/w500{path}"
    return path if isinstance(path, str) else ""


def recommend_movie(title: str, n: int = 10) -> Optional[Dict[str, Any]]:
    """
    Recommends top n similar movies based on cluster similarity and euclidean distance.
    Returns None if the movie is not found or model is unavailable.
    """
    bundle = load_model_bundle()
    if bundle is None:
        raise RuntimeError("Clustering model is not loaded. Please ensure clustering_model.pkl exists.")

    model = bundle["model"]
    df = bundle["df"]
    X = bundle["X"]

    target_title = title.strip().lower()
    movie = df[df["title"].astype(str).str.lower() == target_title]
    if movie.empty:
        movie = df[df["original_title"].astype(str).str.lower() == target_title]
        if movie.empty:
            return None

    movie_index = movie.index[0]
    movie_features = X[movie_index].reshape(1, -1)
    prediction = model.predict(movie_features)
    cluster = int(prediction[0])
    cluster_indices = df[df["cluster"] == cluster].index

    distances = euclidean_distances(movie_features, X[cluster_indices])[0]
    result = pd.DataFrame({"index": cluster_indices, "distance": distances})
    result = result[result["index"] != movie_index]
    result = result.sort_values("distance")
    result = result.head(n)

    cols = [
        "id", "title", "original_title", "overview", "poster_path",
        "release_date", "release_year", "vote_average", "vote_count",
        "runtime", "genres", "original_language", "popularity", "budget", "revenue"
    ]
    available_cols = [c for c in cols if c in df.columns]
    rec_df = df.loc[result["index"], available_cols].copy()
    rec_df["similarity"] = [
        int(round(max(60, min(99, 100 - (d * 5))))) for d in result["distance"]
    ]

    if "poster_path" in rec_df.columns:
        rec_df["poster_path"] = rec_df["poster_path"].apply(format_poster)
    recommendations = rec_df.fillna("").to_dict(orient="records")

    queried_movie = df.loc[movie_index, available_cols].to_dict()
    if "poster_path" in queried_movie:
        queried_movie["poster_path"] = format_poster(queried_movie["poster_path"])

    for k, v in queried_movie.items():
        if pd.isna(v):
            queried_movie[k] = ""

    return {
        "movie": queried_movie,
        "cluster": cluster,
        "recommendations": recommendations,
    }


def search_movies(query: str, limit: int = 15) -> List[Dict[str, Any]]:
    """Searches for movies by title substring across all 69,405 titles in the dataset."""
    bundle = load_model_bundle()
    if bundle is None:
        return []
    df = bundle["df"]
    q = query.strip().lower()
    matches = df[df["title"].astype(str).str.lower().str.contains(q, regex=False, na=False)].sort_values(
        by=["popularity", "vote_average"], ascending=[False, False]
    ).head(limit)
    cols = [
        "id", "title", "original_title", "overview", "release_year",
        "poster_path", "vote_average", "genres", "runtime", "popularity", "original_language"
    ]
    available_cols = [c for c in cols if c in df.columns]
    results = matches[available_cols].copy()
    if "poster_path" in results.columns:
        results["poster_path"] = results["poster_path"].apply(format_poster)
    return results.fillna("").to_dict(orient="records")


def get_movie_by_id(movie_id: Any) -> Optional[Dict[str, Any]]:
    """Fetches complete movie details for a specific ID from the 69,405 dataset."""
    bundle = load_model_bundle()
    if bundle is None:
        return None
    df = bundle["df"]
    try:
        mid = int(movie_id)
        match = df[df["id"] == mid]
    except (ValueError, TypeError):
        match = df[df["id"].astype(str) == str(movie_id)]
    
    if match.empty:
        return None
    
    cols = [
        "id", "title", "original_title", "overview", "poster_path",
        "release_date", "release_year", "vote_average", "vote_count",
        "runtime", "genres", "original_language", "popularity", "budget", "revenue", "cluster"
    ]
    available_cols = [c for c in cols if c in df.columns]
    row = match.iloc[0][available_cols].to_dict()
    if "poster_path" in row:
        row["poster_path"] = format_poster(row["poster_path"])
    
    clean = {}
    for k, v in row.items():
        clean[k] = "" if pd.isna(v) else v
    return clean


def get_movies(
    page: int = 1,
    limit: int = 24,
    genre: Optional[str] = None,
    sort_by: str = "popularity",
    search: Optional[str] = None,
    min_rating: float = 0.0,
    language: Optional[str] = None
) -> Dict[str, Any]:
    """Paginated retrieval of movies from the 69,405 dataset with filtering and sorting."""
    bundle = load_model_bundle()
    if bundle is None:
        return {"total": 0, "page": page, "limit": limit, "total_pages": 0, "results": []}
    
    df = bundle["df"]
    filtered = df

    if search:
        q = search.strip().lower()
        filtered = filtered[filtered["title"].astype(str).str.lower().str.contains(q, regex=False, na=False)]

    if genre:
        filtered = filtered[filtered["genres"].astype(str).str.contains(genre.strip(), case=False, regex=False, na=False)]

    if language:
        filtered = filtered[filtered["original_language"].astype(str).str.lower() == language.strip().lower()]

    if min_rating > 0.0:
        filtered = filtered[filtered["vote_average"] >= min_rating]

    # Sorting
    if sort_by == "rating":
        filtered = filtered.sort_values(by=["vote_average", "vote_count"], ascending=[False, False])
    elif sort_by in ["newest", "year"]:
        filtered = filtered.sort_values(by=["release_year", "popularity"], ascending=[False, False])
    elif sort_by == "oldest":
        filtered = filtered.sort_values(by=["release_year", "popularity"], ascending=[True, False])
    elif sort_by == "az":
        filtered = filtered.sort_values(by="title", ascending=True)
    else:
        filtered = filtered.sort_values(by=["popularity", "vote_average"], ascending=[False, False])

    total = len(filtered)
    total_pages = max(1, (total + limit - 1) // limit)
    start_idx = max(0, (page - 1) * limit)
    end_idx = start_idx + limit

    sliced = filtered.iloc[start_idx:end_idx].copy()
    cols = [
        "id", "title", "original_title", "overview", "poster_path",
        "release_date", "release_year", "vote_average", "vote_count",
        "runtime", "genres", "original_language", "popularity"
    ]
    available_cols = [c for c in cols if c in sliced.columns]
    records = sliced[available_cols].copy()
    if "poster_path" in records.columns:
        records["poster_path"] = records["poster_path"].apply(format_poster)

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": total_pages,
        "results": records.fillna("").to_dict(orient="records")
    }


def get_genres_summary() -> List[Dict[str, Any]]:
    """Returns all unique genres and their movie counts from the dataset."""
    bundle = load_model_bundle()
    if bundle is None:
        return []
    df = bundle["df"]
    genre_counts = {}
    for g_str in df["genres"].dropna():
        for g in str(g_str).split(","):
            name = g.strip()
            if name:
                genre_counts[name] = genre_counts.get(name, 0) + 1
    return [{"name": k, "count": v} for k, v in sorted(genre_counts.items(), key=lambda x: -x[1])]


# Pre-load model on module import for instantaneous endpoint latency
load_model_bundle()

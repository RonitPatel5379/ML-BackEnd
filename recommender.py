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
        rec_df["poster_path"] = rec_df["poster_path"].apply(
            lambda x: f"https://image.tmdb.org/t/p/w500{x}"
            if isinstance(x, str) and x.startswith("/")
            else (x if isinstance(x, str) else "")
        )
    recommendations = rec_df.fillna("").to_dict(orient="records")

    queried_movie = df.loc[movie_index, available_cols].to_dict()
    if isinstance(queried_movie.get("poster_path"), str) and queried_movie["poster_path"].startswith("/"):
        queried_movie["poster_path"] = f"https://image.tmdb.org/t/p/w500{queried_movie['poster_path']}"

    # Replace any NaN values in queried_movie
    for k, v in queried_movie.items():
        if pd.isna(v):
            queried_movie[k] = ""

    return {
        "movie": queried_movie,
        "cluster": cluster,
        "recommendations": recommendations,
    }


def search_movies(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Searches for movies by title prefix or substring for autocomplete."""
    bundle = load_model_bundle()
    if bundle is None:
        return []
    df = bundle["df"]
    q = query.strip().lower()
    matches = df[df["title"].astype(str).str.lower().str.contains(q, regex=False, na=False)].head(limit)
    cols = ["id", "title", "release_year", "poster_path", "vote_average"]
    available_cols = [c for c in cols if c in df.columns]
    results = matches[available_cols].copy()
    if "poster_path" in results.columns:
        results["poster_path"] = results["poster_path"].apply(
            lambda x: f"https://image.tmdb.org/t/p/w500{x}"
            if isinstance(x, str) and x.startswith("/")
            else (x if isinstance(x, str) else "")
        )
    return results.fillna("").to_dict(orient="records")


# Pre-load model on module import for instantaneous endpoint latency
load_model_bundle()

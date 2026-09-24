import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from Movies import Movie, MovieRequest, RecommendationResponse, HealthResponse
from recommender import recommend_movie, is_model_loaded, get_model_metadata, search_movies

# Load environment variables from .env if present
load_dotenv()

app = FastAPI(
    title="Cineverse Hub Recommendation API",
    description="Machine Learning movie recommendation backend powered by TF-IDF, KMeans Clustering, and Euclidean Distance similarity.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS origins from environment or default to permissive for development
cors_origins_env = os.getenv("CORS_ORIGINS", "*")
if cors_origins_env.strip() == "*":
    origins = ["*"]
else:
    origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True if origins != ["*"] else False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
def index() -> Dict[str, Any]:
    """Root endpoint providing service status and quick links."""
    return {
        "service": "Cineverse Hub Recommendation API",
        "status": "online",
        "model_ready": is_model_loaded(),
        "documentation": "/docs",
        "health": "/health",
        "endpoints": {
            "predict": "POST /predict",
            "search": "GET /search?q={query}",
            "health": "GET /health"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    """Health check endpoint reporting API and ML model status."""
    meta = get_model_metadata()
    return {
        "status": "healthy" if meta.get("loaded") else "degraded",
        "model_loaded": meta.get("loaded", False),
        "total_movies": meta.get("total_movies", 0),
        "n_clusters": meta.get("n_clusters")
    }


@app.post("/predict", response_model=RecommendationResponse, tags=["Recommendation"])
def predict_movies(data: MovieRequest):
    """
    Given a movie title, finds its cluster and returns the top N most similar movies.
    """
    title = data.title.strip()
    n = data.n

    if not title:
        raise HTTPException(status_code=400, detail="Movie title cannot be empty.")

    if not (1 <= n <= 50):
        raise HTTPException(status_code=400, detail="Parameter 'n' must be between 1 and 50.")

    try:
        result = recommend_movie(title, n)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Movie '{title}' was not found in the dataset. Try searching via /search?q={title}"
        )

    return result


@app.get("/search", tags=["Search"])
def search(q: str = Query(..., min_length=1, description="Search term for movie title"), limit: int = Query(10, ge=1, le=50)):
    """Search for movie titles for frontend autocomplete."""
    results = search_movies(q, limit=limit)
    return {"query": q, "count": len(results), "results": results}


@app.get("/welcome/{name}", tags=["General"])
def welcome_user(name: str):
    """Greeting endpoint."""
    return {"Welcome To Cineverse Hub": name}
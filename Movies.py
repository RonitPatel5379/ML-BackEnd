from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class MovieRequest(BaseModel):
    title: str = Field(
        ...,
        description="The title of the movie to find recommendations for",
        examples=["Inception"]
    )
    n: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of recommendations to return (between 1 and 50)",
        examples=[10]
    )

# Alias for backward compatibility
Movie = MovieRequest

class MovieDetails(BaseModel):
    id: Optional[int] = None
    title: Optional[str] = None
    original_title: Optional[str] = None
    overview: Optional[str] = None
    poster_path: Optional[str] = None
    release_date: Optional[str] = None
    release_year: Optional[int] = None
    vote_average: Optional[float] = None
    vote_count: Optional[int] = None
    runtime: Optional[int] = None
    genres: Optional[str] = None
    original_language: Optional[str] = None
    popularity: Optional[float] = None
    budget: Optional[int] = None
    revenue: Optional[int] = None
    similarity: Optional[int] = None

class RecommendationResponse(BaseModel):
    movie: Dict[str, Any] = Field(..., description="Details of the queried movie")
    cluster: int = Field(..., description="Cluster ID assigned by KMeans")
    recommendations: List[Dict[str, Any]] = Field(..., description="List of recommended movies with similarity scores")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    total_movies: int
    n_clusters: Optional[int]
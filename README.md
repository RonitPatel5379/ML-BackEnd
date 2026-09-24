# 🎬 Cineverse Hub - ML Recommendation Backend

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Git LFS](https://img.shields.io/badge/Git-LFS%20Enabled-orange.svg?logo=git-lfs&logoColor=white)](https://git-lfs.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A high-performance machine learning backend for **Cineverse Hub**, providing content-based and clustering-driven movie recommendations with sub-second response times.

---

## 🌟 Key Features

- **⚡ Fast Inference**: Pre-trained KMeans clustering + Euclidean Distance matrix with cached in-memory loading (< 50ms inference).
- **🎯 Smart Similarity Scoring**: Normalizes Euclidean distance to an intuitive 0–100% similarity score.
- **🖼️ Poster Integration**: Automatically formats TMDB poster paths to full high-resolution image URLs.
- **🔍 Autocomplete & Search**: `/search` endpoint to query 69,000+ movie titles with fuzzy substring search.
- **🌐 Production-Ready CORS**: Fully configurable CORS support for local development (`localhost:3000`, `localhost:5173`) and deployed frontends (e.g. Vercel).
- **📦 Git LFS Ready**: Pre-configured Git Large File Storage for `clustering_model.pkl` (122 MB) and dataset files without hitting GitHub limits.

---

## 🧠 ML Architecture

```
Raw Movie Data (69k+ records)
          │
          ▼
Preprocessing & Feature Engineering
  ├── Numerical features (vote_average, popularity, runtime, etc.) -> StandardScaler
  ├── Categorical features (genres, original_language) -> One-Hot Encoding
  └── Text content (overview + title + keywords) -> TfidfVectorizer
          │
          ▼
Dense Feature Matrix (69,405 x 111)
          │
          ▼
KMeans Clustering (k=14 clusters)
          │
          ▼
Saved Bundle (clustering_model.pkl)
          │
          ▼
FastAPI Service (/predict)
  ├── 1. Locate movie index & features
  ├── 2. Predict cluster ID
  ├── 3. Compute Euclidean distances within cluster
  └── 4. Return top-N nearest movies + similarity %
```

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.10+**
- **Git** & **[Git LFS](https://git-lfs.com)**

Make sure Git LFS is installed before cloning:
```bash
git lfs install
```

### 2. Clone the Repository
```bash
git clone <YOUR_REPO_URL>
cd ML-BackEnd
git lfs pull
```

> **Note**: `git lfs pull` ensures that `clustering_model.pkl` and `final_preprocessed_movies.csv` are fully downloaded from GitHub LFS storage.

### 3. Create & Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Environment Setup (Optional)
Copy the example environment configuration:
```bash
# Windows PowerShell
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

### 6. Run the Application
```bash
python main.py
```
Or directly with Uvicorn:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The server will be running at `http://localhost:8000`.
Interactive Swagger API documentation is available at `http://localhost:8000/docs`.

---

## 📡 API Reference

### 1. Health Check
`GET /health`
```json
{
  "status": "healthy",
  "model_loaded": true,
  "total_movies": 69405,
  "n_clusters": 14
}
```

---

### 2. Get Movie Recommendations
`POST /predict`

**Request Body:**
```json
{
  "title": "Inception",
  "n": 5
}
```

**Response (200 OK):**
```json
{
  "movie": {
    "id": 27205,
    "title": "Inception",
    "original_title": "Inception",
    "overview": "Cobb, a skilled thief who commits corporate espionage...",
    "poster_path": "https://image.tmdb.org/t/p/w500/oYuLEt3zVCKq57qu2F8dT7NIa6f.jpg",
    "release_date": "2010-07-15",
    "release_year": 2010,
    "vote_average": 8.364,
    "vote_count": 34495,
    "runtime": 148,
    "genres": "Action, Science Fiction, Adventure",
    "original_language": "en",
    "popularity": 83.952,
    "budget": 160000000,
    "revenue": 825532764
  },
  "cluster": 13,
  "recommendations": [
    {
      "id": 157336,
      "title": "Interstellar",
      "poster_path": "https://image.tmdb.org/t/p/w500/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg",
      "release_year": 2014,
      "vote_average": 8.417,
      "vote_count": 32571,
      "similarity": 84
    }
  ]
}
```

---

### 3. Search Movies (Autocomplete)
`GET /search?q=Dark Knight&limit=5`

**Response (200 OK):**
```json
{
  "query": "Dark Knight",
  "count": 2,
  "results": [
    {
      "id": 155,
      "title": "The Dark Knight",
      "release_year": 2008,
      "poster_path": "https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7haRef0WH.jpg",
      "vote_average": 8.512
    }
  ]
}
```

---

## 📁 Repository Structure

```
ML-BackEnd/
├── .gitattributes                # Git LFS tracking configuration (*.pkl, *.csv)
├── .gitignore                    # Python, Jupyter, OS, and env ignore rules
├── .env.example                  # Environment configuration template
├── README.md                     # Comprehensive project documentation
├── requirements.txt              # Production Python dependencies
├── app.py                        # FastAPI application & route handlers
├── main.py                       # Application runner & server entrypoint
├── recommender.py                # Pure-Python ML recommendation service
├── Movies.py                     # Pydantic schemas and validation models
├── clustering_model.pkl          # Trained KMeans bundle & feature matrices (LFS)
├── final_preprocessed_movies.csv # Preprocessed dataset of 69k+ movies (LFS)
├── model.ipynb                   # Model training, evaluation & experimentation
└── Data-Cleaning & Pre-Processing.ipynb  # Data exploration & preprocessing
```

---

## 🚢 Deployment

### Render / Railway / Docker
1. Set start command:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port $PORT
   ```
2. Set environment variables:
   - `CORS_ORIGINS`: Comma-separated allowed frontend domains (e.g., `https://your-frontend.vercel.app`)
   - `PORT`: Set automatically by hosting platform
3. Ensure Git LFS is enabled in your host's build configuration so `clustering_model.pkl` is pulled during build:
   - On Render / Railway: Add build command `git lfs pull && pip install -r requirements.txt`.

---

## 📄 License
This project is open-source under the MIT License.
from __future__ import annotations

from itertools import islice
from pathlib import Path
from typing import Iterable, List

from datasets import load_dataset
from PIL import Image

from src.movies.templates import Movie

DS_NAME = "stzhao/movie_posters_100k_controlnet"

POSTERS_DIR = Path("data/posters")
POSTERS_DIR.mkdir(parents=True, exist_ok=True)


def valid_rows(stream: Iterable[dict], min_len: int = 20) -> Iterable[dict]:
    """
    Filter out movies with very short or empty overviews.
    """
    for row in stream:
        overview = (row.get("overview") or "").strip()
        if len(overview) >= min_len:
            yield row


def build_movie_from_row(row: dict) -> Movie:
    """
    Builds Movie, if None occurs in one of metadatas returns (False, None), else (True, Movie)
    """
    movie_id = str(row["id"])
    title = (row.get("title") or "").strip() or "(untitled)"
    genres_raw = row.get("genres") or []
    genres = [g.get("name") for g in genres_raw if isinstance(g, dict) and g.get("name")]

    overview = (row.get("overview") or "").strip()

    img = row.get("image")
    poster_path = None
    if isinstance(img, Image.Image):
        out_path = POSTERS_DIR / f"{movie_id}.jpg"
        if not out_path.exists():
            img.save(out_path, format="JPEG", quality=95)
        poster_path = str(out_path)

    if not (movie_id and title and genres and overview and poster_path):
        return False, None

    return True, Movie(
        movie_id=movie_id,
        title=title,
        genres=genres,
        overview=overview,
        poster_path=poster_path,
    )


def ingest_movies_to_jsonl(output_path: Path, sample_size: int = 3000, min_overview_len: int = 20) -> List[Movie]:
    """
    Load a subset of the movie_posters_100k dataset, filter by overview length,
    save posters to disk, and write a JSONL file with Movie records.

    JSONL schema per line (Movie.model_dump_json()):
      {
        "movie_id": "...",
        "title": "...",
        "genres": ["..."],
        "overview": "...",
        "poster_path": "data/posters/<id>.jpg"
      }
    """
    print(f"[INFO] Loading dataset: {DS_NAME}")
    stream = load_dataset(DS_NAME, split="train", streaming=True)

    filtered = valid_rows(stream, min_len=min_overview_len)
    sample = list(islice(filtered, sample_size))

    print(f"[INFO] Building Movie objects for {len(sample)} rows")
    movies: List[Movie] = []
    for row in sample:
        success, movie = build_movie_from_row(row)
        if success:
            movies.append(movie)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for movie in movies:
            f.write(movie.model_dump_json() + "\n")

    print(f"[DONE] Saved {len(movies)} valid movies to {output_path}")
    return movies


def load_movies_from_jsonl(path: Path) -> List[Movie]:
    """
    Load Movie records from a JSONL file written by ingest_movies_to_jsonl.
    """
    movies: List[Movie] = []
    if not path.exists():
        raise FileNotFoundError(f"Movies JSONL not found at {path}")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            movies.append(Movie.model_validate_json(line))
    return movies
